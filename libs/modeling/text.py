import torch
import torch.nn as nn
from transformers import CLIPTokenizer  # pip install transformers==4.19.2
from .clip import CLIPModel


def data_split_dir(file, data_split, mode, split_num):
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100-data_split, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100-data_split, r3=split_num)
        return file_dir

class TextFeatures(nn.Module):
    def __init__(
        self,
        model_path,
        subset_file,
        data_split,
        emb_dim,
        split_num,
        freeze_txt_model=True,
    ):
        super().__init__()
        self.train_classes = self._load_classes(subset_file, data_split, 'train', split_num)
        self.test_classes = self._load_classes(subset_file, data_split, 'test', split_num)

        self.cls2desc = self._load_descriptions('/home/ywl/disk1/thumos14_action_descriptions3.txt')

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.txt_model = CLIPModel.from_pretrained(model_path, ignore_mismatched_sizes=True).float()

        if isinstance(self.txt_model.text_projection, nn.Linear):
            self.txt_model.text_projection = nn.Linear(self.txt_model.text_embed_dim, emb_dim, bias=False)
            nn.init.normal_(self.txt_model.text_projection.weight, std=self.txt_model.text_embed_dim ** -0.5)
        else:
            self.txt_model.text_projection = nn.Parameter(torch.empty(self.txt_model.text_embed_dim, emb_dim))
            nn.init.normal_(self.txt_model.text_projection, std=self.txt_model.text_embed_dim ** -0.5)
        self.text_projection = nn.Linear(emb_dim * 2, emb_dim, bias = False)

        if freeze_txt_model:
            self.txt_model.requires_grad_(False)
            self.txt_model.text_projection.requires_grad_(True)

    
    def extract_text_emb(self, cls_name, is_prompt=False):
        if is_prompt:
            train_prompt = self.get_prompt(cls_name)
        else:
            train_prompt = cls_name 
        device = next(iter(self.txt_model.parameters())).device
        texts = self.tokenizer(train_prompt, padding=True, truncation =True,
                                 max_length=77,
                                return_tensors="pt").to(device)

        if hasattr(self.txt_model, 'get_text_features'):
            # 获取未池化的嵌入 (B, class_nums, L, D)
            text_outputs = self.txt_model.text_model(
                input_ids=texts["input_ids"],
                attention_mask=texts["attention_mask"],
                return_dict=True,
                output_hidden_states=True
            )
            # 获取最后一层的隐藏状态 (B, class_nums, L, D)
            hidden_states = self.text_projection(text_outputs.last_hidden_state)
            # Debugging statement to print shapes
            
            
            # 获取池化后的嵌入 (B, class_nums, D)
            pooled_output = self.txt_model.get_text_features(**texts)

        else:
            raise NotImplementedError("The current model does not support extracting text embeddings.")

        return hidden_states, pooled_output
        
    def txt_read(self, file_path, sort=False):
        with open(file_path, 'r') as f:
            cls_name = [cls_name.strip('\n') for cls_name in f.readlines()]

        if sort:
            cls_name = sorted(cls_name)
        
        split_dict = {cls_name: i for i, cls_name in enumerate(cls_name)}

        return split_dict
    
    def _load_descriptions(self, file_path):
        """将txt里的描述读成字典 {class_name: description}"""
        cls2desc = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    cls, desc = line.split(':', 1)
                    cls2desc[cls.strip()] = desc.strip()
        return cls2desc

    def get_prompt(self, cls_names):
        """根据类别名返回对应的描述，如果找不到则用默认提示"""
        prompt_cls_name = []
        for c in cls_names:
            if c in self.cls2desc:
                prompt_cls_name.append(self.cls2desc[c])
            else:
                # fallback: 使用默认模板
                prompt_cls_name.append(f'a video of action {c}')
          
        return prompt_cls_name
    
    def _load_classes(self, subset_file, data_split, mode, split_num):
        file_path = data_split_dir(subset_file, data_split, mode, split_num)
        try:
            with open(file_path, 'r') as f:
                cls_name = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise ValueError(f"No class file: {file_path}")

        return cls_name

    def forward(self, batch_size, mode):
        cls_name = self.train_classes if mode == 'train' else self.test_classes
        cls_name = cls_name + ["no actions"]


        # 获取未池化和池化的嵌入
        text_emb_unpooled, text_emb_pooled = self.extract_text_emb(cls_name, is_prompt=True)
        
        # 如果未池化的嵌入是 3D (B, class_nums, L, D)，需要扩展为 4D
        if len(text_emb_unpooled.size()) == 3:
            text_emb_unpooled = text_emb_unpooled.unsqueeze(0).expand(batch_size, -1, -1, -1)
     
        # 如果池化的嵌入是 2D (B, class_nums, D)，需要扩展为 3D
        if len(text_emb_pooled.size()) == 2:
            text_emb_pooled = text_emb_pooled.unsqueeze(0).expand(batch_size, -1, -1)


        split_num_cls = text_emb_pooled.size(1)

        return text_emb_pooled, text_emb_unpooled, split_num_cls


