import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D, LayerNorm)

@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 0, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        topk_ratio=0.1,        # topk ratio of salient attentive mask 
        threshold=10,          # threshold of salient attentive mask 
     
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe  
        self.topk_ratio = topk_ratio
        self.threshold = threshold
        self.dim = n_embd
        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                    in_channels, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(  
                    n_embd, n_head, max_len,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    topk_ratio=self.topk_ratio,
                    threshold=self.threshold
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(TransformerBlock(
                    n_embd, n_head, max_len,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1+idx],
                    use_rel_pe=self.use_rel_pe,
                    topk_ratio=self.topk_ratio,
                    threshold=self.threshold
                )
            )


       # Text refinement MLP
        self.text_refine = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, vid, txt, txt_fine,  mask):
        B, C, T = vid.size()  
        
        # embedding network
        for idx in range(len(self.embd)):
            vid, mask = self.embd[idx](vid, mask)
            vid = self.relu(self.embd_norm[idx](vid))
        #vid = self.avgpool(vid) 
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            vid = vid + pe[:, :, :T] * mask.to(vid.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            vid = vid + pe[:, :, :T] * mask.to(vid.dtype)

        # # # 视频分段
        L = txt_fine.size(2)  # L=77
        vid_segments = self.seg_vid(vid, mask, L)  # [B, L, D, T_i]

        # Text-Video Interaction
        alpha = 0.3   # top-k ratio
        k = int(alpha * L)  # e.g., 38
        #refined_text_emb = txt_fine[:,:-1,:,:]  # [B, class_num, L, D]
        refined_text_emb = txt_fine
        # 视频查询：分段均值
        vid_query = vid_segments.mean(dim=3)  # [B, L, D]
        vid_query = vid_query.unsqueeze(1)  # [B, 1, L, D]

        sim = torch.einsum('b n l d, b c l d -> b n l c', vid_query, refined_text_emb)  
        H = torch.softmax(sim / self.logit_scale.exp(), dim=-1)  # [B, 1, L, class_num]
        H = H.squeeze(1)   # [B, L, class_num]
        H = H.transpose(1, 2)  # [B, class_num, L]

        # top-k（按 L 上取 k 个 token）
        _, H_indices = H.topk(k, dim=-1)   # -> [B, class_num, k]

        # 构造用于 gather 的索引： unsqueeze(-1) 再 expand 到 D
        index = H_indices.unsqueeze(-1).expand(-1, -1, -1, refined_text_emb.size(-1))  # [B, class_num, k, D]
        index = index.contiguous()
        selected_tokens = torch.gather(refined_text_emb, dim=2, index=index)  # [B, class_num, k, D]

        # 聚合并精炼
        refined_text_emb = self.text_refine(selected_tokens.mean(dim=2))  # [B, class_num, D]



        txt = refined_text_emb + txt

        # stem transformer
        for idx in range(len(self.stem)):
            vid, mask, txt = self.stem[idx](vid, txt, mask)

        # prep for outputs
        lv_vid = tuple()
        lv_masks = tuple()
        lv_vid += (vid,)
        lv_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            vid, mask, txt = self.branch[idx](vid, txt, mask)
            lv_vid += (vid,)
            lv_masks += (mask,)

        return lv_vid, lv_masks, txt

    def seg_vid(self, vid, mask, L=3):
        # vid: [B, D, T], mask: [B, 1, T]
        batch_size, D, T = vid.shape
        T_i = T // L  # 每段长度
        vid_segments = []
        
        for b in range(batch_size):
            valid_vid = vid[b] * mask[b].squeeze(1)  # [D, T]
            segments = []
            for i in range(L):
                start_idx = i * T_i
                end_idx = min((i + 1) * T_i, T)  # 防止越界
                segment = valid_vid[:, start_idx:end_idx]  # [D, T_i]
                segments.append(segment.unsqueeze(0))  # [1, D, T_i]
            # 拼接为 [L, D, T_i]
            segments = torch.cat(segments, dim=0)  # [L, D, T_i]
            vid_segments.append(segments.unsqueeze(0))  # [1, L, D, T_i]
        
        vid_segments = torch.cat(vid_segments, dim=0)  # [B, L, D, T_i]
        return vid_segments

