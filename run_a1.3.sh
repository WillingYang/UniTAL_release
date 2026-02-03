
#!/bin/bash

echo "start training"

#清空旧结果
#> results.txt

# 跑10次
for i in $(seq 0 2)
do
    echo "Run $i"
    CUDA_VISIBLE_DEVICES=$1 python train_eval.py ./configs/anet_i3d.yaml --output output_50 --n $i
done

# 最后调用 Python 脚本做统计
python summarize_results.py 
