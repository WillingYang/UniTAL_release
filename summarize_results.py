import ast

mAPs = []
tIoUs = []

with open("anet_results.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            # 将字符串解析为元组对象
            mAP, tiou = ast.literal_eval(line)
            mAPs.append(mAP)
            tIoUs.append(tiou)
        except Exception as e:
            print("解析失败：", line, e)

# 统计平均值
import numpy as np

mAPs = np.array(mAPs)                          # [N]
tIoUs = np.array(tIoUs)                        # [N, t]

mean_mAP = mAPs.mean()
mean_tiou = tIoUs.mean(axis=0)

print(f"📊 Averaged Results over {len(mAPs)} runs:")
print(f"mean(max_mAP): {mean_mAP:.2f}")
print(f"mean(tIoU_mAPs): {[round(x, 2) for x in mean_tiou.tolist()]}")
