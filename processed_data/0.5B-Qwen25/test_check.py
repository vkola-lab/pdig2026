#%%
import json

# 文件路径
path = "/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/0.5B-Qwen25/extracted_0.5B-Qwen25_MedExpQA.json"

# 读取 JSON 文件
with open(path, "r") as f:
    data = json.load(f)  # data 是 list of dict

# 统计预测准确率（完全匹配）
correct = 0
total = 0

for item in data:
    prediction = item.get("prediction", "").strip().lower()
    ground_truth = item.get("ground_truth", "").strip().lower()
    
    if prediction == ground_truth:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

# %%
