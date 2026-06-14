
#%%
import os
import json
models = ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]
benchmarks = ['MedQA','MedExpQA','USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3']


for benchmark in benchmarks:
    for model_size in models:
        file_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results/q_square_metric/LEM_medicalNER/{model_size}_{benchmark}_q_square.json"
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        with open(file_path, "r") as f:
            data = json.load(f)
        # print(len(data))
        # valid_records = data
        valid_records = [
    item for item in data
    if all(isinstance(item.get(field), (int, float)) for field in ["background_recall", "ques_recall"])]
        # valid_records = [d for d in data if "background_recall" in d and "ques_recall" in d]
        total = len(valid_records)
        # print(total)
        print(f"{model_size}_{benchmark} has {total} valid records.")
        acc_count = sum(1 for d in valid_records if d.get("acc") is True)
        acc_ratio = round(acc_count / total,4)
        print(f"{model_size}_{benchmark} accuracy is: {acc_ratio}")
        avg_b = round(sum(d["background_recall"] for d in valid_records) / total,4)
        avg_q = round(sum(d["ques_recall"] for d in valid_records) / total,4)
        print(f"EntQA_b average is {avg_b}.")
        print(f"EntQA_q average is {avg_q}.")
        print("---------")
        print("\n")

# %%
