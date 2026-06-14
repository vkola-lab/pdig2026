#%%
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# All Data Manipulation function
# Step 1: merge all model responses into one df 
def merge_all_response(benchmark, arg_name): 
    qwen_model_list = ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]
    metrics = ["background_recall","ques_recall"]
    combined_results = defaultdict(dict)

    long_records = []
    for model_name in qwen_model_list:
        path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results/q_square_metric/{arg_name}/{model_name}_{benchmark}_q_square.json"
        with open(path, 'r') as f:
            records = json.load(f)
            for r in records:
                # print("Keys:", r.keys())
                long_records.append({
                    "background": r["background"],
                    "question": r["question"],
                    "model": model_name,
                    "explanation": r.get("explanation", ""),
                    "acc": r["acc"],
                    **{m: r.get(m, None) for m in metrics}
                })
    df = pd.DataFrame(long_records)
    print(f"There are {df['background'].nunique()} background & question in benchmark {benchmark}; the total df shape is {df.shape}")
    return df 

def merge_all_baseline_metrics(benchmark):
    qwen_model_list = [
        '0.5B-Qwen25', '1.5B-Qwen25', '3B-Qwen25',
        '7B-Qwen25', '14B-Qwen25', '32B-Qwen25', '72B-Qwen25'
    ]
    metric_list = [
        "bleu4", "bleu1", "meteor", "rouge1", "rouge2",
        "rougeL", "rougeLsum", "bertscore_f1"
    ]
    
    all_records = []
    for model in qwen_model_list:
        path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results/baseline_metric/{model}_{benchmark}_overlap.csv"
        df = pd.read_csv(path)
        
        if 'background' not in df.columns or 'question' not in df.columns:
            print(f"❗️Missing 'background' or 'question' columns in {path}")
            continue
        
        for _, row in df.iterrows():
            record = {
                "background": row["background"],
                "question": row["question"],
                "explanation": row["explanation"],
                "model": model
            }
            for metric in metric_list:
                record[metric] = row.get(metric, None)
            all_records.append(record)
    
    baseline_df = pd.DataFrame(all_records)
    print(f"✅ Merge completed: {baseline_df['background'].nunique()} questions; baseline metric df shape is {baseline_df.shape}")
    return baseline_df

# %%
# # Filter all right, all wrong question/background
def extract_size(model_name: str) -> float:
    """
    Extract size, e.g, 0.5B, 1.5B, 72B 
    """
    try:
        size_str = model_name.split("B")[0]  # "0.5", "72"
        return float(size_str)
    except:
        return float("inf")

def filter_and_merge_df(arg_name, benchmark):
    print("Arg_name: ",arg_name)
    print("---------------------")
    df = merge_all_response(benchmark, arg_name)
    df_baseline = merge_all_baseline_metrics(benchmark)
    # merge my metrics, baseline metric
    print(f"The original metrics' shape: {df.shape}; the baseline metrics' shape: {df_baseline.shape}.")
    merge_df = df.merge(df_baseline,  on=["background", "question", "model","explanation"], how="left")
    merge_df['model_size'] = merge_df['model'].apply(extract_size)
    print("The shape of merged df: ", merge_df.shape)
    filtered_records = []
    all_right_records = []
    all_wrong_records = []

    grouped = merge_df.groupby(['background', 'question'])
    for (b, q), group in grouped:
        acc_list = group['acc'].tolist()
        if all(a == 1 for a in acc_list):
            all_right_records.append((b, q))
        elif all(a == 0 for a in acc_list):
            all_wrong_records.append((b, q))
        else:
            filtered_records.append((b, q))

    print(f"✔️ Records all models answer right: {len(all_right_records)}")
    print(f"❌ Records all models answer wrong: {len(all_wrong_records)}")
    print(f"🟡 Mixed wrong Records: {len(filtered_records)}")

    def is_in(record_list):
        record_set = set(record_list)
        return lambda row: (row['background'], row['question']) in record_set

    all_right_df = merge_df[merge_df.apply(is_in(all_right_records), axis=1)]
    all_wrong_df = merge_df[merge_df.apply(is_in(all_wrong_records), axis=1)]
    filtered_df  = merge_df[merge_df.apply(is_in(filtered_records), axis=1)]
    path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}"
    os.makedirs(path, exist_ok= True)
    print("Benchmark: ", benchmark)
    print(f"The shape of all-right-df is {all_right_df.shape}; all-wrong-df is {all_wrong_df.shape}; the rest record is {filtered_df.shape}")
    merge_df.to_csv(os.path.join(path, f"Qwen_{benchmark}_all_summary.csv"))
    all_right_df.to_csv(os.path.join(path,f"all_right_Qwen_{benchmark}_records.csv"))
    all_wrong_df.to_csv(os.path.join(path,f"all_wrong_Qwen_{benchmark}_records.csv"))
    filtered_df.to_csv(os.path.join(path,f"Qwen_{benchmark}_filtered_summary.csv"))
    return merge_df, all_right_df, all_wrong_df, filtered_df
# %%
arg_names =['GTE_medicalNER','LEM_medicalNER','GTE_biomedicalNER','LEM_biomedicalNER']
for arg_name in arg_names:
    for benchmark in ['USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3','MedQA', 'MedExpQA']:
        filter_and_merge_df(arg_name, benchmark)
