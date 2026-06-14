#%%
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Step 1: merge all model responses into one df 
def merge_all_response(benchmark): 
    qwen_model_list = ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]
    metrics = [ "background_recall", "ques_recall"]
    combined_results = defaultdict(dict)

    long_records = []
    for model_name in qwen_model_list:
        path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results/q_square_metric/{model_name}_{benchmark}_q_square.json"
        with open(path, 'r') as f:
            records = json.load(f)
            for r in records:
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

# %%
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
    提取模型大小，支持 0.5B, 1.5B, 72B 这种格式
    """
    try:
        size_str = model_name.split("B")[0]  # "0.5", "72"
        return float(size_str)
    except:
        return float("inf")

#%%
## Task 3: In filtered data; does bigger model always better than smaller model? Does acc_seq aligns with metric_seq?
from scipy.stats import spearmanr
summary_df = pd.read_csv("/projectnb/vkolagrp/yiliu/QA_pipeline/utils_vis/Qwen_summary_comparision.csv",index_col = 0)

metrics = [
    "background_precision", "background_recall", "background_F1",
    "ques_precision", "ques_recall", "ques_F1", "background_sim_score","question_sim_score"
]
correlation_results = []
grouped = summary_df.groupby(["background", "question"])
#%%
for (bg, q), group in grouped:
    if group["acc"].nunique() < 2 or group[metric].nunique() < 2:
        continue
    for metric in metrics:
        try:
            corr, _ = spearmanr(group["acc"], group[metric])
            correlation_results.append({
                "background": bg,
                "question": q,
                "metric": metric,
                "spearman_corr": corr
            })
        except:
            continue

# 转为 DataFrame 并展示部分结果
corr_df = pd.DataFrame(correlation_results)


#%%
# Visualization 1: Does metrics change along the model size?
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
for metric in all_metrics:
    plt.plot(Qwen25_MedExpQA_model_size_avg_metric['model_size'], Qwen25_MedExpQA_model_size_avg_metric[metric], marker='o', label=metric)

plt.xlabel('Model Size (B)')
plt.ylabel('Score')
plt.title('Metric Trend vs Model Size of Qwen 2.5 Model, MedExpQA')
plt.grid(True)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig("/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/Model_size_exploration/Qwen25_MedExpQA_metric_model_size_trend.png", bbox_inches='tight')
plt.show()
#%%
# Stat test & model_size
from scipy.stats import pearsonr, spearmanr, kendalltau

results = []
for metric in all_metrics:
    x = Qwen25_MedExpQA_filtered_df['model_size']
    y = Qwen25_MedExpQA_filtered_df[metric]
    pearson_corr, p_p = pearsonr(x, y)
    spearman_corr, p_s = spearmanr(x, y)
    kendall_corr, p_k = kendalltau(x, y)

    results.append({
        'metric': metric,
        'pearson_r': round(pearson_corr, 4),
        'pearson_p': round(p_p, 4),
        'spearman_r': round(spearman_corr, 4),
        'spearman_p': round(p_s, 4),
        'kendall_r': round(kendall_corr, 4),
        'kendall_p': round(p_k, 4)
    })
results_df = pd.DataFrame(results)
results_df.to_csv("/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/Model_size_exploration/Qwen25_MedExpQA_metric_model_size_trend_full_stat.csv")
# Filter all significant correlation
results_df_sig = results_df.loc[(results_df['pearson_p']<0.05)|(results_df['spearman_p']<0.05)| (results_df['kendall_p']<0.05)]
results_df_sig.to_csv("/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/Model_size_exploration/Qwen25_MedExpQA_metric_model_size_trend_sig_stat.csv")
#%%
# Visualization 2:
Qwen25_MedExpQA_acc_avg_metric = Qwen25_MedExpQA_filtered_df.groupby(['model_size','acc'])[all_metrics].mean().reset_index()

from statannotations.Annotator import Annotator
from scipy.stats import ttest_ind
qwen_model_list = ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]

def plot_acc_boxplot(Qwen25_MedExpQA_filtered_df, metric):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=Qwen25_MedExpQA_filtered_df, x='model', y=metric, hue='acc')
    plt.xticks(rotation=45)
    plt.title(f'{metric} by Model and Accuracy')

    # 设置每个模型 acc=True vs False 的显著性对比
    pairs = [((model, True), (model, False)) for model in qwen_model_list]

    annotator = Annotator(ax, pairs, data=Qwen25_MedExpQA_filtered_df, x='model', y=metric, hue='acc')
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=1)
    annotator.apply_and_annotate()

    plt.tight_layout()
    plt.show()

for metric in all_metrics:
    plot_acc_boxplot(Qwen25_MedExpQA_filtered_df, metric)

#%%
# Main


# %%
