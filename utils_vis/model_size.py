#%%
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr

#%%
def corr_with_model_trend(df, metrics):
    results = []
    for metric in metrics:
        x = df["model_size"].values.reshape(-1, 1)
        y = df[metric].values

        # Kendall (秩相关)
        kendall_corr, kendall_p = kendalltau(df["model_size"], y)

        # Spearman (单调趋势)
        spearman_corr, spearman_p = spearmanr(df["model_size"], y)

        # Pearson (线性相关)
        pearson_corr, pearson_p = pearsonr(df["model_size"], y)

        results.append({
            "metric": metric,
            "pearson_corr": round(pearson_corr, 3),
            "pearson_p": round(pearson_p, 3),
            "spearman_corr": round(spearman_corr, 3),
            "spearman_p": round(spearman_p, 3),
            "kendall_corr": round(kendall_corr,3),
            "kendall_p": round(kendall_p, 3)
        })

    results_df = pd.DataFrame(results)
    print(results_df.sort_values("spearman_corr", ascending=False))
    return results_df.sort_values("spearman_corr", ascending=False)
#%%
# Visualization
def plot_line_trend_grouped_model_sizes(filtered_df, metric_list, metric_list_name, benchmark):#, save_path):
    # 把数据 melt 成长格式
    melted = filtered_df.melt(
        id_vars=["model_size"],
        value_vars=metric_list,
        var_name="metric",
        value_name="value"
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value", "model_size"])

    # 映射模型大小到你定义的 group
    def map_model_group(size):
        if size < 5:
            return "<3B"
        elif size <= 13:
            return "~10B"
        elif size <= 40:
            return "32B"
        else:
            return "72B"

    melted["model_group"] = melted["model_size"].apply(map_model_group)

    # 对每个 model_group 和 metric 求平均
    grouped = melted.groupby(["model_group", "metric"])["value"].mean().reset_index()

    # 对每个 metric 做 group 内的归一化
    # 1. MixMax Normalize
    # grouped["normalized_value"] = grouped.groupby("metric")["value"].transform(
    #     lambda x: (x - x.min()) / (x.max() - x.min()))
    # 2. Robust Normalization -> better
    grouped["normalized_value"] = grouped.groupby("metric")["value"].transform(
    lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)))
    # 3. 百分比基线法
    # grouped["normalized_value"] = grouped.groupby("metric")["value"].transform(
    # lambda x: (x - x.iloc[0]) / x.iloc[0] * 100)
    # 4. z-score
    # grouped["normalized_value"] = grouped.groupby("metric")["value"].transform(
    # lambda x: (x - x.mean()) / x.std())


    # 确保横轴顺序
    group_order = ["<3B", "~10B", "32B", "72B"]
    grouped["model_group"] = pd.Categorical(grouped["model_group"], categories=group_order, ordered=True)

    # 绘图
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=grouped,
        x="model_group", y="normalized_value", hue="metric", marker="o"
    )

    plt.title(f"{metric_list_name} Metric Trend (Grouped & Normalized) vs Model Size of Qwen 2.5 Model, {benchmark}")
    plt.xlabel("Model Size Group")
    plt.ylabel("Normalized Score (0-1)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f"{metric_list_name}_line_plot_w_model_size_trend.png"))
    plt.show()
#%%
def analyze_corr_w_trend(df_filtered,metrics, benchmark,arg_name):
    corr_results = corr_with_model_trend(df_filtered, metrics)
    # save_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/Model_size_exploration/{arg_name}/{benchmark}"
    # os.makedirs(save_path, exist_ok=True)
    # corr_results.to_csv(os.path.join(save_path, f"Qwen25_{benchmark}_metric_model_size_trend_full_stat.csv"))
    # Visualization
    my_metrics_list = ['background_recall','ques_recall']
    baseline_list = ['bleu1','bleu4','rouge1','rouge2','rougeL','meteor','bertscore_f1']
    plot_line_trend_grouped_model_sizes(df_filtered, my_metrics_list, "My_Metrics", benchmark)#, save_path)
    plot_line_trend_grouped_model_sizes(df_filtered, baseline_list, "Baseline_Metrics", benchmark)#, save_path)
    return corr_results
#%%
# df = pd.read_csv("/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/LEM_biomedicalNER/MedExpQA/Qwen_MedExpQA_all_summary.csv")
metrics = ['background_recall','ques_recall','bleu1','bleu4','rouge1','rouge2','rougeL','meteor','bertscore_f1']
# result_df = corr_with_model_trend(df, metrics)
# # %%
# for arg_name in ['GTE_medicalNER','LEM_medicalNER','GTE_biomedicalNER','LEM_biomedicalNER']:
arg_name = 'LEM_biomedicalNER'
for benchmark in ['MedQA','MedExpQA',"USMLE_STEP_1","USMLE_STEP_2","USMLE_STEP_3"]:
    df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/Qwen_{benchmark}_all_summary.csv")
    corr_results = analyze_corr_w_trend(df, metrics, benchmark, arg_name)
# %%
