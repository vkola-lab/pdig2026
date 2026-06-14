#%%
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from scipy.stats import spearmanr, kendalltau, pearsonr, pointbiserialr

#%%
# Table 3
def corr_with_model_trend(df, metrics, arg_name):
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
    save_path = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/results/model_scale"
    os.makedirs(save_path, exist_ok=True)
    results_df.sort_values("spearman_corr", ascending=False).to_csv(os.path.join(save_path, f"{arg_name}_model_scale.csv"), index=False)
    # print(results_df.sort_values("spearman_corr", ascending=False))
    return results_df.sort_values("spearman_corr", ascending=False)

    
#%%
# Does metric corrs with acc in group level (if group by model_size first)
def compute_corr_with_acc(df, metric_list, benchmark, arg_name):#, df_type = "merged"):
    # average metric by model_size
    avg_metric_by_model_size = df.groupby(['model'])[metric_list].mean().reset_index()
    results = []

    for metric in metric_list:
        x = avg_metric_by_model_size['acc']
        y = avg_metric_by_model_size[metric]

        pearson_corr, p_pearson = pearsonr(x, y)
        spearman_corr, p_spearman = spearmanr(x, y)
        kendall_corr, p_kendall = kendalltau(x, y)

        results.append({
            "metric": metric,
            "pearson": round(pearson_corr, 4),
            "p_pearson": round(p_pearson, 4),
            "spearman": round(spearman_corr, 4),
            "p_spearman": round(p_spearman, 4),
            "kendall": round(kendall_corr, 4),
            "p_kendall": round(p_kendall, 4),
        })
    result_df = pd.DataFrame(results)
    result_sig_df = result_df.loc[(result_df['p_pearson']< 0.05)|(result_df['p_spearman']< 0.05)|(result_df['p_kendall']< 0.05)]
    save_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/results/accuracy_alignment/{arg_name}/{benchmark}"
    os.makedirs(save_path, exist_ok= True)
    # prefix = f"Qwen_{benchmark}_{df_type}_avg_metric"
    prefix = f"Qwen_{benchmark}_avg_metric"
    result_df.to_csv(os.path.join(save_path, f"{prefix}_corr_w_acc.csv"))
    result_sig_df.to_csv(os.path.join(save_path, f"{prefix}_sig_corr_w_acc.csv"))
    return result_df, result_sig_df

def get_avg_metric_corr_w_acc(metric_list,arg_name):
    all_results = []
    all_sig_results = []
    all_results_f = []
    all_sig_results_f = []
    for benchmark in ['MedQA','MedExpQA','USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3']:
        print("Processing benchmark: ", benchmark)
        df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/Qwen_{benchmark}_all_summary.csv", index_col = 0)
        df_filtered = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/Qwen_{benchmark}_filtered_summary.csv")
        # print("Success loading df.")
        corr_result_df, corr_result_sig_df = compute_corr_with_acc(df, metric_list, benchmark,arg_name)
        # print("Compute model-level correlation")
        corr_result_df['benchmark'] = benchmark
        corr_result_sig_df['benchmark'] = benchmark
        all_results.append(corr_result_df)
        all_sig_results.append(corr_result_sig_df)

        corr_result_df_filtered, corr_result_sig_df_filtered = compute_corr_with_acc(df_filtered, metric_list, benchmark, arg_name)
        corr_result_df_filtered['benchmark'] = benchmark
        corr_result_sig_df_filtered['benchmark'] = benchmark
        all_results_f.append(corr_result_df_filtered)
        all_sig_results_f.append(corr_result_sig_df_filtered)
    final_corr_df = pd.concat(all_results, ignore_index=True)
    final_sig_corr_df = pd.concat(all_sig_results, ignore_index=True)
    final_corr_df_f = pd.concat(all_results_f, ignore_index=True)
    final_sig_corr_df_f = pd.concat(all_sig_results_f, ignore_index=True)
    final_corr_df = final_corr_df.sort_values(by=["benchmark", "spearman"], ascending=[True, False])
    final_sig_corr_df = final_sig_corr_df.sort_values(by=["benchmark", "spearman"], ascending=[True, False])

    # 排序 filtered
    final_corr_df_f = final_corr_df_f.sort_values(by=["benchmark", "spearman"], ascending=[True, False])
    final_sig_corr_df_f = final_sig_corr_df_f.sort_values(by=["benchmark", "spearman"], ascending=[True, False])
        
    return final_corr_df, final_sig_corr_df, final_corr_df_f, final_sig_corr_df_f

#%%
# Pointbiserial Test
def pointbiserial_test_between_acc_metric(df, arg_name, benchmark, metrics=None):
    # df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/Qwen_{benchmark}_all_summary.csv", index_col = 0)
    df["acc"] = df["acc"].astype(int)
    if metrics is None:
        metrics = ['background_recall', 'ques_recall',
                   'bleu4', 'bleu1', 'meteor', 'rouge1', 'rouge2',
                   'rougeL', 'rougeLsum', 'bertscore_f1']
    results = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        r, p = pointbiserialr(df["acc"], df[metric])
        results.append({
            "metric": metric,
            "pointbiserial_corr": round(r,4),
            "p-value": round(p,4)
        })
    save_path = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/results/pointbiserial_test"
    os.makedirs(save_path, exist_ok= True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_path, f"corr_results_{arg_name}_{benchmark}.csv"))
    return pd.DataFrame(results)