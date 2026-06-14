#%%
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr, pointbiserialr
from scipy.stats import ttest_rel
##%

# Does metric corrs with acc in group level (if group by model_size first)
def compute_corr_with_acc(df, metric_list, benchmark, arg_name, df_type = "merged"):
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
    save_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/accuracy_alignment/{arg_name}/{benchmark}"
    os.makedirs(save_path, exist_ok= True)
    prefix = f"Qwen_{benchmark}_{df_type}_avg_metric"
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
        corr_result_df, corr_result_sig_df = compute_corr_with_acc(df, metric_list, benchmark,arg_name, df_type = "merged")
        # print("Compute model-level correlation")
        corr_result_df['benchmark'] = benchmark
        corr_result_sig_df['benchmark'] = benchmark
        all_results.append(corr_result_df)
        all_sig_results.append(corr_result_sig_df)

        corr_result_df_filtered, corr_result_sig_df_filtered = compute_corr_with_acc(df, metric_list, benchmark,arg_name, df_type = "filtered")
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

# %%
# All-right & all-wrong 

def plot_metric_boxplots(all_right_df, all_wrong_df, metric_list, list_name, arg_name, benchmark):
    all_right_df = all_right_df.copy()
    all_wrong_df = all_wrong_df.copy()
    all_right_df['group'] = 'all_right'
    all_wrong_df['group'] = 'all_wrong'
    combined_df = pd.concat([all_right_df, all_wrong_df])

    n_metrics = len(metric_list)
    if n_metrics <= 3:
        n_rows, n_cols = 1, n_metrics
    else:
        n_cols = min(4, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=False)
    axes = axes.flatten()

    for i, metric in enumerate(metric_list):
        ax = axes[i]
        sns.boxplot(data=combined_df, x='group', y=metric, ax=ax, palette="Set2")

        # t-test (By model)
        try:
            right_mean = all_right_df.groupby("model")[metric].mean()
            wrong_mean = all_wrong_df.groupby("model")[metric].mean()
            merged = pd.merge(right_mean, wrong_mean, on="model", suffixes=("_right", "_wrong"))
            if len(merged) >= 2:
                t_stat, p_val = ttest_rel(merged[f"{metric}_right"], merged[f"{metric}_wrong"])
                mean_diff = merged[f"{metric}_right"].mean() - merged[f"{metric}_wrong"].mean()
            else:
                t_stat, p_val, mean_diff = None, None, None
        except:
            t_stat, p_val, mean_diff = None, None, None

        if p_val is None:
            sig = "n/a"
            title = f"{metric}\n(p=n/a)"
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        if p_val is not None:
            title = f"{metric}\n(p={p_val:.3g}, {sig}, Δ={mean_diff:.3f})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(metric)

    for j in range(len(metric_list), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{benchmark}: All-right vs All-wrong", fontsize=16, x=0.5, y=1.02, ha="center")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/accuracy_alignment/{arg_name}/{benchmark}/{list_name}_all_right_wrong_avg_metric_comparision.png")
    plt.show()

def compare_all_right_wrong_box(benchmark, arg_name):
    all_right_df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/all_right_Qwen_{benchmark}_records.csv")
    all_wrong_df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/all_wrong_Qwen_{benchmark}_records.csv")
    print(f"{arg_name}: There are {all_right_df.question.nunique()} all_right question; {all_wrong_df.question.nunique()} all_wrong question in {benchmark}")
    my_metrics = ['background_recall','ques_recall'] #,'background_sim_score','question_sim_score']
    baseline_metrics = ['bleu1','bleu4','rouge1','rouge2','rougeL','meteor','bertscore_f1']
    plot_metric_boxplots(all_right_df, all_wrong_df, my_metrics, "my_metric", arg_name, benchmark)
    plot_metric_boxplots(all_right_df, all_wrong_df, baseline_metrics, "baseline_metric",arg_name, benchmark)

# %%
# Pointbiserial Test
def pointbiserial_test_between_acc_metric(arg_name, benchmark):
    df = pd.read_csv(f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/processed_data/{arg_name}/{benchmark}/Qwen_{benchmark}_all_summary.csv", index_col = 0)
    df["acc"] = df["acc"].astype(int)
    metrics = [
    'background_recall', 'ques_recall', 
    'bleu4', 'bleu1', 'meteor', 'rouge1', 'rouge2',
    'rougeL', 'rougeLsum', 'bertscore_f1']
    results = []
    for metric in metrics:
        r, p = pointbiserialr(df["acc"], df[metric])
        results.append({
            "metric": metric,
            "pointbiserial_corr": round(r,4),
            "p-value": round(p,4)
        })
    save_path = "/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/results/pointbiserial_test"
    os.makedirs(save_path, exist_ok= True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_path, f"corr_results_{arg_name}_{benchmark}.csv"))
    return pd.DataFrame(results)
         
