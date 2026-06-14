#%%
import os
from main_process import filter_and_merge_df
from main_accuracy_align import get_avg_metric_corr_w_acc, pointbiserial_test_between_acc_metric, compare_all_right_wrong_box
# from model_size import 

# %%
# data concatenation, store in Paper-writing/processed_data
# merge_df, all_right_df, all_wrong_df, filtered_df = filter_and_merge_df("LEM_biomedicalNER","MedQA")
# %%
metric_list = ['acc','background_recall','ques_recall', 
        'bleu4', 'bleu1', 'meteor', 'rouge1', 'rouge2',
        'rougeL', 'rougeLsum', 'bertscore_f1']
arg_names =['GTE_medicalNER','LEM_medicalNER','GTE_biomedicalNER','LEM_biomedicalNER']
for arg_name in arg_names:
        final_corr_df, final_sig_corr_df, final_corr_df_f, final_sig_corr_df_f = get_avg_metric_corr_w_acc(metric_list,arg_name)
        save_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Paper_writing/accuracy_alignment/{arg_name}/integrate"    
        os.makedirs(save_path, exist_ok = True)
        final_corr_df.to_csv(os.path.join(save_path, "final_corr_df.csv"))
        final_sig_corr_df.to_csv(os.path.join(save_path, "final_sig_corr_df.csv"))
        final_corr_df_f.to_csv(os.path.join(save_path, "filtered_final_corr_df.csv"))
        final_sig_corr_df_f.to_csv(os.path.join(save_path, "filtered_final_sig_corr_df.csv"))
# %%
# Question-level correlation in the benchmark
# arg_name = "LEM_biomedicalNER"
# corr_df = pointbiserial_test_between_acc_metric(arg_name,"MedQA")
# %%
# Compare all-right vs all-wrong, whether all-right > all-wrong
compare_all_right_wrong_box("MedExpQA","LEM_biomedicalNER")

# %%
# Model size

