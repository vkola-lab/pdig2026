import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm



# 设定模型名
model_1 = "OpenBioLLM-70B"
model_2 = "PodGPT-70B"
benchmarks =['MedExpQA','MedQA', 'USMLE_STEP_1', 'USMLE_STEP_2', 'USMLE_STEP_3']



def combine_results(model_1, model_2, benchmark):
    # 设定文件路径
    path_1 = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/Previous_data/results/{model_1}_{benchmark}_q_square.csv"
    path_2 = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/Previous_data/results/{model_2}_{benchmark}_q_square.csv"
    # 读取数据
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)

    # 用于合并的 key 列
    key_cols = ["background", "question", "option", "ground_truth"]

    # 除了 key 外要加 prefix 的列
    value_cols = [
        "prediction", "acc", "explanation",
        "background_ent_coverage", "ques_ent_coverage",
        "uncovered_background_ent", "uncovered_question_ent",
        "max_sim_question", "question_sim_score"
    ]

    # 重命名列名，加上模型前缀
    df1_renamed = df1[key_cols + value_cols].copy()
    df1_renamed = df1_renamed.rename(columns={col: f"{model_1}_{col}" for col in value_cols})

    df2_renamed = df2[key_cols + value_cols].copy()
    df2_renamed = df2_renamed.rename(columns={col: f"{model_2}_{col}" for col in value_cols})

    # 合并两个数据表
    df_merged = pd.merge(df1_renamed, df2_renamed, on=key_cols, how="inner")

    # 保存合并后的文件
    output_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/Previous_data/results/combined_results/{model_1}_{model_2}_{benchmark}_q_square.csv"
    df_merged.to_csv(output_path, index=False)

for benchmark in tqdm(benchmarks, desc = "Processing benchmark"):
    combine_results(model_1, model_2, benchmark)
