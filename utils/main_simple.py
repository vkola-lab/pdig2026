# This file is to calculate different QA relevancy score on different model's response on medical benchmark
#%%

from overlap_metrics import compute_overlapping_metrics
import os
import json
import pandas as pd
from tqdm import tqdm

# Main Logger set up
import logging
from datetime import datetime

main_logger = logging.getLogger("main_logger")
main_logger.setLevel(logging.INFO)

now = datetime.now()
date_dir = now.strftime("%m_%d") # e.g., "06_15"
log_filename = now.strftime("%m%d%S") + "_main_eval.log" # e.g., "061510.log"

log_dir = f"/projectnb/vkolagrp/yiliu/QA_pipeline/metric_comparision/logs/{date_dir}/other_metrics/main"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_filename)

file_handler = logging.FileHandler(log_path)
stream_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

if main_logger.hasHandlers():
    main_logger.handlers.clear()

main_logger.addHandler(file_handler)
main_logger.addHandler(stream_handler)
main_logger.propagate = False

main_logger.info(f"Main Logger: {log_path}")

def process_all(records, save_path = None, save_every = 10):
    temp_results = []
    for idx, item in tqdm(enumerate(records), total=len(records)):
        question = item.get("question", "")
        explanation = item.get("explanation", "")
        prediction = item.get("prediction", "")
        ground_truth = item.get("ground_truth", "")

        try:
            # Overlapping metric
            overlap_scores = compute_overlapping_metrics(question, explanation)

            result = {
                "background": item.get("background", ""),
                "question": question,
                "option": item.get("option", ""),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "acc": prediction.strip().lower() == ground_truth.strip().lower(),
                "explanation": explanation,
                "bleu4" : round(overlap_scores['bleu4'],4),
                "bleu1":  round(overlap_scores['bleu1'],4),
                "meteor": round(overlap_scores['meteor'],4),
                "rouge1": round(overlap_scores["rouge1"],4),
                "rouge2": round(overlap_scores["rouge2"],4),
                "rougeL": round(overlap_scores["rougeL"],4),
                "rougeLsum": round(overlap_scores["rougeLsum"],4),
                "bertscore_f1": round(overlap_scores["bertscore_f1"],4),
            }
            temp_results.append(result)
        except Exception as e:
            main_logger.error(f"[{idx}] Error Processing: {question[:50]}...error:{e}")
            continue
        if (idx + 1) % save_every == 0 and save_path:
            pd.DataFrame(temp_results).to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))
            # main_logger.info(f"Save {idx + 1} records to {save_path}")
            temp_results.clear()  # 清空临时列表，避免重复写入
    if temp_results and save_path:
        pd.DataFrame(temp_results).to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))
        # main_logger.info(f"Saved the final rest of {len(temp_results)} records to {save_path}.")
    return temp_results


# models = ['0.5B-Qwen25','1.5B-Qwen25','2B-Gemma','3B-Qwen25','6B-Yi','7B-DeepSeek','7B-Qwen25','14B-Qwen25','32B-Qwen25','72B-Qwen25']
def compute_other_metrics(model,benchmark = "MedExpQA"):
    json_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/{model}/extracted_{model}_{benchmark}.json"
    assert os.path.exists(json_path), f"JSON 文件不存在：{json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    correct = sum(1 for item in records if item.get("acc") is True)
    total = len(records)
    json_acc_rate = correct / total if total > 0 else 0.0
    main_logger.info(f"[From JSON] Accuracy of {model}_{benchmark}: {json_acc_rate:.3f} ({correct}/{total})")
    
    save_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results/baseline_metric/{model}_{benchmark}_overlap.csv"
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(save_path):
        os.remove(save_path)
    process_all(records, save_path=save_path, save_every=10)
    df = pd.read_csv(save_path)
    acc_rate = df['acc'].mean()
    avg_bleu_4 = df["bleu4"].mean()
    avg_bleu_1 = df["bleu1"].mean()
    avg_meteor = df["meteor"].mean()
    avg_rouge_1 = df["rouge1"].mean()
    avg_rouge_2 = df["rouge2"].mean()
    avg_rougeL = df["rougeL"].mean()
    avg_rougeLsum = df["rougeLsum"].mean()
    avg_bert = df["bertscore_f1"].mean()
    # avg_bert_avg_sent = df["bertscore_f1_avg_sent"].mean()
    # avg_bert_max_sent = df["bertscore_f1_max_sent"].mean()

    main_logger.info(f"Accuracy of {model}_{benchmark}: {acc_rate:.3f}")
    main_logger.info(f"Avg BLEU_1 of {model}_{benchmark}: {avg_bleu_1:.3f}")
    main_logger.info(f"Avg BLEU_4 of {model}_{benchmark}: {avg_bleu_4:.3f}")
    main_logger.info(f"Avg METEOR of {model}_{benchmark}: {avg_meteor:.3f}")
    main_logger.info(f"Avg ROUGE_1 of {model}_{benchmark}: {avg_rouge_1:.3f}")
    main_logger.info(f"Avg ROUGE_2 of {model}_{benchmark}: {avg_rouge_2:.3f}")
    main_logger.info(f"Avg ROUGE_L of {model}_{benchmark}: {avg_rougeL:.3f}")
    main_logger.info(f"Avg ROUGE_LSum of {model}_{benchmark}: {avg_rougeLsum:.3f}")
    main_logger.info(f"Avg BERTScore of {model}_{benchmark}: {avg_bert:.3f}")
    # main_logger.info(f"Avg BERTScore for sentence average of {model}_{benchmark}: {avg_bert_avg_sent:.3f}")
    # main_logger.info(f"Avg BERTScore for the max sentence of {model}_{benchmark}: {avg_bert_max_sent:.3f}")


if __name__ == "__main__":
    models =  ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]
    benchmarks = ['USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3','MedQA','MedExpQA']
    for model in tqdm(models, desc = "Processing models", position=0):
        for benchmark in tqdm(benchmarks, desc = "Benchmark List", position=1):
            # for benchmark in tqdm(benchmarks, desc = "Processing benchmarks"):
            try:
                main_logger.info(f"Processing responses of {model}.")
                compute_other_metrics(model, benchmark)
                main_logger.info(f"Finishing evaluation of {model}'s responses")
            except Exception as e:
                main_logger.error(f"[{model} | {benchmark}] Error: {e}")

#%%
