import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from extraction import extract_explanation
import argparse
from q_square_revision import analyze_medical_response
from sentence_transformers import SentenceTransformer
from transformers import AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
import torch

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # 不打断 tqdm
            self.flush()
        except Exception:
            self.handleError(record)
            
now = datetime.now()
date_dir = now.strftime("%m_%d") # e.g., "06_15"
log_filename = now.strftime("%m%d%S") + ".log" # e.g., "061510.log"

log_dir = f"/projectnb/vkolagrp/yiliu/QA_pipeline/metric_comparision/logs/{date_dir}/my_metric"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, log_filename)

# 清空现有 handler
root_logger = logging.getLogger()
root_logger.handlers = []

# 设置新的 handler
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(tqdm_handler)

logging.info(f"Logging saved path: {log_path}")


def main(model, benchmark, emb_model, ner_pipeline, arg_name, emb_threshold=0.7, token_overlap_threshold=0.3):
    # extracted_data = extract_explanation(model, benchmark)
    json_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/{model}/extracted_{model}_{benchmark}.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)
    
    # Getting accuracy 
    total = len(extracted_data)
    correct = sum(1 for item in extracted_data if item.get("acc") is True)
    logging.info(f"Total records in {model}_{benchmark}: {len(extracted_data)}")
    accuracy = correct / total if total > 0 else 0.0
    logging.info(f"Ablation Study | parameter: emb_threshold: {emb_threshold}, token_overlap_threshold: {token_overlap_threshold}")
    logging.info(f"[{model} | {benchmark}] Accuracy: {accuracy:.4f} ({correct}/{total})")
    

    for idx, item in enumerate(tqdm(extracted_data, desc="Analyzing responses", position=2, leave=False)):
        try:
            analysis_result = analyze_medical_response(item["background"], item["question"], item["explanation"], emb_model, ner_pipeline, emb_threshold, token_overlap_threshold)
            item.update(analysis_result)
        except Exception as e:
            logging.warning(f"[Record #{idx}] Error in analysis: {e}")
            continue
    logging.info("Finish generating q_square score for all records.")
    # Save extracted response with q square score
    save_dir = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Results_rebuttal/q_square_metric/{arg_name}/{emb_threshold}_{token_overlap_threshold}"
    os.makedirs(save_dir, exist_ok=True)
    extracted_data_path = os.path.join(save_dir, f"{model}_{benchmark}_q_square.json")
    
    # save the extracted_data updated with q square results
    with open(extracted_data_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    valid_all = [
    item for item in extracted_data
    if all(isinstance(item.get(field), (int, float)) for field in ["background_recall", "ques_recall"])]
    logging.info(f"There are {len(valid_all)} records with q_square_score.")

    def compute_avg(field, valid_all):
        return sum(item[field] for item in valid_all) / len(valid_all) if valid_all else 0.0

    avg_scores_1 = {
    "EntQA_b": compute_avg("background_recall", extracted_data),
    "EntQA_q": compute_avg("ques_recall", extracted_data)
    }
    avg_scores_2 = {
    "EntQA_b": compute_avg("background_recall", valid_all),
    "EntQA_q": compute_avg("ques_recall", valid_all),
    }

    logging.info(
        f"In total data: [{model} | {benchmark}] " +
        ", ".join([f"Avg {k}: {v:.4f}" for k, v in avg_scores_1.items()]) +
        f" (on {len(extracted_data)} total records)"
    )

    logging.info(
        f"In valid data:[{model} | {benchmark}] " +
        ", ".join([f"Avg {k}: {v:.4f}" for k, v in avg_scores_2.items()]) +
        f" (on {len(valid_all)} fully valid records)"
    )
    return extracted_data

# Set device and cache_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cache_dir = "/projectnb/vkolagrp/yiliu/.cache"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Possible embedding model:"Salesforce/SFR-Embedding-Mistral", c
    parser.add_argument("--emb_name", type = str, default = "Linq-AI-Research/Linq-Embed-Mistral")
    # possible NER model: "blaze999/Medical-NER"
    parser.add_argument("--ner_name", type = str, default = 'd4data/biomedical-ner-all')
    # parser.add_argument("--qg_name", type=str, default="lmqg/t5-base-squad-qg")
    parser.add_argument("--emb_threshold", type = float, default = 0.7)
    parser.add_argument("--token_overlap_threshold", type = float, default = 0.3)

    args = parser.parse_args()
    # Setting Embedding Model
    logging.info(f"Using Embedding Model: {args.emb_name}")
    logging.info(f"Using NER Model: {args.ner_name}")

    # Setting Embedding Model
    emb_model = SentenceTransformer(args.emb_name, cache_folder = cache_dir, device = device)

    # Setting NER Model
    ner_tokenizer = AutoTokenizer.from_pretrained(args.ner_name, cache_dir = cache_dir,
                                        use_fast = False)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_name, cache_dir = cache_dir, device_map='auto')
    ner_pipeline = pipeline(
        "ner",
        model=args.ner_name,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple")

    models = ['0.5B-Qwen25','1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25',"72B-Qwen25"]
    benchmarks = ['USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3','MedQA','MedExpQA'] #
    arg_name = "LEM_biomedicalNER"
    for model in tqdm(models, desc = "Model List", position=0):
        for benchmark in tqdm(benchmarks, desc = "Benchmark List", position=1):
            result = main(model, benchmark, emb_model, ner_pipeline, arg_name, args.emb_threshold, args.token_overlap_threshold)