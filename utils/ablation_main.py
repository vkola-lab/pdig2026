#%%
import os
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from q_square_revision import extract_entities, compute_ent_recall
from correlation_function import compute_corr_with_acc, corr_with_model_trend, pointbiserial_test_between_acc_metric
#%%
# Logging information
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


now = datetime.now()
date_dir    = now.strftime("%m_%d")
log_filename = now.strftime("%m%d%S") + ".log"

log_dir  = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/logs/{date_dir}"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, log_filename)

root_logger = logging.getLogger()
root_logger.handlers = []
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(tqdm_handler)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "/projectnb/vkolagrp/yiliu/.cache"

MODELS = ['0.5B-Qwen25', '1.5B-Qwen25', '3B-Qwen25', '7B-Qwen25', '14B-Qwen25', '32B-Qwen25', '72B-Qwen25']
MODEL_SIZES = {m: float(m.split("B")[0]) for m in MODELS}

EMB_THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
TOK_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#%%
# Source: original extracted JSONs (same as main.py reads)
EXTRACTED_DIR = "/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data"
# Output: one subfolder per (emb_t, tok_t)
JSON_OUT_ROOT = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/q_square_metric"
# pkl cache for NER results
PRECOMPUTE_DIR = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/ablation/precomputed"
# Correlation results and plots
RESULT_DIR = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/ablation/results"
PLOT_DIR   = "/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/ablation/plots"


# ---------------------------------------------------------------------------
# Step 1: NER (run once per record, save to pkl)
# ---------------------------------------------------------------------------
#%%
def precompute_ents(benchmark, ner_pipeline, emb_model, force=False):
    """
    Read extracted_{model}_{benchmark}.json for all models.
    Run extract_entities (NER) once per record.
    Save pkl: list of {original_item, model, model_size, background_ents,
                       question_ents, response_ents, dropped_*}
    """
    save_path = os.path.join(PRECOMPUTE_DIR, f"{benchmark}_ents.pkl")
    os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

    if os.path.exists(save_path) and not force:
        logging.info(f"Loading precomputed entities from {save_path}")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    logging.info(f"Running NER for benchmark={benchmark} ...")
    # print(f"Running NER for benchmark={benchmark} ...")
    all_records = []

    for model in tqdm(MODELS, desc = "Model reponses json"):
        json_path = os.path.join(EXTRACTED_DIR, model, f"extracted_{model}_{benchmark}.json")
        if not os.path.exists(json_path):
            logging.warning(f"Missing: {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
            logging.info("Load response json!")

        for idx, item in enumerate(tqdm(extracted_data, desc=f"{model}", leave=False)):
            try:
                ent_dict = extract_entities(
                    item["background"], item["question"], item["explanation"], ner_pipeline
                )
            except Exception as e:
                logging.warning(f"[{model} #{idx}] NER error: {e}")
                ent_dict = {
                    "background_ents": [], "question_ents": [], "response_ents": [],
                    "dropped_background_ents": [], "dropped_question_ents": [], "dropped_response_ents": [],
                }
            all_records.append({
                "original": item,          # full original record
                "model": model,
                "model_size": MODEL_SIZES[model],
                **ent_dict,                # background_ents, question_ents, response_ents, dropped_*
            })

    # Encode all unique entity strings once
    all_ent_strings = set()
    for rec in all_records:
        all_ent_strings.update(rec["background_ents"] + rec["question_ents"] + rec["response_ents"])
    all_ent_list = list(all_ent_strings)
    logging.info(f"Encoding {len(all_ent_list)} unique entities ...")
    embeddings = emb_model.encode(all_ent_list, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    emb_lookup = {ent: emb for ent, emb in zip(all_ent_list, embeddings)}

    def get_embs(ents):
        vecs = [emb_lookup[e] for e in ents if e in emb_lookup]
        return np.array(vecs) if vecs else np.array([])

    for rec in all_records:
        rec["b_embs"] = get_embs(rec["background_ents"])
        rec["q_embs"] = get_embs(rec["question_ents"])
        rec["r_embs"] = get_embs(rec["response_ents"])
    
    with open(save_path, "wb") as f:
        pickle.dump(all_records, f)
    logging.info(f"Saved NER + embeddings to {save_path}  ({len(all_records)} records)")
    return all_records

#%%
# ---------------------------------------------------------------------------
# Step 2: sweep thresholds — save JSON + compute correlations
# ---------------------------------------------------------------------------

def sweep_thresholds(all_records, benchmark, base_arg_name="LEM_biomedicalNER"):
    """
    For each (emb_t, tok_t):
      - Run compute_ent_recall (embedding + threshold) for every record
      - Save full JSON (all fields from original + ent_dict + recall_dict)
        to JSON_OUT_ROOT/{base_arg_name}_emb{e}_tok{t}/{model}_{benchmark}_q_square.json
      - Compute 3 correlations from in-memory records
    Returns DataFrame with one row per (emb_t, tok_t) and all correlation values.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    corr_metrics = ["background_recall", "ques_recall"]
    results = []

    for emb_t, tok_t in tqdm(
        list(itertools.product(EMB_THRESHOLDS, TOK_THRESHOLDS)),
        desc="Sweeping thresholds"
    ):
        arg_name = f"{base_arg_name}_emb{emb_t}_tok{tok_t}"
        out_dir = os.path.join(JSON_OUT_ROOT, arg_name)
        os.makedirs(out_dir, exist_ok=True)

        # Group by model for JSON saving
        by_model = {}
        df_rows = []

        for rec in all_records:
            try:
                recall_dict = compute_ent_recall(
                    rec["background_ents"], rec["question_ents"], rec["response_ents"],
                    emb_threshold=emb_t, token_overlap_threshold=tok_t,
                    b_embs=rec["b_embs"] if len(rec["b_embs"]) > 0 else None,
                    q_embs=rec["q_embs"] if len(rec["q_embs"]) > 0 else None,
                    r_embs=rec["r_embs"] if len(rec["r_embs"]) > 0 else None,
                )
            except Exception as e:
                logging.warning(f"[{rec['model']}] compute_ent_recall error: {e}")
                recall_dict = {
                    "background_recall": None, "ques_recall": None,
                    "background_ent_matched": {}, "question_ent_matched": {},
                    "covered_background_ent": [], "covered_question_ent": [],
                    "uncovered_background_ent": [], "uncovered_question_ent": [],
                }

            # Build full output record (original fields + entity fields + recall fields)
            out_item = dict(rec["original"])
            out_item.update({
                "background_ents":          rec["background_ents"],
                "question_ents":            rec["question_ents"],
                "response_ents":            rec["response_ents"],
                "dropped_background_ents":  rec["dropped_background_ents"],
                "dropped_question_ents":    rec["dropped_question_ents"],
                "dropped_response_ents":    rec["dropped_response_ents"],
            })
            out_item.update(recall_dict)

            by_model.setdefault(rec["model"], []).append(out_item)

            df_rows.append({
                "model":              rec["model"],
                "model_size":         rec["model_size"],
                "acc":                int(rec["original"]["acc"]),
                "background_recall":  recall_dict["background_recall"],
                "ques_recall":        recall_dict["ques_recall"],
            })

        # Save JSON files (one per model)
        for model, records in by_model.items():
            out_path = os.path.join(out_dir, f"{model}_{benchmark}_q_square.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSONs → {out_dir}")

        # Compute 3 correlations
        df = pd.DataFrame(df_rows)
        corr_row = {"emb_t": emb_t, "tok_t": tok_t}

        # Corr1: model-level correlation with accuracy
        corr1_df, _ = compute_corr_with_acc(df, ["acc"] + corr_metrics, benchmark, arg_name=arg_name)
        for m in corr_metrics:
            r = corr1_df[corr1_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr1_pearson"]    = r["pearson"]
            corr_row[f"{m}_corr1_p_pearson"]  = r["p_pearson"]
            corr_row[f"{m}_corr1_spearman"]   = r["spearman"]
            corr_row[f"{m}_corr1_p_spearman"] = r["p_spearman"]
            corr_row[f"{m}_corr1_kendall"]    = r["kendall"]
            corr_row[f"{m}_corr1_p_kendall"]  = r["p_kendall"]

        # Corr2: question-level point-biserial
        corr2_df = pointbiserial_test_between_acc_metric(
            df, arg_name=arg_name, benchmark=benchmark, metrics=corr_metrics
        )
        for m in corr_metrics:
            r = corr2_df[corr2_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr2_pb"]   = r["pointbiserial_corr"]
            corr_row[f"{m}_corr2_p_pb"] = r["p-value"]

        # Corr3: correlation with model size trend (question-level, replicates model_size.py)
        corr3_df = corr_with_model_trend(df, corr_metrics, arg_name)
        for m in corr_metrics:
            r = corr3_df[corr3_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr3_pearson"]    = r["pearson_corr"]
            corr_row[f"{m}_corr3_p_pearson"]  = r["pearson_p"]
            corr_row[f"{m}_corr3_spearman"]   = r["spearman_corr"]
            corr_row[f"{m}_corr3_p_spearman"] = r["spearman_p"]
            corr_row[f"{m}_corr3_kendall"]    = r["kendall_corr"]
            corr_row[f"{m}_corr3_p_kendall"]  = r["kendall_p"]

        results.append(corr_row)
        logging.info(
            f"emb={emb_t}, tok={tok_t} | "
            f"b_recall C1_sp={corr_row['background_recall_corr1_spearman']:.3f} "
            f"C2_pb={corr_row['background_recall_corr2_pb']:.3f} "
            f"C3_sp={corr_row['background_recall_corr3_spearman']:.3f}"
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(RESULT_DIR, f"{benchmark}_ablation.csv"), index=False)
    logging.info(f"Saved correlation results → {RESULT_DIR}/{benchmark}_ablation.csv")
    return result_df

#%%
# ---------------------------------------------------------------------------
# Step 2b: extend existing ablation CSV with new thresholds
# ---------------------------------------------------------------------------
def extend_thresholds(all_records, benchmark, new_tok_thresholds=[0, 0.1],
                      new_emb_thresholds=None, base_arg_name="LEM_biomedicalNER"):
    """
    Run sweep only for new (emb_t, tok_t) combinations not already in the CSV,
    then append the new rows and re-save.
    new_emb_thresholds: if None, uses the global EMB_THRESHOLDS.
    """
    emb_list = new_emb_thresholds if new_emb_thresholds is not None else EMB_THRESHOLDS
    corr_metrics = ["background_recall", "ques_recall"]
    csv_path = os.path.join(RESULT_DIR, f"{benchmark}_ablation.csv")
    result_df = pd.read_csv(csv_path)

    existing = set(zip(result_df["emb_t"].round(4), result_df["tok_t"].round(4)))
    new_pairs = [
        (e, t) for e, t in itertools.product(emb_list, new_tok_thresholds)
        if (round(e, 4), round(t, 4)) not in existing
    ]
    if not new_pairs:
        logging.info("No new (emb_t, tok_t) pairs to add.")
        return result_df

    logging.info(f"Extending with {len(new_pairs)} new pairs: {new_pairs}")
    new_rows = []

    for emb_t, tok_t in tqdm(new_pairs, desc="Extending thresholds"):
        arg_name = f"{base_arg_name}_emb{emb_t}_tok{tok_t}"
        out_dir = os.path.join(JSON_OUT_ROOT, arg_name)
        os.makedirs(out_dir, exist_ok=True)

        by_model = {}
        df_rows = []

        for rec in all_records:
            try:
                recall_dict = compute_ent_recall(
                    rec["background_ents"], rec["question_ents"], rec["response_ents"],
                    emb_threshold=emb_t, token_overlap_threshold=tok_t,
                    b_embs=rec["b_embs"] if len(rec["b_embs"]) > 0 else None,
                    q_embs=rec["q_embs"] if len(rec["q_embs"]) > 0 else None,
                    r_embs=rec["r_embs"] if len(rec["r_embs"]) > 0 else None,
                )
            except Exception as e:
                logging.warning(f"[{rec['model']}] compute_ent_recall error: {e}")
                recall_dict = {
                    "background_recall": None, "ques_recall": None,
                    "background_ent_matched": {}, "question_ent_matched": {},
                    "covered_background_ent": [], "covered_question_ent": [],
                    "uncovered_background_ent": [], "uncovered_question_ent": [],
                }

            out_item = dict(rec["original"])
            out_item.update({
                "background_ents":         rec["background_ents"],
                "question_ents":           rec["question_ents"],
                "response_ents":           rec["response_ents"],
                "dropped_background_ents": rec["dropped_background_ents"],
                "dropped_question_ents":   rec["dropped_question_ents"],
                "dropped_response_ents":   rec["dropped_response_ents"],
            })
            out_item.update(recall_dict)
            by_model.setdefault(rec["model"], []).append(out_item)
            df_rows.append({
                "model":             rec["model"],
                "model_size":        rec["model_size"],
                "acc":               int(rec["original"]["acc"]),
                "background_recall": recall_dict["background_recall"],
                "ques_recall":       recall_dict["ques_recall"],
            })

        for model, records in by_model.items():
            out_path = os.path.join(out_dir, f"{model}_{benchmark}_q_square.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved JSONs → {out_dir}")

        df = pd.DataFrame(df_rows)
        corr_row = {"emb_t": emb_t, "tok_t": tok_t}

        corr1_df, _ = compute_corr_with_acc(df, ["acc"] + corr_metrics, benchmark, arg_name=arg_name)
        for m in corr_metrics:
            r = corr1_df[corr1_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr1_pearson"]    = r["pearson"]
            corr_row[f"{m}_corr1_p_pearson"]  = r["p_pearson"]
            corr_row[f"{m}_corr1_spearman"]   = r["spearman"]
            corr_row[f"{m}_corr1_p_spearman"] = r["p_spearman"]
            corr_row[f"{m}_corr1_kendall"]    = r["kendall"]
            corr_row[f"{m}_corr1_p_kendall"]  = r["p_kendall"]

        corr2_df = pointbiserial_test_between_acc_metric(
            df, arg_name=arg_name, benchmark=benchmark, metrics=corr_metrics
        )
        for m in corr_metrics:
            r = corr2_df[corr2_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr2_pb"]   = r["pointbiserial_corr"]
            corr_row[f"{m}_corr2_p_pb"] = r["p-value"]

        corr3_df = corr_with_model_trend(df, corr_metrics, arg_name)
        for m in corr_metrics:
            r = corr3_df[corr3_df["metric"] == m].iloc[0]
            corr_row[f"{m}_corr3_pearson"]    = r["pearson_corr"]
            corr_row[f"{m}_corr3_p_pearson"]  = r["pearson_p"]
            corr_row[f"{m}_corr3_spearman"]   = r["spearman_corr"]
            corr_row[f"{m}_corr3_p_spearman"] = r["spearman_p"]
            corr_row[f"{m}_corr3_kendall"]    = r["kendall_corr"]
            corr_row[f"{m}_corr3_p_kendall"]  = r["kendall_p"]

        new_rows.append(corr_row)
        logging.info(
            f"emb={emb_t}, tok={tok_t} | "
            f"b_recall C1_sp={corr_row['background_recall_corr1_spearman']:.3f} "
            f"C2_pb={corr_row['background_recall_corr2_pb']:.3f} "
            f"C3_sp={corr_row['background_recall_corr3_spearman']:.3f}"
        )

    result_df = pd.concat([result_df, pd.DataFrame(new_rows)], ignore_index=True)
    result_df = result_df.sort_values(["emb_t", "tok_t"]).reset_index(drop=True)
    result_df.to_csv(csv_path, index=False)
    logging.info(f"Extended CSV saved → {csv_path}  (now {len(result_df)} rows)")
    return result_df


#%%

# Step 2c: update corr3 only (patch existing ablation CSV)
# ---------------------------------------------------------------------------
def update_corr3(all_records, benchmark, base_arg_name="LEM_biomedicalNER"):
    """
    Re-compute corr3 with the fixed question-level approach and overwrite
    only the corr3 columns in the existing {benchmark}_ablation.csv.
    Does NOT re-run NER/embedding or touch corr1/corr2.
    """
    corr_metrics = ["background_recall", "ques_recall"]
    csv_path = os.path.join(RESULT_DIR, f"{benchmark}_ablation.csv")
    result_df = pd.read_csv(csv_path)

    for emb_t, tok_t in tqdm(
        list(itertools.product(EMB_THRESHOLDS, TOK_THRESHOLDS)),
        desc="Updating corr3"
    ):
        arg_name = f"{base_arg_name}_emb{emb_t}_tok{tok_t}"
        df_rows = []

        for rec in all_records:
            try:
                recall_dict = compute_ent_recall(
                    rec["background_ents"], rec["question_ents"], rec["response_ents"],
                    emb_threshold=emb_t, token_overlap_threshold=tok_t,
                    b_embs=rec["b_embs"] if len(rec["b_embs"]) > 0 else None,
                    q_embs=rec["q_embs"] if len(rec["q_embs"]) > 0 else None,
                    r_embs=rec["r_embs"] if len(rec["r_embs"]) > 0 else None,
                )
            except Exception as e:
                logging.warning(f"[{rec['model']}] compute_ent_recall error: {e}")
                recall_dict = {"background_recall": None, "ques_recall": None}

            df_rows.append({
                "model":             rec["model"],
                "model_size":        rec["model_size"],
                "background_recall": recall_dict["background_recall"],
                "ques_recall":       recall_dict["ques_recall"],
            })

        df = pd.DataFrame(df_rows)
        corr3_df = corr_with_model_trend(df, corr_metrics, arg_name)

        mask = (result_df["emb_t"] == emb_t) & (result_df["tok_t"] == tok_t)
        for m in corr_metrics:
            r = corr3_df[corr3_df["metric"] == m].iloc[0]
            result_df.loc[mask, f"{m}_corr3_pearson"]  = r["pearson_corr"]
            result_df.loc[mask, f"{m}_corr3_spearman"] = r["spearman_corr"]
            result_df.loc[mask, f"{m}_corr3_kendall"]  = r["kendall_corr"]

        logging.info(
            f"emb={emb_t}, tok={tok_t} | "
            f"C3_sp={result_df.loc[mask, 'background_recall_corr3_spearman'].values[0]:.3f}"
        )

    result_df.to_csv(csv_path, index=False)
    logging.info(f"Updated corr3 → {csv_path}")
    return result_df

#%%
# Step 2c: patch p-values into existing ablation CSV (no NER/embedding re-run)
# ---------------------------------------------------------------------------
def patch_pvalues(all_records, benchmark, base_arg_name="LEM_biomedicalNER"):
    """
    Read the existing {benchmark}_ablation.csv, re-compute p-values from the
    precomputed pkl, and write them back as new columns.
    Does NOT re-run NER/embedding.
    """
    corr_metrics = ["background_recall", "ques_recall"]
    csv_path = os.path.join(RESULT_DIR, f"{benchmark}_ablation.csv")
    result_df = pd.read_csv(csv_path)

    for emb_t, tok_t in tqdm(
        list(itertools.product(EMB_THRESHOLDS, TOK_THRESHOLDS)),
        desc="Patching p-values"
    ):
        arg_name = f"{base_arg_name}_emb{emb_t}_tok{tok_t}"
        df_rows = []

        for rec in all_records:
            try:
                recall_dict = compute_ent_recall(
                    rec["background_ents"], rec["question_ents"], rec["response_ents"],
                    emb_threshold=emb_t, token_overlap_threshold=tok_t,
                    b_embs=rec["b_embs"] if len(rec["b_embs"]) > 0 else None,
                    q_embs=rec["q_embs"] if len(rec["q_embs"]) > 0 else None,
                    r_embs=rec["r_embs"] if len(rec["r_embs"]) > 0 else None,
                )
            except Exception as e:
                logging.warning(f"[{rec['model']}] compute_ent_recall error: {e}")
                recall_dict = {"background_recall": None, "ques_recall": None}

            df_rows.append({
                "model":             rec["model"],
                "model_size":        rec["model_size"],
                "acc":               int(rec["original"]["acc"]),
                "background_recall": recall_dict["background_recall"],
                "ques_recall":       recall_dict["ques_recall"],
            })

        df = pd.DataFrame(df_rows)
        mask = (result_df["emb_t"] == emb_t) & (result_df["tok_t"] == tok_t)

        corr1_df, _ = compute_corr_with_acc(df, ["acc"] + corr_metrics, benchmark, arg_name=arg_name)
        for m in corr_metrics:
            r = corr1_df[corr1_df["metric"] == m].iloc[0]
            result_df.loc[mask, f"{m}_corr1_p_pearson"]  = r["p_pearson"]
            result_df.loc[mask, f"{m}_corr1_p_spearman"] = r["p_spearman"]
            result_df.loc[mask, f"{m}_corr1_p_kendall"]  = r["p_kendall"]

        corr2_df = pointbiserial_test_between_acc_metric(df, arg_name=arg_name, benchmark=benchmark, metrics=corr_metrics)
        for m in corr_metrics:
            r = corr2_df[corr2_df["metric"] == m].iloc[0]
            result_df.loc[mask, f"{m}_corr2_p_pb"] = r["p-value"]

        corr3_df = corr_with_model_trend(df, corr_metrics, arg_name)
        for m in corr_metrics:
            r = corr3_df[corr3_df["metric"] == m].iloc[0]
            result_df.loc[mask, f"{m}_corr3_p_pearson"]  = r["pearson_p"]
            result_df.loc[mask, f"{m}_corr3_p_spearman"] = r["spearman_p"]
            result_df.loc[mask, f"{m}_corr3_p_kendall"]  = r["kendall_p"]

    result_df.to_csv(csv_path, index=False)
    logging.info(f"Patched p-values → {csv_path}")
    return result_df


#%%
# ---------------------------------------------------------------------------
# Step 3: heatmap
# ---------------------------------------------------------------------------

def _sig_stars(p):
    if pd.isna(p):   return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return ""


def plot_heatmaps(result_df, benchmark):
    corr_configs = [
        ("background_recall_corr1_spearman", "background_recall_corr1_p_spearman",
         "ques_recall_corr1_spearman",        "ques_recall_corr1_p_spearman",
         "Corr1: Model-level × Acc (Spearman)"),
        ("background_recall_corr2_pb",        "background_recall_corr2_p_pb",
         "ques_recall_corr2_pb",              "ques_recall_corr2_p_pb",
         "Corr2: Question-level Point-biserial"),
        ("background_recall_corr3_spearman", "background_recall_corr3_p_spearman",
         "ques_recall_corr3_spearman",        "ques_recall_corr3_p_spearman",
         "Corr3: Model Size Trend (Spearman)"),
    ]
    metric_labels = ["EntQA_b", "EntQA_q"]

    has_pval = all(col in result_df.columns for col in ["background_recall_corr1_p_spearman"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for row_idx, cfg in enumerate(corr_configs):
        col_b, pcol_b, col_q, pcol_q, row_title = cfg
        for col_idx, (col_name, pcol_name, metric_label) in enumerate(
            zip([col_b, col_q], [pcol_b, pcol_q], metric_labels)
        ):
            ax = axes[row_idx][col_idx]
            pivot = result_df.pivot(index="tok_t", columns="emb_t", values=col_name)

            if has_pval and pcol_name in result_df.columns:
                pivot_p = result_df.pivot(index="tok_t", columns="emb_t", values=pcol_name)
                annot = pivot.copy().astype(object)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.iloc[i, j]
                        p = pivot_p.iloc[i, j]
                        annot.iloc[i, j] = f"{v:.3f}{_sig_stars(p)}"
            else:
                annot = True

            fmt = "" if has_pval and pcol_name in result_df.columns else ".3f"
            sns.heatmap(
                pivot, ax=ax, annot=annot, fmt=fmt,
                cmap="RdYlGn", vmin=-1, vmax=1,
                linewidths=0.5, cbar=(col_idx == 1)
            )
            ax.set_title(row_title, fontsize=12, pad=22)
            ax.text(0.5, 1.0, metric_label, transform=ax.transAxes,
                    fontsize=12, ha="center", va="bottom")
            ax.set_xlabel("emb_threshold", fontsize=11.5)
            ax.set_ylabel("tok_threshold", fontsize=11.5)

    fig.suptitle(f"Ablation: {benchmark}\n* p<0.05  ** p<0.01  *** p<0.001", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, f"{benchmark}_heatmap.png"), bbox_inches="tight", dpi=150)
    plt.show()
    logging.info(f"Saved heatmap → {PLOT_DIR}/{benchmark}_heatmap.png")

#%%
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",        type=str, default="MedQA")
    parser.add_argument("--emb_name",         type=str, default="Linq-AI-Research/Linq-Embed-Mistral")
    parser.add_argument("--ner_name",         type=str, default="d4data/biomedical-ner-all")
    parser.add_argument("--force_precompute", action="store_true")
    parser.add_argument("--update_corr3",     action="store_true",
                        help="Only re-compute corr3 and patch existing ablation CSV. No model loading needed.")
    parser.add_argument("--patch_pvalues",    action="store_true",
                        help="Add p-value columns to existing ablation CSV and re-draw heatmap. No model loading needed.")
    parser.add_argument("--extend_thresholds", action="store_true",
                        help="Add tok_t=0,0.1 rows to existing ablation CSV. No model loading needed.")
    args = parser.parse_args()

    if args.extend_thresholds:
        pkl_path = os.path.join(PRECOMPUTE_DIR, f"{args.benchmark}_ents.pkl")
        logging.info(f"Loading precomputed entities from {pkl_path}")
        with open(pkl_path, "rb") as f:
            all_records = pickle.load(f)
        result_df = extend_thresholds(all_records, args.benchmark, new_tok_thresholds=[0, 0.1])
        plot_heatmaps(result_df, args.benchmark)
    elif args.patch_pvalues:
        pkl_path = os.path.join(PRECOMPUTE_DIR, f"{args.benchmark}_ents.pkl")
        logging.info(f"Loading precomputed entities from {pkl_path}")
        with open(pkl_path, "rb") as f:
            all_records = pickle.load(f)
        result_df = patch_pvalues(all_records, args.benchmark)
        plot_heatmaps(result_df, args.benchmark)
    elif args.update_corr3:
        # Load pkl directly — no NER/embedding model needed
        pkl_path = os.path.join(PRECOMPUTE_DIR, f"{args.benchmark}_ents.pkl")
        logging.info(f"Loading precomputed entities from {pkl_path}")
        with open(pkl_path, "rb") as f:
            all_records = pickle.load(f)
        result_df = update_corr3(all_records, args.benchmark)
        plot_heatmaps(result_df, args.benchmark)
    else:
        logging.info(f"Using Embedding Model: {args.emb_name}")
        logging.info(f"Using NER Model: {args.ner_name}")
        ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_name, cache_dir=cache_dir, device_map={"":0})
        ner_tokenizer = AutoTokenizer.from_pretrained(args.ner_name, cache_dir=cache_dir, use_fast=False)
        ner_pipeline = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple")
        emb_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
        emb_model = SentenceTransformer(args.emb_name, cache_folder=cache_dir, device=emb_device)
        all_records = precompute_ents(args.benchmark, ner_pipeline, emb_model, force=args.force_precompute)
        result_df   = sweep_thresholds(all_records, args.benchmark)
        plot_heatmaps(result_df, args.benchmark)

# %%
# Adujst heatmap
# for benchmark in ['MedExpQA','USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3']:
# benchmark = 'MedExpQA'
# result_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/Result_rebuttal/ablation/results/{benchmark}_ablation.csv"
# result_df = pd.read_csv(result_path)
# plot_heatmaps(result_df, benchmark)
# %%
