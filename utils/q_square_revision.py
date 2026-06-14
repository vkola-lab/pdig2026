#%%
# %cd /projectnb/vkolagrp/yiliu/QA_pipeline
#%%
import spacy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelWithLMHead, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import time
import logging
import os
import re 

logger = logging.getLogger(__name__)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cache_dir = "/projectnb/vkolagrp/yiliu/.cache"
nlp = spacy.load("en_core_web_sm")

os.environ["HF_HOME"] = "/projectnb/vkolagrp/yiliu/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/projectnb/vkolagrp/yiliu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/projectnb/vkolagrp/yiliu/.cache/huggingface/hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/projectnb/vkolagrp/yiliu/.cache/sentence_transformers"
#%%
# embedding model
# Sentence_Transformer guide: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html?utm_source=chatgpt.com
# "Qwen/Qwen3-Embedding-8B"
# "Salesforce/SFR-Embedding-Mistral"
# "Linq-AI-Research/Linq-Embed-Mistral"
# "Alibaba-NLP/gte-Qwen2-7B-instruct"
# emb_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", cache_folder = cache_dir, device = device)
#%%
# NER
# "d4data/biomedical-ner-all", blaze999/Medical-NER; 
# ner_name = "blaze999/Medical-NER"
# ner_tokenizer = AutoTokenizer.from_pretrained(ner_name,cache_dir = cache_dir,
#                                         use_fast = False)
# ner_model = AutoModelForTokenClassification.from_pretrained(ner_name, cache_dir = cache_dir, device_map='auto')
# ner_pipeline = pipeline(
#     "ner",
#     model=ner_model,
#     tokenizer=ner_tokenizer,
#     aggregation_strategy="simple")

#%%
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def is_meaningful_phrase(phrase):
    # 筛除‘#’ 开头的duplication
    if phrase.strip().startswith("#"):
        return False
    # 1. 清洗：小写，去标点
    phrase_clean = phrase.lower().translate(str.maketrans('', '', string.punctuation))
    words = phrase_clean.split()
    if not words:
        return False
    if all(w in stop_words for w in words):
        return False
    # remove ent with low information; Add more not-important entities
    low_info_phrases = {
        "the following", "other options", "the physician", "the office", "the patient",'class', "come", "comes", "brought", 'this time'
    }
    if phrase_clean in low_info_phrases:
        return False
    # 单个非停用词、非低信息词默认保留
    if len(words) == 1 and phrase_clean not in stop_words:
        return True
    return True

def normalize_phrase(phrase):
    phrase = phrase.strip().lower()
    # 1. 去除前缀#, the, her, his 等冠词或代词
    phrase = re.sub(r"^(#|the|a|an|his|her)\s+", "", phrase)
    # 2. 处理小数点间空格，如 "37 . 0°c" => "37.0°c"
    phrase = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", phrase)
    # 3. 规范斜杠表达："136 / 92 mm" → "136/92 mm"
    phrase = re.sub(r"\s*/\s*", "/", phrase)
    # 4. 规范连接词/单位中间的空格："45 - year - old" → "45-year-old"
    phrase = re.sub(r"\s*-\s*", "-", phrase)
    # 5. 移除多余空格："trimethoprim - sulfamethoxazole " → "trimethoprim-sulfamethoxazole"
    phrase = re.sub(r"\s+", " ", phrase).strip()
    # 6. 标准化单位表达
    phrase = phrase.replace("°c", "°C")
    phrase = phrase.replace("°f", "°F")
    phrase = phrase.replace("kg/ ", "kg/m²")
    return phrase

def depulicate_normalized_phrases(phrase_list):
    seen = []
    result = []
    drop_ent = []
    for phrase in phrase_list:
        if not is_meaningful_phrase(phrase):
            drop_ent.append(phrase)
            continue
        norm_p = normalize_phrase(phrase)
        # 是否已被更长的表达包含（自己是短的）
        if any(norm_p in existing and len(norm_p) < len(existing) for existing in seen):
            drop_ent.append(phrase)
            continue

        # 如果 norm_p 更长，覆盖已有短表达
        to_remove = [i for i, s in enumerate(seen) if s in norm_p and len(s) < len(norm_p)]
        for idx in reversed(to_remove):
            drop_ent.append(result[idx])
            del seen[idx]
            del result[idx]

        seen.append(norm_p)
        result.append(norm_p)
    return result, drop_ent

def get_answer_candidates_huggingface(text, ner_pipeline):
    # 1. 使用 HF 模型识别实体
    ner_results = ner_pipeline(text)
    ner_entities = [ent['word'].strip() for ent in ner_results]
    # 不能按照原文出现顺序排列ent；因为不是所有ent token 都是start index 开头

    # 2. 用 SpaCy 提取 noun chunks
    doc = nlp(text)
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]

    # 3. 合并候选项
    raw_candidates = [cand for cand in ner_entities + noun_chunks if cand and cand.lower() != 'i']
    # 4. normalize phrase, remove unimportant phrase, 
    final_candidates, drop_candidates = depulicate_normalized_phrases(set(raw_candidates))
    print(f"Total candidates: {len(raw_candidates)}-{len(set(raw_candidates))}, remove duplicates / meaningless entities {len(drop_candidates)}, saved {len(final_candidates)}-{len(set(final_candidates))} entities.")
    return list(set(raw_candidates)), list(set(final_candidates)), list(set(drop_candidates))

#%%
# Match Entity
def word_overlap(q, r):
    q_set = set(q.lower().split())
    r_set = set(r.lower().split())
    return len(q_set & r_set) / max(1, len(q_set | r_set))

#%%
def find_sim_ent_from_embs(question_ent, summary_ent, question_embs, ans_embeds,
                           emb_threshold=0.7, token_overlap_threshold=0.3):
    """
    Same logic as find_sim_ent but accepts pre-computed embeddings.
    Use this in ablation to avoid re-encoding entity strings for every threshold.
    question_embs: np.array [n_q, dim]
    ans_embeds:    np.array [n_a, dim]
    """
    best_match = {}
    q_entity_coverage = 0
    not_covered_q_ent = []

    for i in range(len(question_ent)):
        best_sim = emb_threshold
        best_a_ent = None
        for j in range(len(summary_ent)):
            similarity = np.dot(question_embs[i], ans_embeds[j]) / (
                np.linalg.norm(ans_embeds[j]) * np.linalg.norm(question_embs[i])
            )
            if similarity > best_sim and word_overlap(question_ent[i], summary_ent[j]) >= token_overlap_threshold:
                best_sim = similarity
                best_a_ent = summary_ent[j]
        if best_a_ent:
            best_match[question_ent[i]] = best_a_ent
            q_entity_coverage += 1
        else:
            not_covered_q_ent.append(question_ent[i])

    recall = round(q_entity_coverage / len(question_ent), 2)
    return best_match, list(best_match.keys()), not_covered_q_ent, recall


def find_sim_ent(question_ent,summary_ent, emb_model, emb_threshold = 0.7, token_overlap_threshold = 0.3):
    """
    return 
        best_match = {question_ent: (ans_ent, similarity)}, similarity will be above threshold
        list of ans_ent that's found in ques_ant
        q_ent_coverage_rate :.2f
    """
    # array
    question_embs = emb_model.encode(question_ent)
    ans_embeds = emb_model.encode(summary_ent)

    best_match = {}
    q_entity_coverage = 0
    not_covered_q_ent = []
    
    for i in range(len(question_ent)):
        best_sim = emb_threshold
        best_a_ent = None
        for j in range(len(summary_ent)):
            similarity = np.dot(question_embs[i],ans_embeds[j])/ (np.linalg.norm(ans_embeds[j]) * np.linalg.norm(question_embs[i]))
            if similarity > best_sim and word_overlap(question_ent[i], summary_ent[j]) >= token_overlap_threshold:
                best_sim =  similarity
                best_a_ent = summary_ent[j]
        if best_a_ent:
            best_match[question_ent[i]] = best_a_ent
            q_entity_coverage += 1
        else:
            not_covered_q_ent.append(question_ent[i])
    # return covered q_ents; coverage rate, not covered entc v7d6
        
    recall = round(q_entity_coverage/len(question_ent),2)
    print(f"There are {len(list(best_match.keys()))} entities covered, {len(not_covered_q_ent)} entities uncovered, the EntQA is {recall}")
    return best_match, list(best_match.keys()), not_covered_q_ent, recall

#%%
def analyze_medical_response(background, question, response, emb_model, ner_pipeline, emb_threshold = 0.7, token_overlap_threshold = 0.3):
    """
    输入：
        background: str，临床背景描述
        question: str，问题
        response: str，LLM生成的回答
        threshold: float，实体匹配相似度阈值
    返回：
        dict，包括：
            - background_ent_coverage: float
            - ques_ent_coverage: float
            - uncovered_question_ent: list
    """
    ## REVISE: remove question from background
    background_new = background.replace(question, "", 1).strip()

    start_time = time.time()
    # Step 1: 提取实体
    raw_background_ents, background_ents, drop_background_ents = get_answer_candidates_huggingface(background_new, ner_pipeline) or []
    raw_total_question_ents, question_ents, drop_question_ents  = get_answer_candidates_huggingface(question, ner_pipeline) or []
    raw_response_ents, response_ents, drop_response_ents = get_answer_candidates_huggingface(response, ner_pipeline) or []

    # end_time_1 = time.time()
    # print(f"Extract ent from B,Q,A in {end_time_1- start_time} seconds.")

    # Step 2: 实体匹配
    if background_ents and response_ents:
        background_best_match, background_ent_covered, uncovered_background_ent, background_recall = find_sim_ent(background_ents, response_ents, emb_model, emb_threshold = 0.7, token_overlap_threshold = 0.3)
    else:
        background_best_match = {}
        background_ent_covered = []
        background_recall = 0.0
        uncovered_background_ent = background_ents
    
    # logger.info(f"Backrgound Precision = {background_precision}; Recall = {background_recall}; F1 = {background_F1}; Overlap ents in B&A: {background_ent_covered}")
    # logger.info(f"Uncovered ents in B&A: {uncovered_background_ent}")

    if question_ents and response_ents:
        question_best_match, q_ent_covered, uncovered_question_ent, question_recall = find_sim_ent(question_ents, response_ents, emb_model, emb_threshold = 0.7, token_overlap_threshold = 0.3)
    else: 
        question_best_match = {}
        q_ent_covered = []
        question_recall = 0.0
        uncovered_question_ent = question_ents

    end_time_2 = time.time()
    if len(q_ent_covered) + len(uncovered_question_ent) != len(question_ents) or \
    len(background_ent_covered) + len(uncovered_background_ent) != len(background_ents):
        raise ValueError(
            f"Entity coverage mismatch: "
            f"Q covered+uncovered={len(q_ent_covered)+len(uncovered_question_ent)}, total={len(question_ents)}; "
            f"B covered+uncovered={len(background_ent_covered)+len(uncovered_background_ent)}, total={len(background_ents)}"
        )
        
    # logger.info(f"Question Precision = {question_precision}; Recall = {question_recall}; F1 = {question_F1}; Overlap ents in Q&A: {q_ent_covered}")
    # logger.info(f"Uncovered ents in Q&A: {uncovered_question_ent}")
    
    # print(f"Check ent coverage between Q&A, B&A in {end_time_2- end_time_1} seconds.")
    return {
        "background_ents": background_ents,
        "question_ents": question_ents,
        "response_ents": response_ents,
        "dropped_background_ents": drop_background_ents,
        "dropped_question_ents": drop_question_ents,
        "dropped_response_ents": drop_response_ents,
        "background_recall": background_recall,
        "ques_recall": question_recall,
        "background_ent_matched": background_best_match,
        "question_ent_matched": question_best_match,
        "covered_background_ent": background_ent_covered,
        "covered_question_ent": q_ent_covered,
        "uncovered_background_ent": uncovered_background_ent,
        "uncovered_question_ent": uncovered_question_ent
    }


def extract_entities(background, question, response, ner_pipeline):
    """
    Step 1 only: run NER on background/question/response, return entity string lists.
    No embedding, no threshold. Call this once per record to avoid re-running NER.
    """
    background_new = background.replace(question, "", 1).strip()
    print("-----Background-----")
    _, background_ents, drop_background_ents = get_answer_candidates_huggingface(background_new, ner_pipeline) or ([], [], [])
    print("-----Question-----")
    _, question_ents,   drop_question_ents   = get_answer_candidates_huggingface(question, ner_pipeline) or ([], [], [])
    print("-----Response-----")
    _, response_ents,   drop_response_ents   = get_answer_candidates_huggingface(response, ner_pipeline) or ([], [], [])
    return {
        "background_ents": background_ents,
        "question_ents":   question_ents,
        "response_ents":   response_ents,
        "dropped_background_ents": drop_background_ents,
        "dropped_question_ents":   drop_question_ents,
        "dropped_response_ents":   drop_response_ents,
    }


def compute_ent_recall(background_ents, question_ents, response_ents, emb_model=None,
                       emb_threshold=0.7, token_overlap_threshold=0.3,
                       b_embs=None, q_embs=None, r_embs=None):
    """
    Step 2 only: given pre-extracted entity lists, compute recall with the specified thresholds.
    - If b_embs/q_embs/r_embs are provided, uses find_sim_ent_from_embs (no re-encoding).
    - Otherwise falls back to find_sim_ent with emb_model (re-encodes every call).
    Returns the same recall/matched/covered fields as analyze_medical_response.
    """
    use_precomputed = (b_embs is not None and r_embs is not None)

    if background_ents and response_ents:
        if use_precomputed:
            background_best_match, background_ent_covered, uncovered_background_ent, background_recall = \
                find_sim_ent_from_embs(background_ents, response_ents, b_embs, r_embs, emb_threshold, token_overlap_threshold)
        else:
            background_best_match, background_ent_covered, uncovered_background_ent, background_recall = \
                find_sim_ent(background_ents, response_ents, emb_model, emb_threshold, token_overlap_threshold)
    else:
        background_best_match, background_ent_covered, background_recall = {}, [], 0.0
        uncovered_background_ent = background_ents

    use_q_precomputed = (q_embs is not None and r_embs is not None)
    if question_ents and response_ents:
        if use_q_precomputed:
            question_best_match, q_ent_covered, uncovered_question_ent, question_recall = \
                find_sim_ent_from_embs(question_ents, response_ents, q_embs, r_embs, emb_threshold, token_overlap_threshold)
        else:
            question_best_match, q_ent_covered, uncovered_question_ent, question_recall = \
                find_sim_ent(question_ents, response_ents, emb_model, emb_threshold, token_overlap_threshold)
    else:
        question_best_match, q_ent_covered, question_recall = {}, [], 0.0
        uncovered_question_ent = question_ents

    return {
        "background_recall":        background_recall,
        "ques_recall":              question_recall,
        "background_ent_matched":   background_best_match,
        "question_ent_matched":     question_best_match,
        "covered_background_ent":   background_ent_covered,
        "covered_question_ent":     q_ent_covered,
        "uncovered_background_ent": uncovered_background_ent,
        "uncovered_question_ent":   uncovered_question_ent,
    }


# %%
