#%%
from extraction import clean_markdown_symbols
from benchmark_utils import split_background_and_question
import re
from tqdm import tqdm
import os
import json
from nltk.tokenize import sent_tokenize
#%%
#For 46B-Mistral
def add_explanation_if_missing(response: str) -> str:
    if "Explanation:" in response:
        return response

    # 插入在第二个 \n\n 后
    double_newlines = [m.start() for m in re.finditer(r'\n\n', response)]
    if len(double_newlines) >= 2:
        insert_pos = double_newlines[1] + 2
        return response[:insert_pos] + "Explanation: " + response[insert_pos:]

    # 插入在第二个 \n 后
    single_newlines = [m.start() for m in re.finditer(r'\n', response)]
    if len(single_newlines) >= 2:
        insert_pos = single_newlines[1] + 1
        return response[:insert_pos] + "Explanation: " + response[insert_pos:]

    # 最后 fallback：从第二个英文句号后插
    split_match = re.split(r'(?<=[.!?])\s+', response, maxsplit=2)
    if len(split_match) >= 3:
        return split_match[0] + " " + split_match[1] + " Explanation: " + split_match[2]

    return response

# %%
def clean_gemma_response(model, benchmark):
    # Qwen23
    # folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/New_response/metric-evaluate-{model}/RESULTS_{benchmark}"
    # folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/old-evaluate-{model}/RESULTS_{benchmark}/direct_prompt_TOP_32_DOCS"
    folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/New_response/metric-evaluate-{model}/RESULTS_{benchmark}/direct_prompt_TOP_32_DOCS"
    extracted_data = []
    none_cnt = 0
    no_pred_ans = 0

    if os.path.exists(folder_path):
        for filename in tqdm(sorted(os.listdir(folder_path))):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        background = data.get("background", "")
                        ques_sent = split_background_and_question(background)[1]
                        option = data.get("option", "")
                        ground_truth = data.get("ground_truth", "").strip()
                        prediction = data.get("prediction", "").strip()
                        acc = ground_truth == prediction

                        response_raw = data.get("response", "")
                        response_patched = add_explanation_if_missing(response_raw)
                        response_clean = clean_markdown_symbols(response_patched)

                        explanation = None
                        match = re.search(r"Explanation:\s*(.*)", response_clean, re.DOTALL)
                        if match:
                            explanation = match.group(1).strip()
                        else:
                            none_cnt += 1
                            explanation = response_clean
                        extracted_data.append({
                            "background": background,
                            "question": ques_sent,
                            "option": option,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "acc": acc,
                            "explanation": explanation
                        })

                    except Exception as e:
                        no_pred_ans += 1
                        print(f"Error reading {filename}: {e}")
        print(f"There are {no_pred_ans} none pred label in {benchmark}, {model} generated responses.")
        print(f"There are {none_cnt} none explanation in {benchmark}, {model} generated responses.")
        output_file = f"/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/{model}/extracted_{model}_{benchmark}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(extracted_data, f_out, indent=2)
        print("Save file in ",output_file)
#%%
def clean_ultramedical_response(model, benchmark):
    """
    directly use response as explanation
    model = ["8B-UltraMedical", "70B-UltraMedical"]
    """
    folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/New_response/metric-evaluate-{model}/RESULTS_{benchmark}"
    extracted_data = []
    no_pred_ans = 0

    if os.path.exists(folder_path):
        for filename in tqdm(sorted(os.listdir(folder_path))):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        background = data.get("background", "")
                        ques_sent = split_background_and_question(background)[1]
                        option = data.get("option", "")
                        ground_truth = data.get("ground_truth", "").strip()
                        prediction = data.get("prediction", "").strip()
                        acc = ground_truth == prediction

                        response_raw = data.get("response", "")
                        response_clean = response_raw.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", " ").replace("\n\n", " ").replace("  ","")
                        explanation = response_clean
                        extracted_data.append({
                            "background": background,
                            "question": ques_sent,
                            "option": option,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "acc": acc,
                            "explanation": explanation
                        })

                    except Exception as e:
                        no_pred_ans += 1
                        print(f"Error reading {filename}: {e}")
        print(f"There are {no_pred_ans} none pred label in {benchmark}, {model} generated responses.")
        output_file = f"/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/{model}/extracted_{model}_{benchmark}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(extracted_data, f_out, indent=2)
        print("Save file in ",output_file)
#%%
def clean_deepseek_response(model, benchmark):
    """
    directly use response (after Explanation)as explanation
    model = ["67B-DeepSeek"]
    """
    folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/New_response/metric-evaluate-{model}/RESULTS_{benchmark}"
    extracted_data = []
    no_pred_ans = 0

    if os.path.exists(folder_path):
        for filename in tqdm(sorted(os.listdir(folder_path))):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        background = data.get("background", "")
                        ques_sent = split_background_and_question(background)[1]
                        option = data.get("option", "")
                        ground_truth = data.get("ground_truth", "").strip()
                        prediction = data.get("prediction", "").strip()
                        acc = ground_truth == prediction

                        response_raw = data.get("response", "")
                        match = re.search(r"Explanation:\s*(.*)", response_raw, re.DOTALL)
                        if match:
                            explanation = match.group(1).strip()
                        else:
                            explanation = response_raw

                        extracted_data.append({
                            "background": background,
                            "question": ques_sent,
                            "option": option,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "acc": acc,
                            "explanation": explanation
                        })

                    except Exception as e:
                        no_pred_ans += 1
                        print(f"Error reading {filename}: {e}")
        print(f"There are {no_pred_ans} none pred label in {benchmark}, {model} generated responses.")
        output_file = f"/projectnb/vkolagrp/yiliu/QA_pipeline/processed_data/{model}/extracted_{model}_{benchmark}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(extracted_data, f_out, indent=2)
        print("Save file in ",output_file)

#%%
qwen_models = ['1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25','72B-Qwen25']
# models = ['0.5B-Qwen25','2B-Gemma', '6B-Yi','7B-DeepSeek','7B-Falcon','7B-Gemma','7B-LLaMA2','8B-UltraMedical','13B-LLaMA2','34B-Yi','40B-Falcon','46B-Mixtral','67B-DeepSeek','70B-LLaMA2']
benchmark = 'MedQA' #,'MedQA','PubMedQA','USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3']
for model in tqdm(qwen_models, desc="Processing Models"):
    # for benchmark in benchmarks:
    clean_gemma_response(model, benchmark)

# for benchmark in benchmarks:
#     clean_ultramedical_response('70B-UltraMedical', benchmark)

# %%
