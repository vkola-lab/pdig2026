import os
import ast
import re 
import json
from tqdm import tqdm
import logging
from itertools import product
from benchmark_utils import split_background_and_question

# AI Agent response
# model: "0.5B-Qwen25",
# benchmark: "USMLE_STEP_3"

#%%
def clean_markdown_symbols(text):
    # 1. 去掉 markdown 粗体和斜体
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.*?)\*", r"\1", text, flags=re.DOTALL)

    # 2. 删除行首 markdown 标题符号（如 ###、##、#）
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 3. 删除行首 markdown 列表符号（如 - 或 *，后跟空格）
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)

    # 4. 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text
#%%
def check_unwanted_symbols(text):
    print("Symbol in the str: ")
    print(f"  '-' in text: {'-' in text}")
    print(f"  '*' in text: {'*' in text}")
    print(f"  '#' in text: {'#' in text}")

    print("结构级 markdown 检查：")
    if re.search(r"\*\*(.*?)\*\*", text):
        print("  ✅ 粗体 markdown **...** 仍存在")
    if re.search(r"\*(?!\*)(.*?)\*(?!\*)", text):
        print("  ✅ 斜体 markdown *...* 仍存在")
    if re.search(r"^\s*[-*]\s+", text, flags=re.MULTILINE):
        print("  ✅ markdown 列表项仍存在（如 - item）")
    if re.search(r"^\s*#{1,6}\s+", text, flags=re.MULTILINE):
        print("  ✅ markdown 标题仍存在（如 ### Title）")
# Qwen 2.5 series
def extract_explanation(model, benchmark):
    # folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/old-evaluate-{model}/RESULTS_{benchmark}/direct_prompt_TOP_32_DOCS"
    folder_path = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/New_response/metric-evaluate-2B-Gemma/RESULTS_{benchmark}"
    extracted_data = []
    cnt = 0
    none_cnt = 0

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

                        response_raw = data.get("response", "") # str
                        # response 中的特殊符号: -, *,**,###,#
                        response_clean = clean_markdown_symbols(response_raw)

                        explanation = None
                        match = re.search(r"Explanation:\s*(.*)", response_clean, re.DOTALL)
                        if match:
                            explanation = match.group(1).strip()
                        else:
                            none_cnt += 1

                        cnt += 1
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
                        print(f"Error reading {filename}: {e}")

        logging.error(f"There are {none_cnt} none explanation in {benchmark}, {model} generated responses (Overall {cnt} records.)")

        output_file = f"/projectnb/vkolagrp/yiliu/QA_pipeline/AI_agent_pipeline/BruceJia_Response_dataset/extracted_data/{model}/extracted_{model}_{benchmark}.json"
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(extracted_data, f_out, indent=2)

#%%



# models = ['0.5B-Qwen25']#,'1.5B-Qwen25','3B-Qwen25','7B-Qwen25','14B-Qwen25','32B-Qwen25','72B-Qwen25']
# benchmarks = ['MedExpQA','MedQA','PubMedQA','USMLE_STEP_1','USMLE_STEP_2','USMLE_STEP_3']

# models = ['2B-Gemma',]

# for model, benchmark in tqdm(product(models, benchmarks), total=len(models)*len(benchmarks), desc="Processing"):
#     # 你的处理逻辑
#     print(f"Processing: {model} - {benchmark}")
#     extract_explanation(model,benchmark)


#%%
# response_raw = "The correct answer is B. Left coxofemoral arthrosis. Explanation: Coxofemoral arthritis (also known as femoroacetabular impingement) is a condition where there is abnormal wear and tear on the joint between the head of the femur (thigh bone) and the acetabulum (hip socket). This can lead to inflammation and pain, particularly in the area around the hip joint. The symptoms typically include pain that worsens with activities such as lifting the leg with the knee extended, which would explain the patient's discomfort described in the question. Options A, C, and D are less likely given the specific presentation of pain in the buttocks, left trochanteric region, lateral aspect of the left thigh up to the knee, and left leg up to the middle third: - **Gouty arthritis of left hip**: While gout can affect joints, it usually presents with acute attacks of sudden onset severe pain, redness, warmth, and swelling rather than chronic pain localized to one side. - **Left coxofemoral arthrosis**: As mentioned earlier, this is the most appropriate diagnosis based on the description of pain during certain movements and relief upon flexion of the knee. - **Radiated low back pain/lumbosciatica**: Although lumbar spine issues can cause referred pain to the hips, the specific pattern of pain described here does not align well with typical patterns of lumbosacral radicular pain. Therefore, the combination of symptoms points directly towards coxofemoral arthritis as the primary clinical suspicion."
# # print(repr(response_raw))
# cleaned_response = clean_markdown_symbols(response_raw)
# print("Cleaned version: ", cleaned_response)
# match = re.search(r"Explanation:\s*(.*)", cleaned_response, re.DOTALL)
# explanation = match.group(1).strip()
# print("Explanation: ", explanation)

#%%
