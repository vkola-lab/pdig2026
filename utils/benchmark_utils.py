#%%
# %cd /projectnb/vkolagrp/yiliu/QA_pipeline
#%%
import json
import jsonlines
import os
import re
from tqdm import tqdm

#%%
idx_to_key = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E"
}

def split_background_and_question(text):
    """
    Split the real question in background 
    input: background text
    output: patient information + question
    """
    text = text.strip()
    question_matches = re.findall(r'[^.?!]*\?+', text)
    if question_matches:
        question_sent = question_matches[-1].strip()
        split_index = text.rfind(question_sent)
        background = text[:split_index].rstrip()
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        question_sent = sentences[-1].strip() if sentences else text
        background = text.rsplit(question_sent, 1)[0].strip()
    return background, question_sent


def clean_benchmark(benchmark, input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_data = []
    if benchmark == "MedExpQA":
        with jsonlines.open(input_path, mode='r') as reader:
            for item in reader:
                raw_options = item["options"]  # keys: "1", "2", ...
                correct_idx_str = str(item["correct_option"]) # text after ABCD
                options = {idx_to_key[k]: v for k, v in raw_options.items() if k in idx_to_key} # dict {A:option_1..}
                
                background, question_sent = split_background_and_question(item['full_question'])
                processed_item = {
                    "question": item['full_question'],  #
                    "question_sent": question_sent,
                    "background": background,
                    "options": options,
                    "type": item["type"],
                    "correct_option_key": idx_to_key[correct_idx_str],
                    "correct_option_text": raw_options[correct_idx_str]
                }
                processed_data.append(processed_item)
    elif benchmark == "MedMCQA":
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]  
        for item in data:
            correct_idx = item["cop"]
            # print("correct_idx: ", correct_idx)
            correct_key = idx_to_key[str(correct_idx)]
            options = {
                "A": item["opa"],
                "B": item["opb"],
                "C": item["opc"],
                "D": item["opd"]
            }
            background, question_sent = split_background_and_question(item["question"])
            processed_item = {
                "question": item["question"].strip(),
                "question_sent": question_sent,
                "background": background,
                "options": options,
                "correct_option_key": correct_key,
                "correct_option_text": options[correct_key]
            }
            processed_data.append(processed_item)
    elif benchmark == "MedQA":
        with jsonlines.open(input_path, "r") as reader:
            for item in reader:
                correct_key = item["answer_idx"]
                options = item["options"]
                background, question_sent = split_background_and_question(item["question"])

                processed_item = {
                    "question": item["question"].strip(),
                    "question_sent": question_sent,
                    "background": background,
                    "options": options,
                    "correct_option_key": correct_key,
                    "correct_option_text": options[correct_key]
                }
                processed_data.append(processed_item)
    elif "USMLE" in benchmark:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 提取选项
        choice_pattern = re.compile(r"\(([A-G])\)\s*([^()]+?)(?=\s*\([A-E]\)|$)")
        for idx, item in enumerate(data):
            options_str = item["choices"]
            matches = choice_pattern.findall(options_str)  # 提取 [('A', 'BMI'), ('B', 'Family history'), ...]
            options = {k: v.strip() for k, v in matches}
            if not options:
                print(f"skip {benchmark} {idx} records, where there is no options")
                continue
            correct_key = item["answer_id"]
            if correct_key not in options:
                print(f"skip {benchmark} {idx} records, where '{correct_key}' is not in options: {list(options.keys())}")
                continue
            correct_text = options[correct_key]
            background, question_sent = split_background_and_question(item["question"])

            processed_item = {
                "question": item["question"].strip(),
                "question_sent": question_sent,
                "background": background,
                "options": options,
                "correct_option_key": correct_key,
                "correct_option_text": correct_text
            }
            processed_data.append(processed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    return processed_data
