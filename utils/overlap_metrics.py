#%%
# BLEU
# ROUGE; METEOR (> ROUGE, BLEU) 
import evaluate
import nltk
nltk.download("punkt")
import torch

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
# cache_dir = "/projectnb/vkolagrp/yiliu/.cache"

# Load
bleu = evaluate.load("bleu") 
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using devices: ",device)


#%%
def compute_overlapping_metrics(question: str, explanation: str) -> dict:
    results = {}
    predictions = [explanation]
    references = [question]

    # BLEU_4 is default
    bleu_score = bleu.compute(predictions=predictions, references=[[question]])
    results["bleu4"] = bleu_score["bleu"]
    # for i, score in enumerate(bleu_score["precisions"]):
    #     results[f"bleu{i+1}"] = score
    results["bleu1"] = bleu_score["precisions"][0]
    
    # METEOR
    meteor_score = meteor.compute(predictions=predictions, references=references)
    results["meteor"] = meteor_score["meteor"]

    # ROUGE（包含 rouge1, rouge2, rougeL, rougeLsum）
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results.update({
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "rougeL": rouge_score["rougeL"],
        "rougeLsum": rouge_score["rougeLsum"]
    })

    # BERTScore
    score = bertscore.compute(
                        predictions=predictions,
                        references=references,
                        model_type = "microsoft/deberta-xlarge-mnli",
                        lang="en",
                        use_fast_tokenizer=False,
                        device = device,
                        batch_size = 16
                    )
    results["bertscore_f1"] = score['f1'][0]

    return results
#%%
# question = "What are the risk factors for sciatica? What are the biggest risk factors for developing sciatica?"
# response_1 = "Pelvic girdle injury: Between 10 & 30 percent people with chronic low back pain have pain generation from sacroiliac joint (sij) (medical literature research). Sciatica common in people with sij dysfunction (personal observation). To my mind sij most common source of chronic sciatica, & from pelvic presacral plexus impingement from sij dysfunction. Women with joint hypermobility syndrome (jhs) are prone to sciatica."

# results = compute_overlapping_metrics(question, response_1)
# print(results)


# %%
# data_w_human_rating = pd.read_csv("./metric_comparision/dataset_w_human_rating/Non-blind/nonblind_merged_df.csv")
