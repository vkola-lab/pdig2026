#%%
from sentence_transformers import SentenceTransformer
import torch

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
cache_dir = "/projectnb/vkolagrp/yiliu/.cache"

os.environ["HF_HOME"] = "/projectnb/vkolagrp/yiliu/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/projectnb/vkolagrp/yiliu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/projectnb/vkolagrp/yiliu/.cache/huggingface/hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/projectnb/vkolagrp/yiliu/.cache/sentence_transformers"

#%%
# Sentence_Transformer guide: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html?utm_source=chatgpt.com
# "Qwen/Qwen3-Embedding-8B"
# "Salesforce/SFR-Embedding-Mistral"
# "Linq-AI-Research/Linq-Embed-Mistral"
# "Alibaba-NLP/gte-Qwen2-7B-instruct"

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", cache_folder = cache_dir, device = device)

#%%
question_ents = ["kidney failure", "glomerular filtration", "blood pressure"]
summary_ents = ["renal insufficiency", "glomerular basement membrane", "hypertension"]

question_ents_emb = model.encode(question_ents)
summary_ents_emb = model.encode(summary_ents)

# %%
similarity = model.similarity(question_ents_emb, summary_ents_emb)

# %%
def word_overlap(q, r):
    q_set = set(q.lower().split())
    r_set = set(r.lower().split())
    return len(q_set & r_set) / max(1, len(q_set | r_set))
#%%
import numpy as np
#%%
match_dict = {}
q_entity_coverage = 0
not_covered_q_ent = []
for i in range(len(question_ents)):
    best_sim = 0.7
    best_a_ent = None
    print("Question ents: ",question_ents[i])
    for j in range(len(summary_ents)):
        print("Summary ents: ", summary_ents[j])
        similarity = np.dot(question_ents_emb[i],summary_ents_emb[j])/ (np.linalg.norm(summary_ents_emb[j]) * np.linalg.norm(question_ents_emb[i]))
        print("Similarity: ", similarity)
        if similarity > best_sim: #and word_overlap(question_ents[i],summary_ents[j]) >= 0.3:
            best_sim = similarity
            best_a_ent = summary_ents[j]
    if best_a_ent:
        print(f"The best match for {question_ents[i]} is {best_a_ent}")
        match_dict[question_ents[i]] = best_a_ent
        q_entity_coverage += 1
    else:
        print(f"{question_ents[i]} can't find matched ent in summary.")
        not_covered_q_ent.append(question_ents[i])
#%%
recall = round(q_entity_coverage/len(question_ents),2)
precision = round(q_entity_coverage/len(summary_ents),2)
if q_entity_coverage == 0:
    F1 = 0
else:
    F1 = round(2*precision*recall/(precision + recall),2)
# %%
question = "How is today's weather?"
generated_q_list = ["Is today sunny day?", "When will we eat lunch?", "What's the weather today?"]
question_emb = model.encode(question)
generated_q_emb = model.encode(generated_q_list)
similarity = model.similarity(question_emb, generated_q_emb)
idx = similarity.argmax().item()
best_match_q = generated_q_list[idx]
best_q_sim = round(similarity.max().item(),4)
print(f"The most sim question: {best_match_q}, with sim score {best_q_sim}")
# %%
