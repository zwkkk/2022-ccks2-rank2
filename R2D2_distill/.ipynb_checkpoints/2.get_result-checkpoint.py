import os
import json

from dataset import get_model_data, get_model_data_v2
from encoders import R2D2Encoder, R2D2CrossEncoder, R2D2FullCrossEncoder

def run_for_full_cross():
    results = []
    item_id1, item_id2, texts1, texts2, img_paths_1, img_paths_2 = get_model_data("valid", return_ids=True)
    encoder = R2D2FullCrossEncoder("align_checkpoints/best/pytorch_model.bin")
    #src_embeddings, tgt_embeddings = encoder.encode(texts1, img_paths_1, texts2, img_paths_2, batch_size=8)
    src_embeddings, _ = encoder.encode(texts1, img_paths_1, texts1, img_paths_1, batch_size=8)
    _, tgt_embeddings = encoder.encode(texts2, img_paths_2, texts2, img_paths_2, batch_size=8)
    for idx in range(len(item_id1)):
        result = dict()
        result["src_item_id"] = item_id1[idx]
        result["tgt_item_id"] = item_id2[idx]
        result["src_item_emb"] = str(list(src_embeddings[idx]))
        result["tgt_item_emb"] = str(list(tgt_embeddings[idx]))
        result["threshold"] = 0.5
        results.append(result)
    with open("./outputs/zwk_result.jsonl", "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False)+"\n")

def run_for_cross():
    results = []
    item_id1, item_id2, texts1, texts2, img_paths_1, img_paths_2 = get_model_data("valid")
    encoder = R2D2CrossEncoder("align_checkpoints/best/pytorch_model.bin")
    src_embeddings, tgt_embeddings = encoder.encode(texts1, img_paths_1, texts2, img_paths_2, batch_size=8)
    for idx in range(len(item_id1)):
        result = dict()
        result["src_item_id"] = item_id1[idx]
        result["tgt_item_id"] = item_id2[idx]
        result["src_item_emb"] = str(list(src_embeddings[idx]))
        result["tgt_item_emb"] = str(list(tgt_embeddings[idx]))
        result["threshold"] = 0.5
        results.append(result)
    with open("./outputs/zwk_result.jsonl", "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False)+"\n")

def run():
    results = []
    item_id1, item_id2, texts1, texts2, img_paths_1, img_paths_2 = get_model_data("valid")
    encoder = R2D2Encoder("./align_checkpoints/best")
    embeddings1 = encoder.encode(texts1, img_paths_1, return_numpy=True, batch_size=8)
    embeddings2 = encoder.encode(texts2, img_paths_2, return_numpy=True, batch_size=8)
    for idx in range(len(item_id1)):
        result = dict()
        result["src_item_id"] = item_id1[idx]
        result["tgt_item_id"] = item_id2[idx]
        result["src_item_emb"] = str(list(embeddings1[idx]))
        result["tgt_item_emb"] = str(list(embeddings2[idx]))
        result["threshold"] = 0.5
        results.append(result)
    with open("./outputs/zwk_result.jsonl", "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    # run()
    # run_for_cross()
    run_for_full_cross()
    pass
