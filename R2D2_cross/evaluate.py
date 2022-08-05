import os

from similarity import compute
from dataset import get_model_data, get_model_data_v2
from encoders import R2D2Encoder, R2D2CrossEncoder, R2D2FullCrossEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

def evaluate_full_cross():
    texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data("test")
    encoder = R2D2FullCrossEncoder("align_checkpoints/best/pytorch_model.bin")
    src_embeddings, tgt_embeddings = encoder.encode(texts1, img_paths_1, texts2, img_paths_2, batch_size=8)
    preds = []
    threshold = 0.5
    for idx in range(len(src_embeddings)):
        src_embed = src_embeddings[idx]
        tgt_embed = tgt_embeddings[idx]
        score = compute(src_embed.tolist(), tgt_embed.tolist())
        if score > threshold:
            preds.append(1)
        else:
            preds.append(0)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds)
    metrics["recall"] = recall_score(labels, preds)
    print(metrics)

def evaluate_cross():
    texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data("test")
    encoder = R2D2CrossEncoder("align_checkpoints/best/pytorch_model.bin")
    src_embeddings, tgt_embeddings = encoder.encode(texts1, img_paths_1, texts2, img_paths_2, batch_size=8)
    preds = []
    threshold = 0.5
    for idx in range(len(src_embeddings)):
        src_embed = src_embeddings[idx]
        tgt_embed = tgt_embeddings[idx]
        score = compute(src_embed.tolist(), tgt_embed.tolist())
        if score > threshold:
            preds.append(1)
        else:
            preds.append(0)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds)
    metrics["recall"] = recall_score(labels, preds)
    print(metrics)

def evaluate():
    texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data("test")
    encoder = R2D2Encoder("./align_checkpoints/best")
    embeddings_1 = encoder.encode(texts1, img_paths_1, batch_size=8)
    embeddings_2 = encoder.encode(texts2, img_paths_2, batch_size=8)
    preds = []
    threshold = 0.5
    for idx in range(len(embeddings_1)):
        embed1 = embeddings_1[idx]
        embed2 = embeddings_2[idx]
        score = compute(embed1.tolist(), embed2.tolist())
        if score > threshold:
            preds.append(1)
        else:
            preds.append(0)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds)
    metrics["recall"] = recall_score(labels, preds)
    print(metrics)

if __name__ == "__main__":
    # evaluate()
    # evaluate_cross()
    evaluate_full_cross()
    pass
