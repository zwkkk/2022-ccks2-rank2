import os
import pickle
import json

from encoders import R2D2Encoder, R2D2CrossEncoder, R2D2FullCrossEncoder

from similarity import compute
from dataset import get_model_data
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

def get_badcase():
    results = []
    item_ids1, item_ids2, texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data("test", return_ids=True)
    encoder = R2D2FullCrossEncoder("align_checkpoints/best/pytorch_model.bin")
    embeddings_1, tmp = encoder.encode(texts1, img_paths_1, texts1, img_paths_1, batch_size=8)
    embeddings_1 = [(i+j)/2 for i,j in zip(embeddings_1,tmp)]
    tmp, embeddings_2 = encoder.encode(texts2, img_paths_2, texts2, img_paths_2, batch_size=8)
    embeddings_2 = [(i+j)/2 for i,j in zip(embeddings_2,tmp)]
    
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
    badcases = []
    
    good = 0
    for idx in range(len(labels)):
        label = labels[idx]
        pred = preds[idx]
        if label==pred:
            good += 1
        if label!=pred:
            badcases.append({"src_item_id":  item_ids1[idx], "tgt_item_id": item_ids2[idx], "label": label})
    
    pickle.dump(badcases, open("./outputs/badcases.pkl", "wb"))
    print("acc: {}".format(good/len(labels)))


if __name__ == "__main__":
    get_badcase()
