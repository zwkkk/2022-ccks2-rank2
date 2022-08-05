import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from align_model import R2D2ForAlign, R2D2CrossForAlign, R2D2FullCrossForAlign
from typing import Dict
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers.trainer_utils import EvalPrediction
from dataset import get_model_data, R2D2PairDataset, r2d2_pair_collate_fn, get_model_data_v2, R2D2CrossDataset, r2d2_cross_collate_fn
from transformers import TrainingArguments, Trainer
from trainer import MyTrainer
from transformers.utils import logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

logger = logging.get_logger(__name__)

def binary_regress_metrics(output: EvalPrediction) -> Dict[str, float]:
    cos_sim = output.predictions[0]
    labels = output.predictions[3]
    preds = (cos_sim>0.5).astype(int)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="binary")
    metrics["precision"] = precision_score(labels, preds, average="binary")
    metrics["recall"] = recall_score(labels, preds, average="binary")
    logger.info(metrics)
    return metrics

def binary_metrics(output: EvalPrediction) -> Dict[str, float]:
    logits = output.predictions[0]
    labels = output.predictions[3]
    preds = np.argmax(logits, axis=-1)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="binary")
    metrics["precision"] = precision_score(labels, preds, average="binary")
    metrics["recall"] = recall_score(labels, preds, average="binary")
    logger.info(metrics)
    return metrics

def load_pytorch_model(model_path, model, strict=False):
    tmp_model = torch.load(model_path)
    if hasattr(tmp_model,"module"):
        model.load_state_dict(tmp_model.module, strict=strict)
    else:
        model.load_state_dict(tmp_model, strict=strict)
    return model

def run():
    train_texts1, train_texts2, train_img_paths_1, train_img_paths_2, train_labels = get_model_data("train")
    test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, test_labels = get_model_data("test")
    train_dataset = R2D2CrossDataset(train_texts1, train_texts2, train_img_paths_1, train_img_paths_2, train_labels, max_length=400)
    test_dataset = R2D2CrossDataset(test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, test_labels, max_length=400)
    model = R2D2FullCrossForAlign("pretrain_checkpoints/title_pretrain/pytorch_model.bin")
    # model = R2D2FullCrossForAlign('checkpoints/r2d2/r2d2_pretrain_250m.pth')
    # model =  R2D2ForAlign("pretrain_checkpoints/title_pretrain/pytorch_model.bin")
    # model = R2D2ForAlign('checkpoints/r2d2/r2d2_pretrain_250m.pth')
    # model = R2D2ForAlign()
    # model = load_pytorch_model("./align_checkpoints/best/pytorch_model.bin", model,  strict=True)
    training_args = TrainingArguments(output_dir="./align_checkpoints", 
                                      num_train_epochs=5,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      learning_rate=1e-5,
                                      load_best_model_at_end=True,
                                      do_train=True,
                                      do_eval=True,
                                      evaluation_strategy="steps",
                                      logging_steps=400,
                                      eval_steps=400,
                                      save_steps=400,
                                      dataloader_num_workers=4,
                                      metric_for_best_model="f1",
                                      fp16=True)
    # optimizer = torch.optim.Adam([{"params": model.model.parameters(), "lr": 1e-5}], lr=2e-5)
    trainer = MyTrainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset,
                        # optimizers=(optimizer, None),
                        data_collator=r2d2_cross_collate_fn,
                        compute_metrics=binary_regress_metrics)
    trainer.train()
    trainer.save_model("./align_checkpoints/best")

def test():
    test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, test_labels = get_model_data("test")
    test_dataset = R2D2PairDataset(test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, test_labels)
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=r2d2_pair_collate_fn)
    batch = next(iter(dataloader))
    batch = {k:v.to("cuda") for k,v in batch.items()}
    model =  R2D2CrossForAlign("pretrain_checkpoints/title_pretrain/pytorch_model.bin")
    model = model.to("cuda")
    output = model(**batch)
    # output = model(**batch, item_emb=True)
    print(output)

if __name__ == "__main__":
    run()
    # test()
    pass
