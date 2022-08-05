import os
import torch
import numpy as np

from typing import Dict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers.trainer_utils import EvalPrediction
from dataset import get_pretrain_data, R2D2SingleDataset, r2d2_single_collate_fn, get_pretrain_data_v2, get_pretrain_data_v3, r2d2_pair_pretrain_collate_fn
from models.r2d2 import R2D2ForPretrain, load_checkpoint
from transformers import TrainingArguments, Trainer
from trainer import MyTrainer
from transformers.utils import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

logger = logging.get_logger(__name__)

def run_pair_pretrain():
    texts, img_paths = get_pretrain_data_v3()
    dataset = R2D2SingleDataset(texts, img_paths)
    model = R2D2ForPretrain()
    model, _ = load_checkpoint(model, "checkpoints/r2d2/r2d2_pretrain_250m.pth")

    training_args = TrainingArguments(output_dir="./pretrain_checkpoints",
                                      num_train_epochs=1,
                                      per_device_train_batch_size=8,
                                      learning_rate=1e-5,
                                      do_train=True,
                                      do_eval=False,
                                      logging_steps=100,
                                      save_steps=1000,
                                      dataloader_num_workers=4,
                                      dataloader_drop_last=True,
                                      fp16=True)
    trainer = MyTrainer(model=model,
                        args=training_args,
                        train_dataset=dataset,
                        data_collator=r2d2_pair_pretrain_collate_fn)
    trainer.train()
    trainer.save_model("./pretrain_checkpoints/last")

def run():
    texts, img_paths = get_pretrain_data_v2()
    dataset = R2D2SingleDataset(texts, img_paths)
    model = R2D2ForPretrain()
    model, _ = load_checkpoint(model, "checkpoints/r2d2/r2d2_pretrain_250m.pth")

    training_args = TrainingArguments(output_dir="./pretrain_checkpoints",
                                      num_train_epochs=1,
                                      per_device_train_batch_size=8,
                                      learning_rate=1e-5,
                                      do_train=True,
                                      do_eval=False,
                                      logging_steps=100,
                                      save_steps=1000,
                                      dataloader_num_workers=4,
                                      dataloader_drop_last=True,
                                      fp16=True)
    trainer = MyTrainer(model=model,
                        args=training_args,
                        train_dataset=dataset,
                        data_collator=r2d2_single_collate_fn)
    trainer.train()
    trainer.save_model("./pretrain_checkpoints/last")

if __name__ == "__main__":
    # run()
    run_pair_pretrain()
    pass
