import os
import pickle
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import R2D2SingleDataset, r2d2_single_collate_fn, get_model_data, R2D2PairDataset, r2d2_pair_collate_fn, R2D2CrossDataset, r2d2_cross_collate_fn, get_model_data_v2
from align_model import R2D2ForAlign, R2D2CrossForAlign, R2D2FullCrossForAlign

def load_pytorch_model(model_path, model, strict=False):
    tmp_model = torch.load(model_path)
    if hasattr(tmp_model,"module"):
        model.load_state_dict(tmp_model.module, strict=strict)
    else:
        model.load_state_dict(tmp_model, strict=strict)
    return model

class R2D2FullCrossEncoder(object):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = R2D2FullCrossForAlign()
        self.load_model()
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def load_model(self):
        if self.model_path.endswith(".bin"):
            weight_path = self.model_path
        else:
            weight_path = os.path.join(self.model_path, "pytorch_model.bin")
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight, strict=True)

    def encode(self,
               src_texts: List[str],
               src_images: List[str],
               tgt_texts: List[str],
               tgt_images: List[str],
               labels: List[int] = None,
               batch_size=32,
               return_numpy: bool=True,
               parallel: bool = False) -> List:
        assert len(src_texts)==len(src_images)
        assert len(tgt_texts)==len(tgt_images)
        assert len(src_texts)==len(tgt_texts)
        if labels is None:
            labels = [0]*len(src_texts)
        dataset = R2D2CrossDataset(src_texts, tgt_texts, src_images, tgt_images, labels)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=r2d2_cross_collate_fn,
                                shuffle=False,
                                prefetch_factor=batch_size,
                                num_workers=4)
        all_src_embeddings = []
        all_tgt_embeddings = []
        for batch in tqdm(dataloader):
            batch = {k:v.to(self.device) for k,v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch, item_emb=True)
            src_embeddings = output.embeddings_1.detach()
            tgt_embeddings = output.embeddings_2.detach()
            if return_numpy:
                src_embeddings = src_embeddings.cpu().numpy()
                tgt_embeddings = tgt_embeddings.cpu().numpy()
                src_embeddings = [src_embedding for src_embedding in src_embeddings]
                tgt_embeddings = [tgt_embedding for tgt_embedding in tgt_embeddings]
                all_src_embeddings.extend(src_embeddings)
                all_tgt_embeddings.extend(tgt_embeddings)
            else:
                src_embeddings = src_embeddings.chunk(src_embeddings.shape[0])
                tgt_embeddings = tgt_embeddings.chunk(tgt_embeddings.shape[0])
                src_embeddings = list(src_embeddings)
                tgt_embeddings = list(tgt_embeddings)
                src_embeddings = [src_embedding.squeeze() for src_embedding in src_embeddings]
                tgt_embeddings = [tgt_embedding.squeeze() for tgt_embedding in tgt_embeddings]
                all_src_embeddings.extend(src_embeddings)
                all_tgt_embeddings.extend(tgt_embeddings)
        return all_src_embeddings, all_tgt_embeddings


class R2D2CrossEncoder(object):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = R2D2CrossForAlign()
        self.load_model()
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def load_model(self):
        if self.model_path.endswith(".bin"):
            weight_path = self.model_path
        else:
            weight_path = os.path.join(self.model_path, "pytorch_model.bin")
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight, strict=True)

    def encode(self,
               src_texts: List[str],
               src_images: List[str],
               tgt_texts: List[str],
               tgt_images: List[str],
               labels: List[int] = None,
               batch_size=32,
               return_numpy: bool=True,
               parallel: bool = False) -> List:
        assert len(src_texts)==len(src_images)
        assert len(tgt_texts)==len(tgt_images)
        assert len(src_texts)==len(tgt_texts)
        if labels is None:
            labels = [0]*len(src_texts)
        dataset = R2D2PairDataset(src_texts, tgt_texts, src_images, tgt_images, labels)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                collate_fn=r2d2_pair_collate_fn,
                                shuffle=False,
                                prefetch_factor=batch_size,
                                num_workers=4)
        all_src_embeddings = []
        all_tgt_embeddings = []
        for batch in tqdm(dataloader):
            batch = {k:v.to(self.device) for k,v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch, item_emb=True)
            src_embeddings = output.embeddings_1.detach()
            tgt_embeddings = output.embeddings_2.detach()
            if return_numpy:
                src_embeddings = src_embeddings.cpu().numpy()
                tgt_embeddings = tgt_embeddings.cpu().numpy()
                src_embeddings = [src_embedding for src_embedding in src_embeddings]
                tgt_embeddings = [tgt_embedding for tgt_embedding in tgt_embeddings]
                all_src_embeddings.extend(src_embeddings)
                all_tgt_embeddings.extend(tgt_embeddings)
            else:
                src_embeddings = src_embeddings.chunk(src_embeddings.shape[0])
                tgt_embeddings = tgt_embeddings.chunk(tgt_embeddings.shape[0])
                src_embeddings = list(src_embeddings)
                tgt_embeddings = list(tgt_embeddings)
                src_embeddings = [src_embedding.squeeze() for src_embedding in src_embeddings]
                tgt_embeddings = [tgt_embedding.squeeze() for tgt_embedding in tgt_embeddings]
                all_src_embeddings.extend(src_embeddings)
                all_tgt_embeddings.extend(tgt_embeddings)
        return all_src_embeddings, all_tgt_embeddings


class R2D2Encoder(object):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = R2D2ForAlign()
        self.load_model()
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def load_model(self):
        if self.model_path.endswith(".bin"):
            weight_path = self.model_path
        else:
            weight_path = os.path.join(self.model_path, "pytorch_model.bin")
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight, strict=True)

    def encode(self, texts: List[str], images: List[str], batch_size=32, return_numpy: bool=True, parallel: bool = False) -> List:
        assert len(texts)==len(images)
        dataset = R2D2SingleDataset(texts, images)
        if parallel:
            sampler = DistributedSampler(dataset, num_replicas=world_size, shuffle=False)
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=sampler,
                                    collate_fn=r2d2_single_collate_fn,
                                    prefetch_factor=batch_size,
                                    num_workers=4)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    collate_fn=r2d2_single_collate_fn,
                                    shuffle=False,
                                    prefetch_factor=batch_size,
                                    num_workers=4)
        all_embeddings = []
        for batch in tqdm(dataloader):
            batch = {k:v.to(self.device) for k,v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch, item_emb=True)
            embeddings = output.embeddings_1.detach()
            if return_numpy:
                embeddings = embeddings.cpu().numpy()
                embeddings = [embedding for embedding in embeddings]
                all_embeddings.extend(embeddings)
            else:
                embeddings = embeddings.chunk(embeddings.shape[0])
                embeddings = list(embeddings)
                embeddings = [embedding.squeeze() for embedding in embeddings]
                all_embeddings.extend(embeddings)
        return all_embeddings

if __name__ == "__main__":
    texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data_v2("test")
    encoder = R2D2FullCrossEncoder("align_checkpoints/best/pytorch_model.bin")
    src_embeddings, tgt_embeddings = encoder.encode(texts1[:64], img_paths_1[:64], texts2[:64], img_paths_2[:64], labels=labels[:64],batch_size=8)
    print(len(src_embeddings))
    print(len(tgt_embeddings))
    print(src_embeddings[0].shape)
    print(tgt_embeddings[0].shape)
    # encoder = R2D2Encoder("./align_checkpoints/best")
    # embeddings = encoder.encode(texts1[:65], img_paths_1[:65], batch_size=8)
