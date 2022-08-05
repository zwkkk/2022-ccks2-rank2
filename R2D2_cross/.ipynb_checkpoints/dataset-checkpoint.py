import os
import torch
import pickle
import torchvision
import pandas as pd

from tqdm import tqdm
from PIL import Image
from models.r2d2 import MyR2D2
from torch import Tensor
from utils import read_data
from typing import List, Dict
from align_model import R2D2FullCrossForAlign
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


tokenizer = BertTokenizer.from_pretrained("./checkpoints/hfl_roberta")

def merge(pair, info):
    src_pair = pd.merge(pair, info, left_on="src_item_id", right_on="item_id")
    merged_pair = pd.merge(src_pair, info, left_on="tgt_item_id", right_on="item_id", suffixes=("_src",  "_tgt"))
    discard_columns = ["industry_name_src", "industry_name_tgt", "cate_id_src", "cate_id_tgt","cate_name_path_src",
                       "cate_name_src", "cate_name_tgt", "cate_id_path_src", "cate_id_path_tgt", "cate_name_path_tgt",
                      "item_id_src", "item_id_tgt", "item_image_name_src", "item_image_name_tgt"]
    for col in discard_columns:
        del  merged_pair[col]
    return merged_pair

def build_item_pvs_feature(pair):
    src_item_pvs_features = []
    tgt_item_pvs_features = []
    for idx, row in tqdm(pair.iterrows()):
        src_item_pvs_feature = []
        tgt_item_pvs_feature = []
        item_pvs_src = row["item_pvs_src"]
        item_pvs_tgt = row["item_pvs_tgt"]
        common_keys = list(set(item_pvs_src.keys()).intersection(item_pvs_tgt.keys()))
        for key in common_keys:
            src_item_pvs_feature.append(":".join([key, item_pvs_src[key]]))
            tgt_item_pvs_feature.append(":".join([key, item_pvs_tgt[key]]))
        src_item_pvs_features.append("#".join(src_item_pvs_feature))
        tgt_item_pvs_features.append("#".join(tgt_item_pvs_feature))
    pair["src_item_pvs_features"] = src_item_pvs_features
    pair["tgt_item_pvs_features"] = tgt_item_pvs_features
    return pair

def build_sku_pvs_feature(pair):
    src_sku_pvs_features = []
    tgt_sku_pvs_features = []
    for idx, row in tqdm(pair.iterrows()):
        src_sku_pvs_feature = []
        tgt_sku_pvs_feature = []
        sku_pvs_src = row["sku_pvs_src"]
        sku_pvs_tgt = row["sku_pvs_tgt"]
        common_keys = list(set(sku_pvs_src.keys()).intersection(sku_pvs_tgt.keys()))
        for key in common_keys:
            src_sku_pvs_feature.append(":".join([key, sku_pvs_src[key]]))
            tgt_sku_pvs_feature.append(":".join([key, sku_pvs_tgt[key]]))
        src_sku_pvs_features.append("#".join(src_sku_pvs_feature))
        tgt_sku_pvs_features.append("#".join(tgt_sku_pvs_feature))
    pair["src_sku_pvs_features"] = src_sku_pvs_features
    pair["tgt_sku_pvs_features"] = tgt_sku_pvs_features
    return pair

def build_pair_text_feature(pair, info):
    merged_pair = merge(pair, info)
    merged_pair = build_sku_pvs_feature(merged_pair)
    merged_pair = build_item_pvs_feature(merged_pair)
    merged_pair["src_text_features"] = merged_pair.apply(lambda row: "@".join([row["title_src"], row["src_sku_pvs_features"], row["src_item_pvs_features"]]),axis=1)
    merged_pair["tgt_text_features"] = merged_pair.apply(lambda row: "@".join([row["title_tgt"], row["tgt_sku_pvs_features"], row["tgt_item_pvs_features"]]),axis=1)
    return merged_pair

def get_model_data_v2(dname: str, return_ids: bool = False):
    """
    文本特征为title和公共pvs
    """
    if dname=="train":
        pair_file = "../data/item_train_pair_train.jsonl"
        info_file = "../data/item_train_info.jsonl"
    elif dname=="test":
        pair_file = "../data/item_train_pair_test.jsonl"
        info_file = "../data/item_train_info.jsonl"
    elif dname=="valid":
        pair_file = "../data/item_valid_pair.jsonl"
        info_file = "../data/item_valid_info.jsonl"
    pair = read_data(pair_file)
    info = read_data(info_file)
    if dname=="valid":
        info["image_path"] = info["item_image_name"].apply(lambda name: os.path.join("../data/item_valid_images", name))
    else:
        info["image_path"] = info["item_image_name"].apply(lambda name: os.path.join("../data/item_train_images", name))
    info["item_pvs"] = info["item_pvs"].apply(lambda pvs: pvs.split("#;#") if pd.notna(pvs) else [])
    info["item_pvs"] = info["item_pvs"].apply(lambda pvs: {p.split("#:#")[0]:p.split("#:#")[1] for p in pvs})
    info["sku_pvs"] = info["sku_pvs"].apply(lambda pvs: pvs.split("#;#") if pd.notna(pvs) else [])
    info["sku_pvs"] = info["sku_pvs"].apply(lambda pvs: {p.split("#:#")[0]:p.split("#:#")[1] for p in pvs})
    pair = build_pair_text_feature(pair, info)
    item_id1 = pair["src_item_id"].tolist()
    item_id2 = pair["tgt_item_id"].tolist()
    texts1 = pair["src_text_features"].tolist()
    texts2  = pair["tgt_text_features"].tolist()
    img_paths_1 = pair["image_path_src"].tolist()
    img_paths_2 = pair["image_path_tgt"].tolist()
    result = tuple()
    if return_ids:
       result += (item_id1, item_id2)
    result += (texts1, texts2, img_paths_1, img_paths_2)
    if dname=="valid":
        return result
    else:
        labels = pair["item_label"].apply(int).tolist()
        result += (labels,)
        return result

def cross_preprocess(src_image, tgt_image, image_size=224):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    min_size = min(src_image.size+tgt_image.size)
    single_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
        ])
    src_image = single_transform(src_image)
    tgt_image = single_transform(tgt_image)
    image = torchvision.utils.make_grid([src_image, tgt_image])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        normalize,
        ])
    return transform(image)

def preprocess(image, image_size=224):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    return transform(image)

def get_pretrain_data_v3():
    """
    1. title和图像对;
    2. 正样本对的title拼接和图像拼接;
    """
    texts, img_paths = get_pretrain_data()
    train_texts1, train_texts2, train_img_paths_1, train_img_paths_2, train_labels = get_model_data("train")
    test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, test_labels = get_model_data("test")
    texts1 = train_texts1 + test_texts1
    texts2 = train_texts2 + test_texts2
    img_paths_1 = train_img_paths_1 + test_img_paths_1
    img_paths_2 = train_img_paths_2 + test_img_paths_2
    labels = train_labels +  test_labels
    pair_texts = [t1+"[SEP]"+t2 for t1, t2, label in zip(texts1, texts2, labels) if label==1]
    texts = texts + pair_texts
    pair_img_paths = [(i1, i2) for i1,i2,label in zip(img_paths_1, img_paths_2, labels) if label==1]
    img_paths = img_paths + pair_img_paths
    return texts, img_paths


def get_pretrain_data_v2():
    """
    文本特征为：tilte+公共pvs
    """
    train_texts1, train_texts2, train_img_paths_1, train_img_paths_2, _ = get_model_data_v2("train")
    test_texts1, test_texts2, test_img_paths_1, test_img_paths_2, _ = get_model_data_v2("test")
    valid_texts1, valid_texts2, valid_img_paths_1, valid_img_paths_2 = get_model_data_v2("valid")
    texts1 = train_texts1 + test_texts1 + valid_texts1
    texts2 = train_texts2 + test_texts2 + valid_texts2
    img_paths_1 = train_img_paths_1 + test_img_paths_1 + valid_img_paths_1
    img_paths_2 = train_img_paths_2 + test_img_paths_2 + valid_img_paths_2
    texts = texts1 + texts2
    img_paths = img_paths_1 + img_paths_2
    # 去重
    tmp = list(set(["\t".join([text, img_path]) for text, img_path in zip(texts, img_paths)]))
    texts = [t.split("\t")[0] for t in tmp]
    img_paths = [t.split("\t")[1] for t in tmp]
    print(len(texts))
    return texts, img_paths

def get_pretrain_data():
    """
    文本特征为title
    """
    train_items = read_data("../data/item_train_info.jsonl")
    valid_items = read_data("../data/item_valid_info.jsonl")
    del train_items["image_path"]
    train_items["image_path"] = train_items["item_image_name"].apply(lambda i_name: os.path.join("../data/item_train_images/", i_name))
    valid_items["image_path"] = valid_items["item_image_name"].apply(lambda i_name: os.path.join("../data/item_valid_images/", i_name))
    items = pd.concat([train_items, valid_items])
    texts = items["title"].tolist()
    img_paths = items["image_path"].tolist()
    return texts, img_paths

def get_model_data(dname: str, return_ids: bool = False):
    if dname=="valid":
        # text_feature_file = "../data/item_valid_text_title_industry_name_catename_pvs_feature.pkl"
        text_feature_file = "../data/item_valid_text_title_feature.pkl"
        item_info_file = "../data/item_valid_info.jsonl"
        images_file = "../data/item_valid_images/"
    else:
        text_feature_file = "../data/item_train_text_title_feature.pkl"
        # text_feature_file = "../data/item_train_text_title_industry_name_catename_pvs_feature.pkl"
        item_info_file = "../data/item_train_info.jsonl"
        images_file = "../data/item_train_images/"
    item2text = pickle.load(open(text_feature_file, "rb"))
    items = read_data(item_info_file)
    items["image_path"] = items["item_image_name"].apply(lambda i_name: os.path.join(images_file, i_name))
    items.index = items["item_id"]
    item2imgpath = items["image_path"].to_dict()
    if dname=="train":
        data=read_data("../data/item_train_pair_train.jsonl")
    elif dname=="test":
        data=read_data("../data/item_train_pair_test.jsonl")
    elif dname=="valid":
        data=read_data("../data/item_valid_pair.jsonl")
    data["src_text"] = data["src_item_id"].apply(lambda item_id: item2text[item_id])
    data["tgt_text"] = data["tgt_item_id"].apply(lambda item_id: item2text[item_id])
    data["src_img_path"] = data["src_item_id"].apply(lambda item_id: item2imgpath[item_id])
    data["tgt_img_path"] = data["tgt_item_id"].apply(lambda item_id: item2imgpath[item_id])
    item_id1 = data["src_item_id"].tolist()
    item_id2 = data["tgt_item_id"].tolist()
    texts1 = data["src_text"].tolist()
    texts2 = data["tgt_text"].tolist()
    img_paths_1 = data["src_img_path"].tolist()
    img_paths_2 = data["tgt_img_path"].tolist()
    if "item_label" in data:
        labels = data["item_label"].apply(int).tolist()
        if return_ids:
            return item_id1, item_id2, texts1, texts2, img_paths_1, img_paths_2, labels
        else:
            return texts1, texts2, img_paths_1, img_paths_2, labels
    else:
        return item_id1, item_id2, texts1, texts2, img_paths_1, img_paths_2

class R2D2SingleDataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 img_paths: List[str],
                 max_length=64):
        print("Building Dataset")
        assert len(texts)==len(img_paths)

        self.inputs = []

        for idx in tqdm(range(len(texts))):
            inp = dict()
            text = texts[idx]
            img_path = img_paths[idx]
            label = 0 # 伪标签
            text_inp = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            text_inp = {"src_"+k:v for k,v in text_inp.items()}
            inp.update(text_inp)
            inp["src_pixel_value"] = img_path
            inp["label"] = label
            self.inputs.append(inp)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class R2D2CrossDataset(Dataset):
    def __init__(self,
                 texts1: List[str],
                 texts2: List[str],
                 img_paths_1: List[str],
                 img_paths_2: List[str],
                 labels: List[int],
                 max_length=64):
        print("Building Dataset.")
        assert len(texts1)==len(texts2)
        assert len(texts1)==len(img_paths_1)
        assert len(texts1)==len(img_paths_2)
        assert len(texts1)==len(labels)
        self.inputs = []
        half_max_length = int(max_length/2)

        for idx in tqdm(range(len(texts1))):
            inp = dict()
            text1 = texts1[idx]
            text2= texts2[idx]
            if len(text1)+len(text2)>max_length:
                text1 =  text1[:half_max_length]
                text2 = text2[:max_length-half_max_length]
            text12 = text1 + "[SEP]" + text2
            img_path_1 = img_paths_1[idx]
            img_path_2 = img_paths_2[idx]
            label = labels[idx]
            text12_inp = tokenizer(text12, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            inp.update(text12_inp)
            inp["src_pixel_value"] = img_path_1
            inp["tgt_pixel_value"] = img_path_2
            inp["label"] = label
            self.inputs.append(inp)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

class R2D2PairDataset(Dataset):
    def __init__(self,
                 texts1: List[str],
                 texts2: List[str],
                 img_paths_1: List[str],
                 img_paths_2: List[str],
                 labels: List[int],
                 max_length=64):
        print("Building Dataset.")
        assert len(texts1)==len(texts2)
        assert len(texts1)==len(img_paths_1)
        assert len(texts1)==len(img_paths_2)
        assert len(texts1)==len(labels)
        self.inputs = []

        for idx in tqdm(range(len(texts1))):
            inp = dict()
            text1 = texts1[idx]
            text2= texts2[idx]
            img_path_1 = img_paths_1[idx]
            img_path_2 = img_paths_2[idx]
            label = labels[idx]
            text1_inp = tokenizer(text1, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            text2_inp = tokenizer(text2, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            text1_inp = {"src_"+k:v for k,v in text1_inp.items()}
            text2_inp = {"tgt_"+k:v for k,v in text2_inp.items()}
            inp.update(text1_inp)
            inp.update(text2_inp)
            inp["src_pixel_value"] = img_path_1
            inp["tgt_pixel_value"] = img_path_2
            inp["label"] = label
            self.inputs.append(inp)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def r2d2_single_collate_fn(features: List[Dict]):
    # text
    batch_src_input_ids = [feature["src_input_ids"] for feature in features]
    batch_src_attention_mask = [feature["src_attention_mask"] for feature in features]
    batch_src_token_type_ids = [feature["src_token_type_ids"] for feature in features]
    batch_src_input_ids = pad_sequence(batch_src_input_ids, batch_first=True, padding_value=0).squeeze(dim=1)
    batch_src_attention_mask = pad_sequence(batch_src_attention_mask, batch_first=True, padding_value=0).squeeze(dim=1)
    batch_src_token_type_ids = pad_sequence(batch_src_token_type_ids, batch_first=True, padding_value=0).squeeze(dim=1)

    # image
    batch_src_pixel_value = [preprocess(Image.open(feature["src_pixel_value"]).convert("RGB")).unsqueeze(0) for feature in features]
    batch_src_pixel_value = torch.concat(batch_src_pixel_value, dim=0)

    # label
    batch_label = torch.LongTensor([feature["label"] for feature in features])

    return {"src_input_ids": batch_src_input_ids,
            "src_attention_mask": batch_src_attention_mask,
            "src_token_type_ids": batch_src_token_type_ids,
            "src_pixel_value": batch_src_pixel_value,
            "labels": batch_label}

def r2d2_pair_pretrain_collate_fn(features: List[Dict]):
    # text
    batch_src_input_ids = [feature["src_input_ids"] for feature in features]
    batch_src_attention_mask = [feature["src_attention_mask"] for feature in features]
    batch_src_token_type_ids = [feature["src_token_type_ids"] for feature in features]
    batch_src_input_ids = pad_sequence(batch_src_input_ids, batch_first=True, padding_value=0).squeeze(dim=1)
    batch_src_attention_mask = pad_sequence(batch_src_attention_mask, batch_first=True, padding_value=0).squeeze(dim=1)
    batch_src_token_type_ids = pad_sequence(batch_src_token_type_ids, batch_first=True, padding_value=0).squeeze(dim=1)

    # image
    batch_src_pixel_value = []
    for feature in features:
        src_pixel_value = feature["src_pixel_value"]
        if isinstance(src_pixel_value, str):
            # 单图像路径
            batch_src_pixel_value.append(preprocess(Image.open(src_pixel_value).convert("RGB")).unsqueeze(0))
        else:
            # 双图像路径
            batch_src_pixel_value.append(cross_preprocess(Image.open(src_pixel_value[0]).convert("RGB"), Image.open(src_pixel_value[1]).convert("RGB")).unsqueeze(0))
    batch_src_pixel_value = torch.concat(batch_src_pixel_value, dim=0)

    # label
    batch_label = torch.LongTensor([feature["label"] for feature in features])

    return {"src_input_ids": batch_src_input_ids,
            "src_attention_mask": batch_src_attention_mask,
            "src_token_type_ids": batch_src_token_type_ids,
            "src_pixel_value": batch_src_pixel_value,
            "labels": batch_label}

def r2d2_cross_collate_fn(features: List[Dict]):
    # text
    batch_cross_input_ids = [feature["input_ids"] for feature in features]
    batch_cross_attention_mask = [feature["attention_mask"] for feature in features]
    batch_cross_token_type_ids = [feature["token_type_ids"] for feature in features]

    batch_cross_input_ids = pad_sequence(batch_cross_input_ids, batch_first=True, padding_value=0).squeeze()
    batch_cross_attention_mask = pad_sequence(batch_cross_attention_mask, batch_first=True, padding_value=0).squeeze()
    batch_cross_token_type_ids = pad_sequence(batch_cross_token_type_ids, batch_first=True, padding_value=0).squeeze()

    # image
    batch_cross_pixel_value = [cross_preprocess(Image.open(feature["src_pixel_value"]).convert("RGB"),Image.open(feature["tgt_pixel_value"]).convert("RGB")).unsqueeze(0) for feature in features]
    batch_cross_pixel_value = torch.concat(batch_cross_pixel_value, dim=0)

    # label
    batch_label = torch.LongTensor([feature["label"] for feature in features])

    return {"input_ids": batch_cross_input_ids,
            "attention_mask": batch_cross_attention_mask,
            "token_type_ids": batch_cross_token_type_ids,
            "pixel_value": batch_cross_pixel_value,
            "labels": batch_label}


def r2d2_pair_collate_fn(features: List[Dict]):
    # text
    batch_src_input_ids = [feature["src_input_ids"] for feature in features]
    batch_src_attention_mask = [feature["src_attention_mask"] for feature in features]
    batch_src_token_type_ids = [feature["src_token_type_ids"] for feature in features]
    batch_tgt_input_ids = [feature["tgt_input_ids"] for feature in features]
    batch_tgt_attention_mask = [feature["tgt_attention_mask"] for feature in features]
    batch_tgt_token_type_ids = [feature["tgt_token_type_ids"] for feature in features]

    batch_src_input_ids = pad_sequence(batch_src_input_ids, batch_first=True, padding_value=0).squeeze()
    batch_src_attention_mask = pad_sequence(batch_src_attention_mask, batch_first=True, padding_value=0).squeeze()
    batch_src_token_type_ids = pad_sequence(batch_src_token_type_ids, batch_first=True, padding_value=0).squeeze()
    batch_tgt_input_ids = pad_sequence(batch_tgt_input_ids, batch_first=True, padding_value=0).squeeze()
    batch_tgt_attention_mask = pad_sequence(batch_tgt_attention_mask, batch_first=True, padding_value=0).squeeze()
    batch_tgt_token_type_ids = pad_sequence(batch_tgt_token_type_ids, batch_first=True, padding_value=0).squeeze()

    # image
    batch_src_pixel_value = [preprocess(Image.open(feature["src_pixel_value"]).convert("RGB")).unsqueeze(0) for feature in features]
    batch_tgt_pixel_value = [preprocess(Image.open(feature["tgt_pixel_value"]).convert("RGB")).unsqueeze(0) for feature in features]
    batch_src_pixel_value = torch.concat(batch_src_pixel_value, dim=0)
    batch_tgt_pixel_value = torch.concat(batch_tgt_pixel_value, dim=0)

    # label
    batch_label = torch.LongTensor([feature["label"] for feature in features])

    return {"src_input_ids": batch_src_input_ids,
            "src_attention_mask": batch_src_attention_mask,
            "src_token_type_ids": batch_src_token_type_ids,
            "src_pixel_value": batch_src_pixel_value,
            "tgt_input_ids": batch_tgt_input_ids,
            "tgt_attention_mask": batch_tgt_attention_mask,
            "tgt_token_type_ids": batch_tgt_token_type_ids,
            "tgt_pixel_value": batch_tgt_pixel_value,
            "labels": batch_label}

if __name__ == "__main__":
    # get_model_data_v2("valid")
    texts1, texts2, img_paths_1, img_paths_2, labels = get_model_data("test")
    dataset = R2D2CrossDataset(texts1, texts2, img_paths_1, img_paths_2, labels, max_length=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=r2d2_cross_collate_fn)
    # dataset = R2D2SingleDataset(texts1[:65], img_paths_1[:65])
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=r2d2_single_collate_fn)
    batch = next(iter(dataloader))
    batch = {k:v.to("cuda") for k,v in batch.items()}
    model = R2D2FullCrossForAlign()
    model = model.to("cuda")
    output = model(**batch)
    print(output)
    pass
