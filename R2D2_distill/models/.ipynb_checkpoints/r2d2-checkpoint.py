import os
import transformers
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass
from models.bert import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import ModelOutput

@dataclass
class R2D2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_mlm: Optional[torch.FloatTensor] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    image_embeddings: Optional[torch.FloatTensor] = None

class ContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features1, features2):
        # features1 = features1.unsqueeze(1)
        # features2 = features2.unsqueeze(0)
        cos_sim = features1 @ features2.T / self.temp
        # cos_sim = self.cos(features1, features2)/self.temp
        labels = torch.arange(cos_sim.size(0)).long().to("cuda")
        return self.loss_fct(cos_sim, labels), cos_sim

class R2D2ForPretrain(nn.Module):
    def __init__(self, image_size=224, embed_dim=768):
        super().__init__()
        vision_width = 1024
        clip_config=transformers.CLIPConfig.from_pretrained('./checkpoints/r2d2_config/vision_config.json')
        clip = transformers.CLIPModel(config=clip_config).eval()
        self.visual_encoder = clip.vision_model
        num_patches = (image_size // 14)**2
        if self.visual_encoder.embeddings.num_patches != num_patches:
            self.visual_encoder.embeddings.num_patches = num_patches
            self.visual_encoder.embeddings.position_embedding = nn.Embedding(num_patches + 1,
                                                                             self.visual_encoder.embeddings.embed_dim)
            self.visual_encoder.embeddings.position_ids = torch.arange(num_patches + 1).unsqueeze(0)

        text_config = BertConfig.from_pretrained('./checkpoints/hfl_roberta')
        self.tokenizer = BertTokenizer.from_pretrained('./checkpoints/hfl_roberta')
        self.text_encoder = transformers.BertModel(config=text_config)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # create joint layers
        bert_config='./checkpoints/r2d2_config/bert_config.json'
        encoder_config = BertConfig.from_json_file(bert_config)
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = 6
        self.text_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        encoder_config.encoder_width = text_width
        encoder_config.vocab_size = 21128
        self.img_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        self.img_joint_proj = nn.Linear(vision_width, text_width)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(text_width, 2)
        self.itm_head_i = nn.Linear(text_width, 2)
        self.mlm_head = BertOnlyMLMHead(encoder_config)
        self.contrastive_loss = ContrastiveLoss(temp=0.05)

    def forward(self,
                src_input_ids=None,
                src_attention_mask=None,
                src_token_type_ids=None,
                src_pixel_value=None,
                labels=None):
        input_ids = src_input_ids
        attention_mask = src_attention_mask
        token_type_ids = src_token_type_ids
        pixel_value = src_pixel_value
        bs = input_ids.size(0)
        text_feats = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_embed = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        image_feats = self.visual_encoder(pixel_value).last_hidden_state
        image_embed = F.normalize(self.vision_proj(image_feats[:, 0, :]), dim=-1)

        encoder_output = image_feats
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to("cuda")
        loss_itc = None
        loss_itm = None

        ## 1. contrastive loss
        loss_itc_t2i, sim_t2i = self.contrastive_loss(text_embed, image_embed)
        loss_itc_i2t, sim_i2t = self.contrastive_loss(image_embed, text_embed)
        # sim_i2t = sim_t2i.clone().T
        loss_itc = (loss_itc_t2i + loss_itc_i2t)/2

        text_output_pos = self.text_joint_layer(
            encoder_embeds=text_feats,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )

        image_output_pos = self.img_joint_layer(encoder_embeds=self.img_joint_proj(encoder_output),
                                       attention_mask=encoder_att,
                                       encoder_hidden_states=text_feats,
                                       encoder_attention_mask=attention_mask)

        ## 2. image-text pair match loss
        with torch.no_grad():
            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_feats[neg_idx])
            text_atts_neg.append(attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_feats[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        text_embeds_all = torch.cat([text_feats, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_feats],dim=0)
        image_atts_all = torch.cat([encoder_att, encoder_att],dim=0)

        text_output_neg = self.text_joint_layer(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        image_output_neg = self.img_joint_layer(encoder_embeds=self.img_joint_proj(image_embeds_all),
                                       attention_mask=image_atts_all,
                                       encoder_hidden_states=text_embeds_all,
                                       encoder_attention_mask=text_atts_all)

        text_embeddings = torch.cat([text_output_pos.last_hidden_state[:, 0, :], text_output_neg.last_hidden_state[:, 0, :]], dim=0)
        image_embeddings = torch.cat([image_output_pos.last_hidden_state[:, 0, :], image_output_neg.last_hidden_state[:, 0, :]], dim=0)
        text_output = self.itm_head(text_embeddings)
        image_output = self.itm_head_i(image_embeddings)
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to("cuda")
        loss_itm = F.cross_entropy(text_output, itm_labels) + F.cross_entropy(image_output, itm_labels)

        ## 3. mlm loss
        loss_mlm = None
        mlm_input_ids = input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        mlm_input_ids, labels = self.mask(mlm_input_ids, self.text_encoder.config.vocab_size, "cuda", targets=labels, probability_matrix=probability_matrix)
        mlm_text_feats = self.text_encoder(mlm_input_ids, attention_mask=attention_mask).last_hidden_state
        mlm_output = self.text_joint_layer(
            encoder_embeds=mlm_text_feats,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        mlm_prediction_scores = self.mlm_head(mlm_output[0])
        loss_mlm = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss_mlm = loss_fct(mlm_prediction_scores.view(-1, self.text_encoder.config.vocab_size), labels.view(-1))

        loss = loss_itc + loss_itm + loss_mlm

        return R2D2Output(loss=loss, loss_itc=loss_itc, loss_itm=loss_itm, loss_mlm=loss_mlm, text_embeddings=None, image_embeddings=None)

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

class MyR2D2(nn.Module):
    def __init__(self, image_size=224, embed_dim=768):
        super().__init__()
        vision_width = 1024
        clip_config=transformers.CLIPConfig.from_pretrained('./checkpoints/r2d2_config/vision_config.json')
        clip = transformers.CLIPModel(config=clip_config).eval()
        self.visual_encoder = clip.vision_model
        num_patches = (image_size // 14)**2
        if self.visual_encoder.embeddings.num_patches != num_patches:
            self.visual_encoder.embeddings.num_patches = num_patches
            self.visual_encoder.embeddings.position_embedding = nn.Embedding(num_patches + 1,
                                                                             self.visual_encoder.embeddings.embed_dim)
            self.visual_encoder.embeddings.position_ids = torch.arange(num_patches + 1).unsqueeze(0)

        text_config = BertConfig.from_pretrained('./checkpoints/hfl_roberta')
        self.text_encoder = transformers.BertModel(config=text_config)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # create joint layers
        bert_config='./checkpoints/r2d2_config/bert_config.json'
        encoder_config = BertConfig.from_json_file(bert_config)
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = 6
        self.text_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        encoder_config.encoder_width = text_width
        encoder_config.vocab_size = 21128
        self.img_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        self.img_joint_proj = nn.Linear(vision_width, text_width)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(text_width, 2)
        self.itm_head_i = nn.Linear(text_width, 2)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                pixel_value=None):
        text_feats = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_embed = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        image_feats = self.visual_encoder(pixel_value).last_hidden_state
        image_embed = F.normalize(self.vision_proj(image_feats[:, 0, :]), dim=-1)

        # encoder_output = image_feats.repeat(len(texts), 1, 1)
        encoder_output = image_feats
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to("cuda")

        text_output = self.text_joint_layer(
            encoder_embeds=text_feats,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )

        image_output = self.img_joint_layer(encoder_embeds=self.img_joint_proj(encoder_output),
                                       attention_mask=encoder_att,
                                       encoder_hidden_states=text_feats,
                                       encoder_attention_mask=attention_mask)

        return text_output.last_hidden_state[:, 0, :], image_output.last_hidden_state[:, 0, :]
        

class R2D2(nn.Module):
    def __init__(
        self,
        image_size=224,
        embed_dim=768,
    ):
        """
        Args:
            image_size (int): input image size
            embed_dim (int): output embedding size
        """
        super().__init__()
        vision_width = 1024
        clip_config=transformers.CLIPConfig.from_pretrained('./checkpoints/r2d2_config/vision_config.json')
        clip = transformers.CLIPModel(config=clip_config).eval()
        self.visual_encoder = clip.vision_model
        num_patches = (image_size // 14)**2
        if self.visual_encoder.embeddings.num_patches != num_patches:
            self.visual_encoder.embeddings.num_patches = num_patches
            self.visual_encoder.embeddings.position_embedding = nn.Embedding(num_patches + 1,
                                                                             self.visual_encoder.embeddings.embed_dim)
            self.visual_encoder.embeddings.position_ids = torch.arange(num_patches + 1).unsqueeze(0)

        self.tokenizer = BertTokenizer.from_pretrained('./checkpoints/hfl_roberta')
        text_config = BertConfig.from_pretrained('./checkpoints/hfl_roberta')
        self.text_encoder = transformers.BertModel(config=text_config)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # create joint layers
        bert_config='./checkpoints/r2d2_config/bert_config.json'
        encoder_config = BertConfig.from_json_file(bert_config)
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = 6
        self.text_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        encoder_config.encoder_width = text_width        
        encoder_config.vocab_size = 21128
        self.img_joint_layer = BertModel(config=encoder_config, add_pooling_layer=False)
        self.img_joint_proj = nn.Linear(vision_width, text_width)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.itm_head = nn.Linear(text_width, 2)
        self.itm_head_i = nn.Linear(text_width, 2)


    def tokenize_text(self, text):
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        return tokenized_text

    def encode_text(self, text):
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask).last_hidden_state
        text_embed = F.normalize(self.text_proj(text_output[:, 0, :]), dim=-1)
        return text_output, text_embed

    def encode_image(self, image):
        image_output = self.visual_encoder(image).last_hidden_state
        image_embed = F.normalize(self.vision_proj(image_output[:, 0, :]), dim=-1)
        return image_output, image_embed  

def r2d2(pretrained='', **kwargs):
    # model = R2D2(**kwargs)
    model = MyR2D2(**kwargs)
    # model = R2D2ForPretrain(**kwargs)
    if pretrained.endswith("pth"):
        model, _ = load_checkpoint(model, pretrained)
    elif pretrained.endswith("bin"):
        model = load_pytorch_model(pretrained, model)
    else:
        raise Exception("Unsupport model")
    return model

def load_pytorch_model(model_path,model,strict=False):
    print(f"Loading model from {model_path}")
    tmp_model = torch.load(model_path)
    if hasattr(tmp_model,"module"):
        model.load_state_dict(tmp_model.module, strict=strict)
    else:
        model.load_state_dict(tmp_model, strict=strict)
    return model
    
def load_checkpoint(model, url_or_filename):
    print(f"Loading model from {url_or_filename}")
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    load_msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, load_msg
