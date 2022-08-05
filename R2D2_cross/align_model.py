import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.bert import BertConfig, BertModel
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from models.r2d2 import r2d2, MyR2D2
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

@dataclass
class AlignOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings_1: Optional[torch.FloatTensor] = None
    embeddings_2: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class MLPLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y)

class R2D2CrossForAlign(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super(R2D2CrossForAlign, self).__init__()
        if checkpoint_path:
            self.model = r2d2(pretrained=checkpoint_path)
        else:
            self.model = MyR2D2()

        bert_config='./checkpoints/r2d2_config/bert_config.json'
        cross_config = BertConfig.from_json_file(bert_config)
        cross_config.encoder_width = 768
        cross_config.num_hidden_layers = 2
        cross_config.add_cross_attention=False
        self.cross_layer = BertModel(config=cross_config, add_pooling_layer=False)
        self.mlp = MLPLayer(768,768)
        self.src_proj = MLPLayer(768, 768)
        self.tgt_proj = MLPLayer(768, 768)
        self.sim = Similarity()
        self.loss_func = nn.CrossEntropyLoss()

    def train_forward(self,
                      src_input_ids=None,
                      src_attention_mask=None,
                      src_token_type_ids=None,
                      src_pixel_value=None,
                      tgt_input_ids=None,
                      tgt_attention_mask=None,
                      tgt_token_type_ids=None,
                      tgt_pixel_value=None,
                      labels=None):
        src_text_hidden_states = self.model(input_ids=src_input_ids,
                                           attention_mask=src_attention_mask,
                                           token_type_ids=src_token_type_ids,
                                           pixel_value=src_pixel_value)

        tgt_text_hidden_states = self.model(input_ids=tgt_input_ids,
                                           attention_mask=tgt_attention_mask,
                                           token_type_ids=tgt_token_type_ids,
                                           pixel_value=tgt_pixel_value)
        text_hidden_states =  torch.concat([src_text_hidden_states, tgt_text_hidden_states], axis=1)
        attention_mask = torch.concat([src_attention_mask, tgt_attention_mask], axis=1)
        cls = self.cross_layer(inputs_embeds=text_hidden_states,
                               attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # src_text_cls = src_text_hidden_state[:, 0, :]
        # tgt_text_cls = tgt_text_hidden_state[:, 0, :]
        # cls = torch.concat([src_text_cls, tgt_text_cls, torch.absolute(src_text_cls - tgt_text_cls)], axis=-1)
        # keep dropout and forward twice
        cls = self.mlp(cls)     
        #cls1 = self.mlp(cls)
        #kl_loss = compute_kl_loss(cls, cls1)
        
        src_cls = self.src_proj(cls)
        tgt_cls = self.tgt_proj(cls)
        src_cls = src_cls/src_cls.norm(dim=-1, keepdim=True)
        tgt_cls = tgt_cls/tgt_cls.norm(dim=-1, keepdim=True)
        cos_sim = self.sim(src_cls, tgt_cls)

        loss = F.mse_loss(cos_sim,labels.float())
        return AlignOutput(loss=loss, logits=cos_sim, embeddings_1=src_cls, embeddings_2=tgt_cls, labels=labels)

    def emb_forward(self,
                      src_input_ids=None,
                      src_attention_mask=None,
                      src_token_type_ids=None,
                      src_pixel_value=None,
                      tgt_input_ids=None,
                      tgt_attention_mask=None,
                      tgt_token_type_ids=None,
                      tgt_pixel_value=None,
                      labels=None):
        src_text_hidden_state = self.model(input_ids=src_input_ids,
                                           attention_mask=src_attention_mask,
                                           token_type_ids=src_token_type_ids,
                                           pixel_value=src_pixel_value)

        tgt_text_hidden_state = self.model(input_ids=tgt_input_ids,
                                           attention_mask=tgt_attention_mask,
                                           token_type_ids=tgt_token_type_ids,
                                           pixel_value=tgt_pixel_value)

        # cls = self.cross_layer(encoder_embeds=src_text_hidden_state,
        #                        attention_mask=src_attention_mask,
        #                        encoder_hidden_states=tgt_text_hidden_state,
        #                        encoder_attention_mask=tgt_attention_mask).last_hidden_state[:, 0, :]
        src_text_cls = src_text_hidden_state[:, 0, :]
        tgt_text_cls = tgt_text_hidden_state[:, 0, :]
        cls = torch.concat([src_text_cls, tgt_text_cls, torch.absolute(src_text_cls - tgt_text_cls)], axis=-1)
        cls = self.mlp(cls)
        src_cls = self.src_proj(cls)
        tgt_cls = self.tgt_proj(cls)
        src_cls = src_cls/src_cls.norm(dim=-1, keepdim=True)
        tgt_cls = tgt_cls/tgt_cls.norm(dim=-1, keepdim=True)
        return AlignOutput(embeddings_1=src_cls, embeddings_2=tgt_cls, labels=labels)

    def forward(self,
                src_input_ids=None,
                src_attention_mask=None,
                src_token_type_ids=None,
                src_pixel_value=None,
                tgt_input_ids=None,
                tgt_attention_mask=None,
                tgt_token_type_ids=None,
                tgt_pixel_value=None,
                labels=None,
                item_emb=False):
        if item_emb:
            return self.emb_forward(src_input_ids,
                                    src_attention_mask,
                                    src_token_type_ids,
                                    src_pixel_value,
                                    tgt_input_ids,
                                    tgt_attention_mask,
                                    tgt_token_type_ids,
                                    tgt_pixel_value,
                                    labels)
        else:
            return self.train_forward(src_input_ids,
                                      src_attention_mask,
                                      src_token_type_ids,
                                      src_pixel_value,
                                      tgt_input_ids,
                                      tgt_attention_mask,
                                      tgt_token_type_ids,
                                      tgt_pixel_value,
                                      labels)

class R2D2FullCrossForAlign(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super(R2D2FullCrossForAlign, self).__init__()
        if checkpoint_path:
            self.model = r2d2(pretrained=checkpoint_path)
        else:
            self.model = MyR2D2()
        self.mlp = MLPLayer(768, 768)
        self.sim = Similarity()
        self.src_proj = MLPLayer(768, 768)
        self.tgt_proj = MLPLayer(768, 768)
        self.loss_func = nn.CrossEntropyLoss()

    def train_forward(self,
                      input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      pixel_value=None,
                      labels=None):
        text_hidden_state = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       pixel_value=pixel_value)
        cls = text_hidden_state[:, 0, :]
        src_cls = self.src_proj(cls)
        tgt_cls = self.tgt_proj(cls)
        src_cls = src_cls/src_cls.norm(dim=-1, keepdim=True)
        tgt_cls = tgt_cls/tgt_cls.norm(dim=-1, keepdim=True)
        cos_sim = self.sim(src_cls, tgt_cls)
        loss = F.mse_loss(cos_sim,labels.float())
        return AlignOutput(loss=loss, logits=cos_sim, embeddings_1=src_cls, embeddings_2=tgt_cls, labels=labels)

    def emb_forward(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    pixel_value=None,
                    labels=None):
        text_hidden_state = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       pixel_value=pixel_value)
        cls = text_hidden_state[:, 0, :]
        src_cls = self.src_proj(cls)
        tgt_cls = self.tgt_proj(cls)
        src_cls = src_cls/src_cls.norm(dim=-1, keepdim=True)
        tgt_cls = tgt_cls/tgt_cls.norm(dim=-1, keepdim=True)
        return AlignOutput(embeddings_1=src_cls, embeddings_2=tgt_cls, labels=labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                pixel_value=None,
                src_pixel_value=None,
                tgt_pixel_value=None,
                labels=None,
                item_emb=False):
        if item_emb:
            return self.emb_forward(input_ids,
                                    attention_mask,
                                    token_type_ids,
                                    pixel_value)
        else:
            return self.train_forward(input_ids,
                                      attention_mask,
                                      token_type_ids,
                                      pixel_value,
                                      labels)

class R2D2ForAlign(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super(R2D2ForAlign, self).__init__()
        if checkpoint_path:
            self.model = r2d2(pretrained=checkpoint_path)
        else:
            self.model = MyR2D2()
        self.mlp = MLPLayer(768, 768)
        self.sim = Similarity()
        self.loss_func = nn.CrossEntropyLoss()

    def train_forward(self,
                      src_input_ids=None,
                      src_attention_mask=None,
                      src_token_type_ids=None,
                      src_pixel_value=None,
                      tgt_input_ids=None,
                      tgt_attention_mask=None,
                      tgt_token_type_ids=None,
                      tgt_pixel_value=None,
                      labels=None):
        src_text_hidden_state = self.model(input_ids=src_input_ids,
                                           attention_mask=src_attention_mask,
                                           token_type_ids=src_token_type_ids,
                                           pixel_value=src_pixel_value)

        tgt_text_hidden_state = self.model(input_ids=tgt_input_ids,
                                           attention_mask=tgt_attention_mask,
                                           token_type_ids=tgt_token_type_ids,
                                           pixel_value=tgt_pixel_value)
        src_text_cls = src_text_hidden_state[:, 0, :]
        tgt_text_cls = tgt_text_hidden_state[:, 0, :]
        # 1. text
        src_cls = self.mlp(src_text_cls)
        tgt_cls = self.mlp(tgt_text_cls)
        # 2. image
        # src_cls = self.mlp(src_img_cls)
        # tgt_cls = self.mlp(tgt_img_cls)
        # 3. both
        # src_cls = self.mlp(torch.concat([src_text_cls, src_img_cls, torch.absolute(src_text_cls - src_img_cls)], axis=-1))
        # tgt_cls = self.mlp(torch.concat([tgt_text_cls, tgt_img_cls, torch.absolute(tgt_text_cls - tgt_img_cls)], axis=-1))
        src_cls = src_cls/src_cls.norm(dim=-1, keepdim=True)
        tgt_cls = tgt_cls/tgt_cls.norm(dim=-1, keepdim=True)
        cos_sim = self.sim(src_cls, tgt_cls)
        loss = F.mse_loss(cos_sim,labels.float())
        return AlignOutput(loss=loss, logits=cos_sim, embeddings_1=src_cls, embeddings_2=tgt_cls, labels=labels)

    def emb_forward(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    pixel_value=None):
        text_cls, img_cls = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       pixel_value=pixel_value)
        embeddings = self.mlp(text_cls)
        embeddings = embeddings/embeddings.norm(dim=-1, keepdim=True)
        return AlignOutput(embeddings_1=embeddings)
        

    def forward(self,
                src_input_ids=None,
                src_attention_mask=None,
                src_token_type_ids=None,
                src_pixel_value=None,
                tgt_input_ids=None,
                tgt_attention_mask=None,
                tgt_token_type_ids=None,
                tgt_pixel_value=None,
                labels=None,
                item_emb=False):
        if item_emb:
            return self.emb_forward(src_input_ids,
                                    src_attention_mask,
                                    src_token_type_ids,
                                    src_pixel_value)
        else:
            return self.train_forward(src_input_ids,
                                      src_attention_mask,
                                      src_token_type_ids,
                                      src_pixel_value,
                                      tgt_input_ids,
                                      tgt_attention_mask,
                                      tgt_token_type_ids,
                                      tgt_pixel_value,
                                      labels)
