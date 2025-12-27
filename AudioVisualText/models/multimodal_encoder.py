import torch
from torch import nn,Tensor
import json
import math
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from einops import rearrange
from typing import Optional,Tuple,Type,Any,List,Mapping
from transformers import CLIPVisionModel, CLIPImageProcessor,BertTokenizer

from models.Qformer import BertConfig,BertLMHeadModel
from models.beats.BEATs import BEATs,BEATsConfig


def maybe_autocast(dtype=torch.bfloat16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)
    

class VisualEncoder(nn.Module):

    def __init__(
        self,
        model_name_or_path = 'clip-vit-large-patch14',
        select_layer_list = [-11,-1],
        select_feature = 'patch',
    ) -> None:
        super().__init__()
        
        self.select_layer_list = select_layer_list
        self.select_feature = select_feature

        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path,local_files_only=True)
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name_or_path,local_files_only=True)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()


    def feature_select(self, image_forward_outs):
        features = []
        for lyr in self.select_layer_list:
            image_features = image_forward_outs.hidden_states[lyr]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            features.append(image_features)
        return features
    

    @torch.no_grad()
    def encode_video(self,video):
        b,t,c,h,w = video.shape
        video = video.reshape(b*t,c,h,w)
        video_forward_outs = self.vision_tower(video, output_hidden_states=True)
        video_feature = self.feature_select(video_forward_outs)
        return video_feature


    def forward(self,video) -> List[Tensor]:
        b,t,c,h,w = video.shape
        feature_list = self.encode_video(video)
        new_feature_list = []
        for feature in feature_list:
            bt,n,d = feature.shape
            feature = feature.reshape(b,t*n,d)
            new_feature_list.append(feature)

        return new_feature_list
    

class VLProjector(nn.Module):
    def __init__(
        self,
        bert_ckpt_path = 'google-bert-base-uncased', 
        hidden_size = 1024,
        image_token_nums = 256,
        num_query_token = 32, 
        num_hidden_layers = 2, 
        d_model = 3584,
        depth = 2
    ) -> None:
        super().__init__()
        self.num_query_token = num_query_token
        self.image_token_nums = image_token_nums
        self.visual_ln = nn.LayerNorm(hidden_size)

        self.tokenizer = BertTokenizer.from_pretrained(bert_ckpt_path, local_files_only=True, truncation_side='right')
        
        encoder_config = BertConfig.from_pretrained(bert_ckpt_path,local_files_only=True)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = hidden_size
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        # encoder_config.query_length = num_query_token
        self.visual_Qformer = BertLMHeadModel(config=encoder_config)
        self.visual_query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        self.visual_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        self.visual_proj = build_mlp(depth=depth,hidden_size=encoder_config.hidden_size,output_hidden_size=d_model)
        

    def forward(self,visual_feature,question):
        '''
            visual_feature: b,t*n,d
            text_ids: b,L
        '''
        device = visual_feature.device
        b,tn,dim = visual_feature.shape
        t = tn // self.image_token_nums
        visual_feature = visual_feature.reshape(b*t,self.image_token_nums,-1)

        visual_feature = self.visual_ln(visual_feature)
        visual_atts = torch.ones(visual_feature.size()[:-1], dtype=torch.int32, device=device) # bt,n
        
        query_tokens = self.visual_query_tokens.expand(visual_feature.shape[0], -1, -1) # bt,32,d
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.int32).to(device) # bt,32
        
        if question is not None:
            text_Qformer = self.tokenizer(
                question,
                padding='longest',
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_atts = text_Qformer.attention_mask.unsqueeze(1).expand(-1,t,-1).reshape(b*t,-1)
            text_input_ids = text_Qformer.input_ids.unsqueeze(1).expand(-1,t,-1).reshape(b*t,-1)

            Qformer_atts = torch.cat([query_atts,text_atts],dim=1) # bt,L
            # print('input_ids: ',text_input_ids.device,' text_atts: ',text_atts.device)
            query_output = self.visual_Qformer.bert(
                text_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=visual_feature,
                encoder_attention_mask=visual_atts,
                return_dict=True,
            )
            # print('query output...')
        else:
            query_output = self.visual_Qformer.bert(
                attention_mask=query_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=visual_feature,
                encoder_attention_mask=visual_atts,
                return_dict=True,
            )
        
        visual_embeds = query_output.last_hidden_state # bt,32,d
        visual_embeds = self.visual_proj(visual_embeds[:,:self.num_query_token])
        visual_embeds = visual_embeds.reshape(b,t*self.num_query_token,-1) # b,t*32,dim
        return visual_embeds



class AudioEncoder(nn.Module):

    def __init__(
        self, 
        ckpt_path = 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    ) -> None:
        super().__init__()

        # BEATs
        beats_ckpt = torch.load(ckpt_path, map_location='cpu')
        beats_cfg = BEATsConfig(beats_ckpt['cfg'])
        beats_cfg.encoder_layerdrop = 0.
        self.audio_encoder = BEATs(beats_cfg)
        self.audio_encoder.load_state_dict(beats_ckpt['model'],strict=False)
        self.audio_encoder.requires_grad_(False)
        self.audio_encoder.eval()
        self.audio_encoder.training = False


    @torch.no_grad()
    def encode_audio(self,audio):
        audio_padding_mask = torch.zeros(audio.shape[:-1],device=audio.device).bool()
        audio_embeds, _ = self.audio_encoder.extract_features(audio, padding_mask=audio_padding_mask, feature_only=True)
        return audio_embeds
    

    def forward(self,audio):
        # audio: b,t,L,128
        b,t,L,d = audio.shape
        audio = audio.reshape(b*t,L,d)
        audio_embeds = self.encode_audio(audio) # bt,n,d
        n = audio_embeds.shape[1]
        audio_embeds = audio_embeds.reshape(b,t,n,-1)
        return audio_embeds


class ALProjector(nn.Module):
    def __init__(
        self, 
        bert_ckpt_path = 'google-bert-base-uncased', 
        hidden_size = 768, 
        num_query_token = 32, 
        num_hidden_layers = 2, 
        d_model = 3584, 
        depth = 2
    ) -> None:
        super().__init__()

        self.audio_ln = nn.LayerNorm(hidden_size)
        self.num_query_token = num_query_token
        self.tokenizer = BertTokenizer.from_pretrained(bert_ckpt_path, local_files_only=True, truncation_side='right')
        # tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    
        encoder_config = BertConfig.from_pretrained(bert_ckpt_path,local_files_only=True)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = hidden_size
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        # encoder_config.query_length = num_query_token
        self.audio_Qformer = BertLMHeadModel(config=encoder_config)
        self.audio_query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        self.audio_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.audio_proj = build_mlp(depth=depth,hidden_size=encoder_config.hidden_size,output_hidden_size=d_model)
        # print('init al_projector finished...')   

    def forward(self,audio_feature,question):
        '''
            audio_feature: b,t,n,d
            text_ids: b,L
        '''
        device = audio_feature.device
        b,t,n,dim = audio_feature.shape
        audio_feature = audio_feature.reshape(b*t, n, -1)

        audio_feature = self.audio_ln(audio_feature)
        audio_atts = torch.ones(audio_feature.size()[:-1], dtype=torch.int32, device=device) # bt,n
        
        query_tokens = self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1) # bt,32,d
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.int32).to(device) # bt,32
        if question is not None:
            text_Qformer = self.tokenizer(
                question,
                padding='longest',
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_atts = text_Qformer.attention_mask.unsqueeze(1).expand(-1,t,-1).reshape(b*t,-1) # bt,n
            text_input_ids = text_Qformer.input_ids.unsqueeze(1).expand(-1,t,-1).reshape(b*t,-1) # bt,n

            Qformer_atts = torch.cat([query_atts,text_atts],dim=1) # bt,L
            query_output = self.audio_Qformer.bert(
                text_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=audio_feature,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
        else:
            query_output = self.audio_Qformer.bert(
                attention_mask=query_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=audio_feature,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
        audio_embeds = query_output.last_hidden_state # bt,L,d
        audio_embeds = self.audio_proj(audio_embeds[:,:self.num_query_token])
        audio_embeds = audio_embeds.reshape(b,t*self.num_query_token,-1) # b,t*32,dim
        return audio_embeds


