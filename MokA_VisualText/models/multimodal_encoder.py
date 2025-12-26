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

# from models.vision_encoder import VisionEncoder
from models.Qformer import BertConfig,BertLMHeadModel
from models.beats.BEATs import BEATs,BEATsConfig
from models.taming_transformer.vqgan import VQModel
from models.loss import dice_loss,overlap_loss,sigmoid_ce_loss,F10_IoU_BCELoss


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
        

    def forward(self,visual_feature,question=None):
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


'''
Module for postprocess segmentation task.
'''
class SegModule(nn.Module):

    def __init__(
        self,
        d_model = 3584,
        vit_image_embedding_dim = 1024,
        prompt_embed_dim = 256,
        image_scale_nums = 2,
        mask_decoder_transformer_depth = 2,
        token_nums_per_scale = 3,
        avs_query_num = 300,
        num_classes = 1,
        query_generator_num_layers = 2,
        image_size = 224,
        patch_size = 14,
        image_embedding_size = 16,
        dice_loss_weight = 0.5,
        bce_loss_weight = 2.0
    ) -> None:
        super().__init__()

        self.image_scale_nums = image_scale_nums
        self.token_nums_per_scale = token_nums_per_scale
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.patch_size = patch_size
        assert patch_size * image_embedding_size == image_size
        self.num_classes = num_classes

        scalar = 1 / self.token_nums_per_scale
        self.multiseg_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(self.token_nums_per_scale)]
        
        scalar = 1 / self.image_scale_nums
        self.multiscale_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(self.image_scale_nums)]

        # Projection layer: hidden_states -> sparse prompt embedding
        text_fc = [
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, prompt_embed_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

        # dense embedding
        self.no_mask_embed = nn.Embedding(1, prompt_embed_dim)

        # multi scale image embedding
        self.image_feature_neck = nn.Sequential(
            nn.Conv2d(
                vit_image_embedding_dim,
                prompt_embed_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(prompt_embed_dim),
            nn.Conv2d(
                prompt_embed_dim,
                prompt_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(prompt_embed_dim),
        )

        # dense pos embedding
        self.pe_layer = PositionEmbeddingRandom(prompt_embed_dim // 2)

        # multi scale mask decoder
        image_feature_scale_num = image_scale_nums
        self.mask_decoder=MaskDecoderMultiScale(
            transformer=TwoWayTransformer(
                depth=mask_decoder_transformer_depth,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            image_feature_scale_num=image_feature_scale_num,
            avs_query_num=avs_query_num,
            num_classes=num_classes,
            query_generator_num_layers=query_generator_num_layers,
        )
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer((self.image_embedding_size,self.image_embedding_size)).unsqueeze(0)
    

    def forward(
        self,
        pred_embeddings,  # b,n,dim
        multi_scale_image_feature_list, # [(b,n,dim), (b,n,dim), ... , (b,n,dim)]
        low_res_mask_size = 112,
        gt_mask = None,
        batch_task_names = [], # ['s4','ms3','avss',...]
    ):
        # assert pred_embeddings.shape[0] == multi_scale_image_feature_list[0].shape[0]
        # assert len(batch_task_names) == pred_embeddings.shape[0]
        seg_token_num = self.token_nums_per_scale
        feat_scale_num = self.image_scale_nums
        
        ## pred embedding
        hidden_states = []
        hidden_states.append(self.text_hidden_fcs[0](pred_embeddings))
        pred_embeddings = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        bs,n,dim = pred_embeddings.shape
        object_nums = n // (self.image_scale_nums * self.token_nums_per_scale) 
        pred_embeddings = pred_embeddings.reshape(bs,object_nums,self.image_scale_nums,self.token_nums_per_scale,dim)

        fused_pred_embeddings = torch.zeros([bs,object_nums,feat_scale_num,dim]).to(pred_embeddings)
        for i in range(seg_token_num):
            # bs,obj_nums,scale,dim
            fused_pred_embeddings = fused_pred_embeddings + self.multiseg_scalar[i] * pred_embeddings[:, :, :, i]  
        
        # image embedding
        token_num = self.image_embedding_size * self.image_embedding_size
        multi_scale_grid_image_feature = []
        for image_feature in multi_scale_image_feature_list:
            bs,n,dim = image_feature.shape
            img_nums = n // token_num
            image_feature = image_feature.reshape(bs,img_nums,self.image_embedding_size,self.image_embedding_size,dim)
            image_feature = rearrange(image_feature,'bs nums size1 size2 dim -> bs nums dim size1 size2')
            image_feature = image_feature.contiguous()
            image_feature = image_feature[:,0]  # bs,dim,size,size
            multi_scale_grid_image_feature.append(image_feature)
        multi_scale_grid_image_feature = torch.stack(multi_scale_grid_image_feature,dim=1) # bs,level,dim,size,size

        pred_masks = []
        for i in range(bs):
            sparse_embeddings = fused_pred_embeddings[i] # 1,scale,dim
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                sparse_embeddings.shape[0], -1, self.image_embedding_size, self.image_embedding_size
            )  # 1,256 -> 1,256,1,1 -> 1,256,16,16
            # print('sparse_embeddings: ',sparse_embeddings.shape) # torch.Size([1, 2, 256])
            # print('dense_embeddings: ',dense_embeddings.shape) # torch.Size([1, 256, 16, 16])
            _img_embeddings = self.image_feature_neck(multi_scale_grid_image_feature[i]) #[Lev, 256, 16, 16]
            # print('_img_embeddings: ',_img_embeddings.shape) # torch.Size([2, 256, 16, 16])
            out_size = low_res_mask_size
            num_classes = 71 if batch_task_names[i] == 'avss' else 1
            low_res_masks = torch.zeros([sparse_embeddings.shape[0], num_classes, out_size, out_size]).to(_img_embeddings)
            if self.image_scale_nums > 1:
                for l in range(self.image_scale_nums):
                    l_low_res_masks = self.mask_decoder(
                        image_embeddings=_img_embeddings[l].unsqueeze(0), # 1, 256, 16, 16
                        image_pe=self.get_dense_pe().to(_img_embeddings[l]), 
                        sparse_prompt_embeddings=sparse_embeddings[:, l].unsqueeze(1), # 1,1,dim
                        dense_prompt_embeddings=dense_embeddings, # 1, 256, 16, 16
                        previous_masks = l_low_res_masks if l>0 else None, 
                        level_num = l,
                        task_name = batch_task_names[i],
                        is_last = (l == self.image_scale_nums - 1)
                    )
                    # torch.Size([1, 1, 32, 32])
                    # torch.Size([1, 1, 64, 64])
                    # print('l_low_res_masks: ',l_low_res_masks.shape)
                    low_res_masks = low_res_masks + self.multiscale_scalar[l] * F.interpolate(l_low_res_masks.float(), (out_size, out_size),mode="bilinear",align_corners=False,).to(l_low_res_masks)
        
            pred_mask = self.postprocess_masks(
                low_res_masks,
                input_size=(self.image_size,self.image_size),
                original_size=None,
            ) # object_nums, num_classes, h, w   (1, num_classes, 224, 224)
            pred_masks.append(pred_mask[0])  # [num_classes, 224, 224]

        if gt_mask is None:
            return {
                'pred_masks':pred_masks,
                # 'mask_scores':mask_scores
            }
        
        # model_output = output
        gt_masks = gt_mask
        ms3_s4_mask_bce_loss = 0
        ms3_s4_mask_dice_loss = 0
        avss_ce_loss = 0
    
        num_masks = 0
        ms3_s4_sample_nums = 0
        avss_sample_nums = 0
        for batch_idx in range(len(pred_masks)):
            task_name = batch_task_names[batch_idx]
            gt_mask = gt_masks[batch_idx]  # 1,224,224
            pred_mask = pred_masks[batch_idx] # num_classes,224,224
            '''for ms3 and s4 task'''
            if task_name == 'ms3' or task_name == 's4' or task_name == 'ref-avs':
                ms3_s4_mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                ms3_s4_mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                ms3_s4_sample_nums += 1
            elif task_name == 'avss':
                gt_mask = gt_mask.to(torch.long)
                avss_ce_loss += (
                    F10_IoU_BCELoss(pred_mask=pred_mask.unsqueeze(0),ten_gt_masks=gt_mask,gt_temporal_mask_flag=None)
                )
                avss_sample_nums += 1
            num_masks += gt_mask.shape[0]
        
        bce_loss_weight = 1.0
        dice_loss_weight = 0.5
        if ms3_s4_sample_nums > 0:
            ms3_s4_mask_bce_loss = bce_loss_weight * ms3_s4_mask_bce_loss * (ms3_s4_sample_nums / (ms3_s4_sample_nums + avss_sample_nums))
            ms3_s4_mask_dice_loss = dice_loss_weight * ms3_s4_mask_dice_loss * (ms3_s4_sample_nums / (ms3_s4_sample_nums + avss_sample_nums))
        if avss_sample_nums > 0:
            avss_ce_loss = bce_loss_weight * avss_ce_loss * (avss_sample_nums / (ms3_s4_sample_nums + avss_sample_nums))
        
        mask_loss = ms3_s4_mask_bce_loss + ms3_s4_mask_dice_loss + avss_ce_loss
        
        return {
            'mask_loss':mask_loss
        }
    


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        
        target_size = max(input_size)
        dtype = masks.dtype
        # if self.vision_tower_for_mask:
        masks = F.interpolate(
            masks.float(),
            (target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # if not self.masks_process_with_clip:
        #     assert input_size[0] <= target_size
        #     assert input_size[1] <= target_size
        #     masks = masks[..., : input_size[0], : input_size[1]]
        #     masks = F.interpolate(
        #         masks, original_size, mode="bilinear", align_corners=False
        #     )
        
        masks = masks.to(dtype)
        # 
        return masks    


class MaskEncoder(nn.Module):
    def __init__(self, token_shift=32000) -> None:
        super().__init__()

        ddconfig={
            'double_z':False,
            'z_channels':256,
            'resolution':256,
            'in_channels':3,
            'out_ch':3,
            'ch':128,
            'ch_mult':(1,1,2,2,4),
            'num_res_blocks':2,
            'attn_resolutions':(16,),
            'dropout':0.0
        }
        self.vqgan=VQModel(
            ddconfig=ddconfig,
            lossconfig=None,
            n_embed=16384,
            embed_dim=256,
            # ckpt_path='/group/40061/cserdu/pretrain/vqgan/vqgan_imagenet_f16_16384/last.ckpt'
            ckpt_path = '/group/40061/cserdu/pretrain/vqgan/vqgan_imagenet_f16_16384/weight.ckpt'
        )
        self.vqgan.requires_grad_(False)
        self.vqgan.eval()
        print('init vqgan finished...')
        self.token_shift = token_shift

    @torch.no_grad()
    def encode_mask(self,mask):
        # mask: b,c,h,w
        indices = self.vqgan.get_codebook_indices(mask)
        indices = indices + self.token_shift    
        return indices


    @torch.no_grad()
    def decode_mask(self,indices):
        # indices: b,n
        indices = indices - self.token_shift
        indices = indices.to(torch.long)
        tokens = torch.clip(indices,0,16384-1)
        image = self.vqgan.decode_code(tokens)
        return image # b,c,h,w
    
        # image = (image + 1) * 127.5
        # image = torch.clip(image,0,255)
        # image=image.permute(0,2,3,1).contiguous() # b,h,w,c
        # return image
    

    def forward(self,mask):
        # mask: b,c,h,w
        indices = self.encode_mask(mask)
        return indices  # b,n
    

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif text_embeds is not None:
            return text_embeds.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor], # N,level,256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, text_embeds)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if text_embeds is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )  # 1,256 -> 1,256,1,1 -> b,256,16,16

        return sparse_embeddings, dense_embeddings



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1

        if coords.dtype != self.positional_encoding_gaussian_matrix.dtype:
            coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)

        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones(
            (h, w), device=device, dtype=self.positional_encoding_gaussian_matrix.dtype
        )
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MLP_conv(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class MaskDecoderMultiScale(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        image_feature_scale_num: int = 1,
        avs_query_num = 300,
        num_classes = 1,
        query_generator_num_layers = 2,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.avs_query_num = avs_query_num
        self.num_classes = num_classes
        self.transformer = nn.ModuleList([deepcopy(transformer) for _ in range(image_feature_scale_num)])

        self.avs_query_tokens = nn.Embedding(avs_query_num,transformer_dim) # 300,256
        self.query_generator = QueryGenerator(num_layers=query_generator_num_layers,
                                              embed_dim=transformer_dim,num_heads=8,
                                              hidden_dim=2048)

        self.hyper_mlp_out = MLP_conv(input_dim=avs_query_num,hidden_dim=transformer_dim,
                             output_dim=transformer_dim//8,num_layers=3)
        
        self.hyper_mlp = MLP(input_dim=transformer_dim,hidden_dim=transformer_dim,
                             output_dim=transformer_dim//8,num_layers=3)

        # if last_feature:
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 8, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 8),
            activation(),
        )

        self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        transformer_dim, transformer_dim, kernel_size=2, stride=2),
                        LayerNorm2d(transformer_dim),
                        activation(),)


        self.pe1=PositionEmbeddingRandom(transformer_dim//2)
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )

        self.image_feature_scale_num = image_feature_scale_num
        self.level_embed = nn.Embedding(image_feature_scale_num, transformer_dim)  # 2,256
        self.ms3_s4_classfier=nn.Conv2d(transformer_dim//8,1,kernel_size=1,stride=1, padding=0, bias=False)
        self.avss_classifier = nn.Conv2d(transformer_dim//8,71,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        level_num: int,
        previous_masks=None,
        task_name = '',
        is_last = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            level_num=level_num,
            previous_masks=previous_masks,
            task_name = task_name,
            is_last = is_last,
        )

        return masks

    # def predict_masks(
    #     self,
    #     image_embeddings: torch.Tensor, # 1,256,32,32
    #     image_pe: torch.Tensor, # 
    #     sparse_prompt_embeddings: torch.Tensor, # 1,N,256
    #     dense_prompt_embeddings: torch.Tensor, # N,256,16,16
    #     level_num: int, # 0 1
    #     previous_masks=None
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Predicts masks. See 'forward' for more details."""
    #     # Concatenate output tokens
    #     # import pdb;pdb.set_trace()
    #     output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  # 5,256
    #     output_tokens = output_tokens.unsqueeze(0).expand(
    #         sparse_prompt_embeddings.size(0), -1, -1
    #     ) # 1,5,256

    #     tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # 1,5+N,256
    #     level = torch.tensor([level_num, ], dtype=torch.long, device=tokens.device).expand((tokens.size(0), 1)) # 1,1
    #     level_embed = self.level_embed(level) # 1,1,256
    #     tokens = tokens + level_embed

    #     # image_embeddings: [1, C, H, W], tokens: [B, N, C]
    #     # dense_prompt_embeddings: [B, C, H, W]
    #     # Expand per-image data in batch direction to be per-mask
    #     src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    #     if level_num > 0:
    #         src = self.upsample_2x(src)
    #         b, c, h, w = src.shape
    #         previous_masks = torch.mean(previous_masks, dim=1)
    #         src = (torch.repeat_interleave(previous_masks[:, None], 256, dim=1).sigmoid() + 1) * src
    #         image_pe=self.pe1((h, w)).unsqueeze(0)
    #         dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings.float(), size=(h, w), mode="bilinear", align_corners=False).to(dense_prompt_embeddings)
        
    #     src = src + dense_prompt_embeddings # B, C, 32, 32
    #     pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    #     b, c, h, w = src.shape

    #     # Run the transformer
    #     hs, src = self.transformer[level_num](src, pos_src, tokens)
    #     iou_token_out = hs[:, 0, :]
    #     mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

    #     # Upscale mask embeddings and predict masks using the mask tokens
    #     src = src.transpose(1, 2).view(b, c, h, w)
    #     # print('src: ',src.shape) #torch.Size([1, 256, 16, 16])
    #     # if level_num == self.image_feature_scale_num-1:
    #     upscaled_embedding = self.output_upscaling(src)
    #     # print('upscaled_embedding: ',upscaled_embedding.shape) # torch.Size([1, 32, 32, 32])
    #     hyper_in_list: List[torch.Tensor] = []
    #     for i in range(self.num_mask_tokens):
    #         hyper_in_list.append(
    #             self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
    #         )
    #     hyper_in = torch.stack(hyper_in_list, dim=1)
    #     b, c, h, w = upscaled_embedding.shape
    #     masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
    #         b, self.num_mask_tokens, h, w
    #     )

    #     # else:
    #     #     hyper_in_list: List[torch.Tensor] = []
    #     #     for i in range(self.num_mask_tokens):
    #     #         hyper_in_list.append(
    #     #             self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
    #     #         )
    #     #     hyper_in = torch.stack(hyper_in_list, dim=1) #B, num_mask_tokens, C
    #     #     upscaled_embedding = self.output_upscaling(src)
    #     #     b, c, h, w = upscaled_embedding.shape
    #     #     masks = (hyper_in @ upscaled_embedding.view(b, 32, h * w)).view(
    #     #         b, self.num_mask_tokens, h, w
    #     #     )
    #     # masks = torch.sigmoid(torch.mean(masks, dim=1, keepdim=True)
    #     # Generate mask quality predictions
    #     iou_pred = self.iou_prediction_head(iou_token_out)

    #     return masks, iou_pred


    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # 1,256,32,32
        image_pe: torch.Tensor, # 
        sparse_prompt_embeddings: torch.Tensor, # 1,N,256
        dense_prompt_embeddings: torch.Tensor, # N,256,16,16
        level_num: int, # 0 1 2
        previous_masks=None,
        task_name = '',
        is_last = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        avs_query_tokens = self.avs_query_tokens.weight.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        ) # 1,300,256
        ### use query generator
        tokens = self.query_generator(avs_query_tokens,sparse_prompt_embeddings) # b,300,256
        # tokens = torch.cat((avs_query_tokens, sparse_prompt_embeddings), dim=1)  # 1,300+N,256
        
        level = torch.tensor([level_num, ], dtype=torch.long, device=tokens.device).expand((tokens.size(0), 1)) # 1,1
        level_embed = self.level_embed(level) # 1,1,256
        tokens = tokens + level_embed

        # image_embeddings: [1, C, H, W], tokens: [B, N, C]
        # dense_prompt_embeddings: [B, C, H, W]
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if level_num > 0:
            src = self.upsample_2x(src)
            b, c, h, w = src.shape
            previous_masks = torch.mean(previous_masks, dim=1)
            src = (torch.repeat_interleave(previous_masks[:, None], 256, dim=1).sigmoid() + 1) * src
            image_pe=self.pe1((h, w)).unsqueeze(0)
            dense_prompt_embeddings = F.interpolate(dense_prompt_embeddings.float(), size=(h, w), mode="bilinear", align_corners=False).to(dense_prompt_embeddings)
        
        src = src + dense_prompt_embeddings # B, C, 32, 32
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer[level_num](src, pos_src, tokens)

        query_tokens_out = hs[:,:self.avs_query_num] # b,q,d
        query_tokens_out = self.hyper_mlp(query_tokens_out)  # b,q,d//8
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        # src = torch.einsum('bqd,bdhw->bqhw',query_tokens_out,src) # b,q,h,w
        upscaled_embedding = self.output_upscaling(src)  # b,d//8,2h,2w
        # hyper_output = self.hyper_mlp(src) # b,q,h,w -> b,d,h,w
        # hyper_output = self.output_upscaling(hyper_output)  # b,d/8,h,w
        b, c, h, w = upscaled_embedding.shape
        masks = (query_tokens_out @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        ) # b,q,h,wss
        masks = self.hyper_mlp_out(masks) # b,d//8,h,w
        if task_name == 'avss':
            pred_masks = self.avss_classifier(masks)
        else:
            pred_masks = self.ms3_s4_classfier(masks) # b,N,h,w

        return pred_masks


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.to(image_embedding)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



'''
query generator
'''
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, audio_feat):
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query+out1)
        out2 = self.cross_attn(query, audio_feat, audio_feat)[0]
        query = self.norm2(query+out2)
        out3 = self.ffn(query)
        query = self.norm3(query+out3)
        return query


class QueryGenerator(nn.Module):
    def __init__(self, num_layers, embed_dim=256, num_heads=8, hidden_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        # self.query_num = query_num
        self.embed_dim = embed_dim
        # self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [AttentionLayer(embed_dim, num_heads, hidden_dim)
             for i in range(num_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, avs_query, sparse_embedding):
        for layer in self.layers:
            query = layer(avs_query, sparse_embedding)
        return query


