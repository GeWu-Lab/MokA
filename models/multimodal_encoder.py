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
        model_name_or_path = 'pre-trained/av_unified/clip-vit-large-patch14',
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
        bert_ckpt_path = 'pre-trained/av_unified/google-bert-base-uncased', 
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
        ckpt_path = 'pre-trained/av_unified/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
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
        bert_ckpt_path = 'pre-trained/av_unified/google-bert-base-uncased', 
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


