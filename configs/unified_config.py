
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class ModelArguments:  
    # llm
    model_name_or_path: Optional[str] = field(default="video-llama2/Mistral-7B-Instruct-v0.2")
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    llm_name: str = field(default='qwen')
    ## visual module
    vit_ckpt_path: str = field(default='pretrain/openai-clip-vit-large-patch14-224')
    select_layer_list = [14,23]
    select_feature: str = field(default='patch')
    image_size: int = field(default=224)
    patch_size: int = field(default=14)
    visual_query_token_nums: int = field(default=32)
    ## audio module
    BEATs_ckpt_path: str = field(default='pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    audio_query_token_nums: int = field(default=32)
    ## seg module
    prompt_embed_dim: int = field(default=256)
    mask_decoder_transformer_depth: int = field(default=2)
    low_res_mask_size: int = field(default=112)
    image_scale_nums: int = field(default=2)
    token_nums_per_scale: int = field(default=3)
    avs_query_num: int = field(default=300)
    num_classes: int = field(default=1)
    query_generator_num_layers: int = field(default=2)


@dataclass
class InferenceArguments:
    # used for inference
    ckpt_dir: str = field(default='')
    
    # for infer avs
    adapter_ckpt_path: str = field(default=None)
    test_name: str = field(default='test') # for ref-avs: test_u,test_s,test_n
    cut_folds:  int = field(default=2)

    device: str = field(default='cuda:0')
    

@dataclass
class DataArguments:
    # pretrain
    video_frame_nums: int = field(default=8)
    image_size = ModelArguments.image_size
    image_caption_task: bool = field(default=False)
    video_caption_task: bool = field(default=False)
    audio_caption_task: bool = field(default=False)
    segmentation_task: bool = field(default=False)
    # fine-tune
    avqa_task: bool = field(default=False)
    ave_task: bool = field(default=False)
    avvp_task: bool = field(default=False)
    arig_task: bool = field(default=False)
    ms3_task: bool = field(default=False)
    s4_task : bool = field(default=False)
    avss_task: bool = field(default=False)
    avcap_task: bool = field(default=False)
    ref_avs_task: bool = field(default=False)
    multi_frames: bool = field(default=False) # avs task input single frame



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=32,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 44
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    ce_loss_weight: float = field(default=1.0)
    dice_loss_weight: float = field(default=0.5)
    bce_loss_weight: float = field(default=2.0)
    ## my
    reserved_modality: str = field(default=None)
    loramethod: str = field(default=None)

    audio_branch: bool = field(default=False)
    visual_branch: bool = field(default=False)
    seg_branch: bool = field(default=False)

    save_modules: str = field(default='vl_projector,al_projector,lora')

    exp_desc: str = field(default='exp')
