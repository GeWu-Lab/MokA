import torch
from dataclasses import asdict
from packaging import version
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled


from configs.unified_config import ModelArguments

# def get_qwen_and_tokenizer(model_args,evaluation_mode=True,return_tokenizer_only=False,attn_impl=None,**kwargs):
#     from models.longvideo_qwen import LongVideoForCausalLM
#     from models.qwen2.configuration_qwen2 import Qwen2Config
#     from transformers import Qwen2Tokenizer
    
#     model_args: ModelArgs
#     model_args_dict = asdict(model_args)
#     model_args_dict.update(**kwargs)

#     model_name_or_path = model_args_dict["model_name_or_path"]

#     # dtype & device
#     dtype = model_args_dict["dtype"]
#     if dtype == "bf16":
#         dtype = torch.bfloat16
#     elif dtype == "fp16":
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     tokenizer = Qwen2Tokenizer.from_pretrained(
#         model_name_or_path,  
#         padding_side=model_args_dict["padding_side"], 
#         local_files_only=True,
#         use_fast=True,
#     )

#     tokenizer.bos_token_id = 151643
#     tokenizer.eos_token_id = 151645
#     tokenizer.pad_token_id = 151645

#     # if tokenizer.pad_token_id is None:
#     #     tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     # if tokenizer.bos_token_id is None:
#     #     tokenizer.bos_token_id = 1
#     # print(f'bos: {tokenizer.bos_token_id}')
    
#     if return_tokenizer_only:
#         return tokenizer

#     # beacon
#     beacon_kwargs = {}
#     for k, v in model_args_dict.items():
#         if k.startswith("beacon") and v is not None:
#             beacon_kwargs[k] = v

    
#     if model_args_dict["enable_beacon"]:
#         config = Qwen2Config.from_pretrained(
#             model_name_or_path, 
#             torch_dtype=dtype,
#             trust_remote_code=True,
#             local_files_only=True,
#             **beacon_kwargs,
#         )
#     else:
#         config = Qwen2Config.from_pretrained(
#             model_name_or_path,
#             torch_dtype=dtype,
#             trust_remote_code=True,
#             local_files_only=True,
#         )

#     config._attn_implementation = attn_impl

#     model = LongVideoForCausalLM.from_pretrained(
#         model_name_or_path,
#         config=config, 
#         torch_dtype=dtype,
#         local_files_only=True,
#         # enable_beacon=model_args_dict["enable_beacon"],
#     )
#     model.config.use_cache = False
    
#     # load lora
#     if model_args_dict["lora"] is not None:
#         from peft import PeftModel
#         model = PeftModel.from_pretrained(
#             model, 
#             model_args_dict["lora"],
#             torch_dtype=dtype,
#             # device_map=device,
#         )
#         if model_args_dict["lora_unload"]:
#             model = model.merge_and_unload()

#     if model_args_dict["enable_tp"]:
#         import tensor_parallel as tp
#         # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
#         model = tp.tensor_parallel(model, sharded=True)

#         if model.generation_config.eos_token_id == 128001:
#             model.generation_config.eos_token_id = [128001, 128009]

#     if isinstance(model, transformers.modeling_utils.PreTrainedModel):
#         model = model.eval()
#         if evaluation_mode:
#             # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
#             model.requires_grad_(False)

#     # override the default generation config
#     generation_config = model_args.get_generation_config()
#     if len(generation_config):
#         model.generation_config.update(**generation_config)

#     return model, tokenizer



def get_llama_and_tokenizer(model_args,evaluation_mode=True,return_tokenizer_only=False,attn_impl=None,**kwargs):

    from UnifiedLLM.models.unified_llama import UnifiedLLM
    from transformers import LlamaConfig

    model_args: ModelArguments
    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)

    model_name_or_path = model_args_dict["model_name_or_path"]

    # dtype & device
    # dtype = model_args_dict["dtype"]
    # if dtype == "bf16":
    #     dtype = torch.bfloat16
    # elif dtype == "fp16":
    #     dtype = torch.float16
    # else:
    #     dtype = torch.float32
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,  
        padding_side='left', 
        local_files_only=True,
        use_fase=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer
    
    config = LlamaConfig.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )

    config._attn_implementation = attn_impl

    model = UnifiedLLM.from_pretrained(
        model_name_or_path,
        config=config, 
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.config.use_cache = False

    # load lora
    # if model_args_dict["lora"] is not None:
    #     from peft import PeftModel
    #     model = PeftModel.from_pretrained(
    #         model, 
    #         model_args_dict["lora"],
    #         torch_dtype=dtype,
    #         # device_map=device,
    #     )
    #     if model_args_dict["lora_unload"]:
    #         model = model.merge_and_unload()

    # if model_args_dict["enable_tp"]:
    #     import tensor_parallel as tp
    #     # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
    #     model = tp.tensor_parallel(model, sharded=True)

    #     if model.generation_config.eos_token_id == 128001:
    #         model.generation_config.eos_token_id = [128001, 128009]

    if isinstance(model, transformers.modeling_utils.PreTrainedModel):
        model = model.eval()
        if evaluation_mode:
            # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
            model.requires_grad_(False)

    # override the default generation config
    # generation_config = model_args.get_generation_config()
    # if len(generation_config):
    #     model.generation_config.update(**generation_config)

    return model, tokenizer



