import os,sys
sys.path.append(os.getcwd())
from os.path import join
import pathlib
import json
import torch
from dataclasses import asdict
import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments


from dataset.unified_dataset import get_dataset_collator

from trainer import UnifiedTrainer

from utils.util import set_seed,rank0_print,find_all_linear_names,rank0write2txt
from utils.deepspeed_utils import *

local_rank = None

def train(attn_implementation=None):
    # print('lalala')
    # return
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    saved_config = {
        'model_args':asdict(model_args),
        'data_args':asdict(data_args),
        'training_args':asdict(training_args)
    }
    os.makedirs(output_dir,exist_ok=True)
    with open(join(output_dir,'saved_config.json'),'w') as f:
        f.write(json.dumps(saved_config,indent=4))

    if model_args.llm_name == 'llama':
        d_model = 4096

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        ### hyper lora
        from peft_hyper import LoraConfig,get_peft_model
        lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
        target_modules = lora_trainable.split(',')
        lora_rank = training_args.lora_r
        lora_alpha = 16
        lora_dropout = 0.05
        lora_nums = int(len(str(training_args.lora_r)))
        print('lora_nums: ',lora_nums)
        modules_to_save = None
        peft_config = LoraConfig(
            task_type = "CAUSAL_LM",
            target_modules = target_modules,
            inference_mode = False,
            r = lora_rank, 
            loramethod= training_args.loramethod,
            reserved_modality=training_args.reserved_modality,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            lora_nums = lora_nums,
            blc_alpha= training_args.blc_alpha,
            blc_weight=training_args.blc_weight,
        )
        model = get_peft_model(model, peft_config)


    
    if model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id

    image_scale_nums = model_args.image_scale_nums
    token_nums_per_scale = model_args.token_nums_per_scale

    model.initialize_MM_tokenizer(tokenizer)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums,' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)

    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,BEATs_ckpt_path=model_args.BEATs_ckpt_path)


   
    audio_ckpt_dir = 'pre-trained/av_unified/audio-pretrain'
    visual_ckpt_dir = 'pre-trained/av_unified/visual-pretrain'


    ckpt = torch.load(join(audio_ckpt_dir,'audio_pretrain.bin'),map_location='cpu')
    weight = ckpt.pop('model.embed_tokens.weight',None)
    model.model.load_state_dict(ckpt,strict=False)
    if weight is not None:
        print(f'pop embed weight, shape: {weight.shape} load ckpt from path: {audio_ckpt_dir}')
    
    ckpt = torch.load(join(visual_ckpt_dir,'visual_pretrain.bin'),map_location='cpu')
    weight = ckpt.pop('model.embed_tokens.weight',None)
    model.model.load_state_dict(ckpt,strict=False)
    if weight is not None:
        print(f'pop embed weight, shape: {weight.shape}  load ckpt from path: {visual_ckpt_dir}')


    save_modules = training_args.save_modules
    print(f'save_modules: {save_modules}')
    save_modules = save_modules.split(',')
    for name, param in model.named_parameters():
        require_grad = False
        for target in save_modules:
            if target in name:
                require_grad = True
                break
        param.requires_grad_(require_grad)


    if local_rank == 0:
        with open(join(output_dir,'model_trainable_params.txt'),'w') as f:
            f.write('\n')
        params = []
        for name,param in model.named_parameters():
            if param.requires_grad==True:
                with open(join(output_dir,'model_trainable_params.txt'),'a') as f:
                    f.write(name + '  ' + str(param.shape))
                    f.write('\n')
                params.append(param.numel())
        trainable_params = sum(params) /1e6
        with open(join(output_dir,'model_trainable_params.txt'),'a') as f:
            f.write(f'trainable_params: {trainable_params:.3f}MB')
        print(f'trainable_params: {trainable_params:.3f}MB')

        with open(join(output_dir,'model.txt'),'w') as f:
            f.write(str(model))
    
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor)
    trainer = UnifiedTrainer(model=model, tokenizer=tokenizer, args=training_args,
                             train_dataset=dataset, data_collator=collator)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))


if __name__ == "__main__":
    train()



