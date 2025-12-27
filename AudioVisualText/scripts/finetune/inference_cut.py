import os,sys
sys.path.append(os.getcwd())
from os.path import join,exists
import pathlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset
from PIL import Image
import torch
import itertools
try:
    import torch_npu
    import torch_npu_acc
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import transformers

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments,InferenceArguments

from dataset.unified_dataset import get_dataset_collator
from utils.util import set_seed,find_all_linear_names,prepare_sample,write2json,load_ckpt
from utils.deepspeed_utils import *

local_rank = None

from torch.utils.data.distributed import DistributedSampler

class Test_DistributedSampler(DistributedSampler): 
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False): 
        super(Test_DistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        N = len(self.dataset)
        R = self.num_replicas
        base_num_samples = N // R
        remainder = N % R
        if self.rank < remainder:
            self.num_samples = base_num_samples + 1
        else:
            self.num_samples = base_num_samples

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def inference(dataloader,ckpt_dir,model,tokenizer,task):
    save_dir = join(ckpt_dir,f'inference_results')
    os.makedirs(save_dir,exist_ok=True)
    pbar = tqdm(total=len(dataloader),desc=f'inference {task}')
    fp = join(save_dir,f'inference_{task}.jsonl')
    for step, sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data = sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':500,
            }
        )
        with torch.no_grad():
            output = model.generate(**sample)
            output = tokenizer.batch_decode(output,skip_special_tokens=False)
        for i in range(bs):
            metadata = batch_metadata[i]
            metadata['predict'] = output[i]

            write2json(fp=fp,dict_data=metadata)
        
        pbar.update(1)
    pbar.close()


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

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

    model.config.use_cache = True

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
        from peft_hyper import LoraConfig,get_peft_model
        lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
        target_modules = lora_trainable.split(',')
        lora_rank = training_args.lora_r
        lora_alpha = 16
        lora_dropout = 0.05
        lora_nums = int(len(str(training_args.lora_r)))
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
    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,BEATs_ckpt_path=model_args.BEATs_ckpt_path)

    model.initialize_MM_tokenizer(tokenizer)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)


    ckpt_dir = infer_args.ckpt_dir


    ckpt_path = join(ckpt_dir,'non_lora_trainables.bin')
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    print(f'load ckpt from {ckpt_path} finished...')
    
    ckpt_path = join(ckpt_dir,'adapter_model.bin')
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    print(f'load ckpt from {ckpt_path} finished...')


    model.eval()
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    
    image_processor = model.module.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode='test')
    

    batch_size = 8

    sampler = Test_DistributedSampler(dataset,num_replicas=torch.distributed.get_world_size(),rank=local_rank,shuffle=False)

    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,sampler=sampler,collate_fn=collator,drop_last=False,num_workers=4)

    if data_args.avqa_task:
        inference(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model.module,tokenizer=tokenizer,task = 'avqa')
    if data_args.ave_task:
        inference(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model.module,tokenizer=tokenizer,task = 'ave')



if __name__ == "__main__":
    train()

