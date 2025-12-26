# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import torch.nn.functional as F
import os
import os
import socket
import subprocess
from datetime import timedelta
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from transformers import (HfArgumentParser, TrainingArguments,
                          set_seed)


import sys
import logging
import argparse
from dataclasses import dataclass, field
import ast
import torch
from transformers import Trainer
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pickle
from pycocotools import mask as maskUtils
import warnings
from tqdm import tqdm
import deepspeed
import cv2
from dist_utils import init_dist
import json
from safetensors.torch import load_file

from modified_peft import LoraConfig
from modified_peft import PeftMixedModel

from transformers import CLIPImageProcessor
from modified_models.modelling_llava import LlavaForConditionalGeneration
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from modified_models.configuration_llava import LlavaConfig
import transformers


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'



import jsonlines
def write2json(fp,dict_data,mode='a'):
    with jsonlines.open(fp,mode=mode) as f:
        f.write(dict_data)

class TrainDataset(Dataset):
    def __init__(self, config,image_processor,tokenizer):
        self.config = config
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.samples=[]

        self.init_mme_dataset()
    
    def __len__(self):
        return len(self.samples)


    def init_mme_dataset(self):

        print('evaluation on MME benchmark')
        import os
        from os.path import join,exists
        data_root = 'MME_Benchmark_release_version'
        subtasks = os.listdir(data_root)
        for subtask in subtasks:
            subdir = join(data_root, subtask)
            files = os.listdir(subdir)
            # qa_files = [item for item in files if '.txt' in item]
            image_files = [item for item in files if '.txt' not in item]
            for file in image_files:
                qa_path = join(subdir, file[:-4] + '.txt')
                image_path = join(subdir, file)
                with open(qa_path,'r') as f:
                    qas = f.readlines()
                for qa in qas:
                    qa = qa.strip()
                    question = qa.split('Please answer yes or no.')[0]
                    answer = qa.split("Please answer yes or no.")[1]
                    question = question.strip()
                    answer = answer.strip()
                    self.samples.append({
                        'conversation':[
                            {'from':'human','value':'This is an image:\n<image_start><image><image_end>\nPlease answer this question with yes or no, and no other content. Question: '+question},
                        ],
                        'image_path':image_path,
                        'answer':answer,
                        'question':question,
                        'subtask':subtask,
                    })
    


    def tokenizer_target(self,input_ids, return_tensors=None):

        # 获取 [/INST] 和 </s> 的 token IDs
        # [/INST] 可能是单个token或多个token，需要编码获取
        inst_end_ids = self.tokenizer.encode('[/INST]', add_special_tokens=False)
        eos_ids = self.tokenizer.encode('</s>', add_special_tokens=False)
        
        # print('inst_end_ids ([/INST]):', inst_end_ids)
        # print('eos_ids (</s>):', eos_ids)
        
        # 辅助函数：检查从指定位置开始是否匹配给定的token序列
        def matches_token_sequence(input_list, start_pos, token_sequence):
            if start_pos + len(token_sequence) > len(input_list):
                return False
            return input_list[start_pos:start_pos + len(token_sequence)] == token_sequence
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 初始化target_ids，全部设为-100
        target_ids = torch.full_like(input_ids.squeeze(), -100, dtype=input_ids.dtype, device=input_ids.device)
        input_ids_squeezed = input_ids.squeeze()
        input_ids_list = input_ids_squeezed.tolist()
        
        # 寻找 [/INST] ... </s> 的模式
        i = 0
        while i < len(input_ids_list):
            # 检查是否匹配 [/INST]
            if matches_token_sequence(input_ids_list, i, inst_end_ids):
                # 找到 [/INST]，从其后面开始
                start_idx = i + len(inst_end_ids)
                
                # 寻找对应的 </s>
                end_idx = -1
                for j in range(start_idx, len(input_ids_list) - len(eos_ids) + 1):
                    if matches_token_sequence(input_ids_list, j, eos_ids):
                        end_idx = j
                        break
                
                if end_idx != -1:
                    # 将从 start_idx 到 end_idx + len(eos_ids) (包括</s>)的target_ids设为对应的input_ids
                    target_ids[start_idx:end_idx + len(eos_ids)] = input_ids_squeezed[start_idx:end_idx + len(eos_ids)]
                    
                    # 跳过已处理的部分
                    i = end_idx + len(eos_ids)
                else:
                    # 没找到对应的 </s>，跳过这个 [/INST]
                    i += 1
            else:
                i += 1
        
        return target_ids


    
    def __getitem__(self, idx):

        image_path = self.samples[idx]['image_path']

        image = Image.open(image_path).convert('RGB')





        conversation=self.samples[idx]['conversation']

        image = Image.open(image_path).convert('RGB')
        image = image.resize((224,224))
        image = self.image_processor.preprocess([image],return_tensors='pt')
        
        # 检查pixel_values是否包含无效值
        pixel_values = image['pixel_values']
        if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
            raise ValueError(f"Invalid pixel values detected in image {image_path}")

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]
        
        # 将 conversation 转换为 messages 格式
        # 对于 Llama-2 的 chat_template，content 必须是字符串
        for turn in conversation:
            if turn['from'] == 'human':
                # 将 <image> 替换为特殊的图像占位符 token
                content = turn['value'].replace('<image>', '<image_pad>')
                messages.append({
                    "role": "user",
                    "content": content,
                })
            elif turn['from'] == 'gpt':
                messages.append({
                    "role": "assistant",
                    "content": turn['value'],
                })
        
        instruction = self.tokenizer.apply_chat_template(conversation=messages,tokenize=False,add_generation_prompt=True)

        # print(instruction)

        ## 把所有的<image_pad>替换为<image>
        instruction = instruction.replace('<image_pad>', '<image>')


        # print(instruction)
        # print('--------------------------------')


        ## 把<image>替换为256个<image_pad>
        inputs = instruction.replace('<image>', '<image>' * 32)

        # print(instruction)
        # print('--------------------------------')


        ## get input_ids
        input_ids = self.tokenizer.encode(inputs, add_special_tokens=False,return_tensors='pt')


        # print(input_ids)
        # print('--------------------------------')


        ### replace 0 token with <image_pad>
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<image>")


        ### get my_image_mask
        my_image_mask = input_ids == image_pad_id
        my_text_mask= input_ids != image_pad_id

        input_ids = torch.where(input_ids == image_pad_id, 0, input_ids)

        # print(input_ids)
        # print('--------------------------------')




        target_ids = self.tokenizer_target(input_ids, return_tensors="pt").unsqueeze(0)

        ## question_mask: not image token & target_ids is -100 & after image token
        # 找到最后一个image_token的位置
        image_positions = torch.where(my_image_mask)[1]  # 获取所有image token的列索引
        if len(image_positions) > 0:
            # 找到最后一个image_token的位置（因为image token是连续的一组）
            last_image_token_pos = image_positions[-1]
            # 创建一个mask，表示位置在最后一个image_token之后
            seq_len = my_image_mask.size(1)
            position_mask = torch.arange(seq_len, device=my_image_mask.device) > last_image_token_pos
            position_mask = position_mask.unsqueeze(0)  # 添加batch维度 [1, seq_len]
        else:
            # 如果没有image token，则所有位置都为False
            position_mask = torch.zeros_like(my_image_mask, dtype=torch.bool)
        
        question_mask = ~(my_image_mask) & (target_ids == -100) & position_mask

        # print(target_ids)

        # pixel_values已经在上面定义了
        # print(pixel_values.shape)

        ## bool attention_mask, all True
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        inputs_kwargs={}
        inputs_kwargs['input_ids']=input_ids.squeeze(0)
        inputs_kwargs['labels']=target_ids.squeeze(0)
        inputs_kwargs['attention_mask']=attention_mask.squeeze(0)
        inputs_kwargs['position_ids']=position_ids.squeeze(0)
        inputs_kwargs['pixel_values']=pixel_values
        inputs_kwargs['my_image_mask']=my_image_mask.squeeze(0)
        inputs_kwargs['my_text_mask']=my_text_mask.squeeze(0)
        inputs_kwargs['answer']=self.samples[idx]['answer']
        inputs_kwargs['image_path']=image_path
        inputs_kwargs['question']=self.samples[idx]['question']
        inputs_kwargs['subtask']=self.samples[idx]['subtask']
        inputs_kwargs['question_mask']=question_mask.squeeze(0)



        return inputs_kwargs



@dataclass
class DataCollatorForTrainDataset(object):

    def __call__(self, features):


        batch = {}
        batch_size = len(features)

        max_len = max(len(feature['input_ids']) for feature in features)


        ## self.tokenizer.pad_token_id
        input_ids_pad=2
        target_ids_pad=-100
        attention_mask_pad=False
        position_ids_pad=1
        mask_pad=False

        for key in features[0].keys():
            if key == 'input_ids':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=input_ids_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'my_image_mask':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=mask_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'my_text_mask':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=mask_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'question_mask':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=mask_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'labels':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=target_ids_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'attention_mask':
                for i in range(batch_size):
                    padding_length = max_len - len(features[i][key])
                    features[i][key] = F.pad(features[i][key], (0, padding_length), value=attention_mask_pad).unsqueeze(0)
                batch[key] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            elif key == 'position_ids':
                # 创建一个最长的position_ids序列，所有样本都使用这个
                max_position_ids = torch.arange(0, max_len, dtype=torch.long)
                # 所有样本都使用相同的max_position_ids
                batch[key] = max_position_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
            elif key == 'pixel_values':
                batch['pixel_values'] = torch.cat([features[i][key] for i in range(batch_size)], dim=0)
            else:
                ## catch as list
                batch[key] = [features[i][key] for i in range(batch_size)]
        return batch


def init_tokenizer(tokenizer):
    added_tokens = []
    image_tokens = ['<image>','<image_start>','<image_end>']
    added_tokens += image_tokens
    video_tokens = ['<video>','<video_start>','<video_end>']
    added_tokens += video_tokens
    audio_tokens = ['<audio>','<audio_start>','<audio_end>']
    added_tokens += audio_tokens
    mask_tokens = ['<mask_start>','<mask_end>']
    added_tokens += mask_tokens
    num_new_tokens = tokenizer.add_tokens(added_tokens,special_tokens=True)

    mask_token_nums=6
    seg_tokens = [f'<mask_{i}>' for i in range(mask_token_nums)]
    num_new_tokens += tokenizer.add_tokens(seg_tokens,special_tokens=False)
    added_tokens += seg_tokens

    print(f'Added {num_new_tokens} tokens to LLaMA tokenizer: {added_tokens}')
    ## print id of all added tokens
    for token in added_tokens:
        print(f'Token ID: {tokenizer.convert_tokens_to_ids(token)}')
    


    return tokenizer


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_path: str = field(
        default=None,
        metadata={'help': 'Path to the model checkpoint'}
    )
    attn_weight: float = field(default=0.05, metadata={"help": "Attn weight for LoRA"})
    name_tag: str = field(default='', metadata={"help": "Name tag for the run"})


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    model_args, training_args = parser.parse_args_into_dataclasses()


    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        print("Logging is enabled")
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()


    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')


    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    print('########################################################')
    print('########################################################')




    llama2_path='Llama-2-7b-chat-hf'
    vit_path='clip-vit-large-patch14'

    # 加载配置
    
    print('Loading LLaMA config...')
    llm_config = AutoConfig.from_pretrained(llama2_path, trust_remote_code=True)
    
    print('Loading ViT config...')
    vision_config = AutoConfig.from_pretrained(vit_path, trust_remote_code=True)

    image_processor = CLIPImageProcessor.from_pretrained(vit_path,local_files_only=True)


    # 从llama2加载tokenizer
    print('Loading tokenizer from LLaMA...')
    tokenizer = AutoTokenizer.from_pretrained('Llama-2-7b-chat-hf', trust_remote_code=True)

    print('tokenizer.vocab_size',tokenizer.vocab_size)
    print('llm_config.vocab_size',llm_config.vocab_size)
    print('len (tokenizer)',len(tokenizer))
    
    # 为 Llama-2 tokenizer 设置 chat_template
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


    print('former tokenizer length',len(tokenizer))
    
    tokenizer=init_tokenizer(tokenizer)

    print('after tokenizer length',len(tokenizer))
    print('tokenizer.vocab_size',tokenizer.vocab_size)
    print('llm_config.vocab_size',llm_config.vocab_size)



    image_size=224
    patch_size=14

    image_token_nums=(image_size//patch_size) * (image_size//patch_size)

    projector_config={
        'hidden_size':1024,
        'image_token_nums':image_token_nums,
        'd_model':4096,
        'depth':2,
        'image_token_nums':256,
        'num_query_token':32,
        'num_hidden_layers':2,
        'bert_ckpt_path':'bert-base-uncased'
    }

    # 创建 Llava 配置
    print('Creating LlavaConfig...')
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=llm_config,
        image_token_id=0,  # 根据你的tokenizer设置
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )
    
    # 初始化模型（直接从预训练路径加载，multi_modal_projector 随机初始化）
    print('Initializing LlavaForConditionalGeneration...')
    model = LlavaForConditionalGeneration(
        llava_config, 
        projector_config=projector_config,
        llama2_path=llama2_path,  # 直接传入预训练路径
        vit_path=vit_path  # 直接传入预训练路径
    )


    model.resize_token_embeddings(len(tokenizer))
    print('resize token embeddings finished...')

    ## multi_model_projector的参数设置为BF16
    for name, param in model.model.multi_modal_projector.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    ## lm_head的参数设置为BF16
    for name, param in model.lm_head.named_parameters():
        param.data = param.data.to(torch.bfloat16)

    target_modules = []
    lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
    projs = lora_trainable.split(',')


    ### select llm parameters to target_modules, if the parameter name contains the proj names, then add to target_modules
    for name, param in model.named_modules():
        if('language_model' in name):
            for proj in projs:
                if(proj in name):
                    target_modules.append(name)
                    break


    r=4
    lora_alpha=16
    lora_dropout=0.05
    lora_config = LoraConfig(
        inference_mode = False,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type='CAUSAL_LM',
        attn_weight=model_args.attn_weight,
    )


    model=PeftMixedModel(model,lora_config,adapter_name='image')
    model.add_adapter('text',lora_config)
    model.set_adapter(['image','text'])


    ## load weight from model_args.model_path
    state_dict = load_file(os.path.join(model_args.model_path, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=True)

    print('load weight from',model_args.model_path,'finished...')




    model.cuda()
    model.eval()


    test_dataset=TrainDataset(model_args,image_processor,tokenizer)

    collator = DataCollatorForTrainDataset()

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collator, sampler=sampler)

    sampler.set_epoch(0)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # results_dir=os.path.join(model_args.results_path,model_args.model_name_or_path.split('/')[-1],model_args.eval_dataset)


    name_tag=model_args.name_tag

    results_dir='eval_results/MME/'+name_tag

    if rank == 0:
        if os.path.exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    fp=os.path.join(results_dir,f"result_rank{rank}.jsonl")


    from tqdm import tqdm

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            answer=data.pop('answer')[0]
            question=data.pop('question')[0]
            image_path=data.pop('image_path')[0]
            subtask=data.pop('subtask')[0]


            data.update(
                        {
                            'max_new_tokens':50,
                            'pad_token_id':tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            'do_sample':False,
                        })
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v):
                        data[k] = v.to(dtype=torch.bfloat16, device='cuda')
                    else:
                        data[k] = v.cuda()
            output = model.generate(**data)
    
        
            input_length = data['input_ids'].shape[1]
            outputs = output[:, input_length:]
            outputs = tokenizer.batch_decode(outputs,skip_special_tokens=False)

            
            result = {
                "rank": rank,
                "index": index,
                "subtask": subtask,
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "output": outputs
            }

            write2json(fp=fp,dict_data=result)



