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


class TrainDataset(Dataset):
    def __init__(self, config,image_processor,tokenizer):
        self.config = config
        self.image_processor = image_processor
        self.tokenizer = tokenizer


        json_path='data/llava_instruct_150k_aokvqa12k.json'

        with open(json_path,'r') as f:
            samples = json.load(f)
        
        image_root_path='raw_images/coco/train2017'
        
        self.samples=[]
        for i,sample in enumerate(samples):
            image_path=os.path.join(image_root_path,sample['image'])
            conversation=sample['conversations']
            
            # make sinlge-turn conversation
            for j in range(0, len(conversation), 2):  
                if j+1 < len(conversation):  
                    human_turn = conversation[j]
                    gpt_turn = conversation[j+1]
                    
                    # 确保是human和gpt的配对
                    if human_turn['from'] == 'human' and gpt_turn['from'] == 'gpt':
                        # 去除原有的<image>标记
                        human_value = human_turn['value'].replace('<image>', '').strip()
                        # 在开头插入新的格式
                        human_value = 'This is an image:\n<image_start><image><image_end>\nPlease answer this question: ' + human_value
                        
                        # 创建单轮对话
                        single_turn_conversation = [
                            {'from': 'human', 'value': human_value},
                            {'from': 'gpt', 'value': gpt_turn['value']}
                        ]
                        
                        self.samples.append({
                            'image': image_path,
                            'conversation': single_turn_conversation,
                        })
    
    def __len__(self):
        return len(self.samples)
    


    def tokenizer_target(self,input_ids, return_tensors=None):

        inst_end_ids = self.tokenizer.encode('[/INST]', add_special_tokens=False)
        eos_ids = self.tokenizer.encode('</s>', add_special_tokens=False)
        
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
            if matches_token_sequence(input_ids_list, i, inst_end_ids):
                start_idx = i + len(inst_end_ids)
                

                end_idx = -1
                for j in range(start_idx, len(input_ids_list) - len(eos_ids) + 1):
                    if matches_token_sequence(input_ids_list, j, eos_ids):
                        end_idx = j
                        break
                
                if end_idx != -1:
                    target_ids[start_idx:end_idx + len(eos_ids)] = input_ids_squeezed[start_idx:end_idx + len(eos_ids)]
                    
                    i = end_idx + len(eos_ids)
                else:
                    i += 1
            else:
                i += 1
        
        return target_ids


    
    def __getitem__(self, idx):

        image_path=self.samples[idx]['image']
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
        
        for turn in conversation:
            if turn['from'] == 'human':
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
        
        instruction = self.tokenizer.apply_chat_template(conversation=messages,tokenize=False)

        instruction = instruction.replace('<image_pad>', '<image>')



        inputs = instruction.replace('<image>', '<image>' * 32)


        ## get input_ids
        input_ids = self.tokenizer.encode(inputs, add_special_tokens=False,return_tensors='pt')



        ### replace 0 token with <image_pad>
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<image>")


        ### get my_image_mask
        my_image_mask = input_ids == image_pad_id
        my_text_mask= input_ids != image_pad_id

        input_ids = torch.where(input_ids == image_pad_id, 0, input_ids)




        target_ids = self.tokenizer_target(input_ids, return_tensors="pt").unsqueeze(0)


        image_positions = torch.where(my_image_mask)[1] 
        if len(image_positions) > 0:

            last_image_token_pos = image_positions[-1]
            seq_len = my_image_mask.size(1)
            position_mask = torch.arange(seq_len, device=my_image_mask.device) > last_image_token_pos
            position_mask = position_mask.unsqueeze(0)  
        else:
            position_mask = torch.zeros_like(my_image_mask, dtype=torch.bool)
        
        question_mask = ~(my_image_mask) & (target_ids == -100) & position_mask





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
                raise ValueError(f"Unsupported key: {key}")
            
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


    visual_adapter_path='visual_pretrain.bin'
    state_dict = torch.load(visual_adapter_path, map_location='cpu')
    embed_tokens_weight = state_dict.pop('model.embed_tokens.weight')
    ## load embed_tokens_weight to model.language_model.embed_tokens.weight
    model.language_model.embed_tokens.load_state_dict({'weight': embed_tokens_weight}, strict=False)
    print('load embed_tokens_weight to model.language_model.embed_tokens.weight finished...')

    new_state_dict = {}
    for key in state_dict.keys():
        if 'model.vl_projector.' in key:
            new_key = key.replace('model.vl_projector.', '')
            new_state_dict[new_key] = state_dict[key]
    

    ## load new_state_dict to model.model.multi_modal_projector
    model.model.multi_modal_projector.load_state_dict(new_state_dict, strict=False)

    print('load new_state_dict to model.model.multi_modal_projector finished...')

    ## multi_model_projector的参数设置为BF16
    for name, param in model.model.multi_modal_projector.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    ## lm_head的参数设置为BF16
    for name, param in model.lm_head.named_parameters():
        param.data = param.data.to(torch.bfloat16)

    print('Model initialization complete!')

    ### freeze model parameters
    ### only train model.model.multi_modal_projector parameters
    for name, param in model.named_parameters():
        if('model.multi_modal_projector' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
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
        attn_weight=model_args.attn_weight
    )


    model=PeftMixedModel(model,lora_config,adapter_name='image')
    model.add_adapter('text',lora_config)
    model.set_adapter(['image','text'])

    # 检查 attn_weight 是否正确设置
    if dist.get_rank() == 0:
        print('=' * 100)
        print(f'[CHECK] Checking attn_weight in LoRA layers after initialization...')
        print(f'[CHECK] Expected attn_weight from config: {model_args.attn_weight}')
        for name, module in model.named_modules():
            if hasattr(module, 'attn_weight'):
                print(f'[CHECK] {name}: attn_weight = {module.attn_weight}')
        print('=' * 100)

    for name, param in model.named_parameters():
        if('model.multi_modal_projector' in name):
            param.requires_grad = True
        elif('lora' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    
#==============================================

    if dist.get_rank() == 0:
        logger.info('trainable parameters: ')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    



    train_dataset=TrainDataset(model_args,image_processor,tokenizer)

    collator = DataCollatorForTrainDataset()

    model.cuda()
    model.train()


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()




