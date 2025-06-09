import json
import ast
import os
import csv
import librosa
from os.path import join
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
import torchaudio.compliance.kaldi as ta_kaldi

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from decord import VideoReader
from transformers import CLIPImageProcessor

from dataset.audio_processor import preprocess

'''
Image caption
Video caption
Audio caption
Grounded VQA
'''
class PretrainDataset(Dataset):

    def __init__(
        self,
        image_annotation_path='video-llava/train_json/llava_image_.json',
        video_annotation_path='video-llava/train_json/valid_valley_.json',
        video_llava_data_root='video-llava',
        image_caption_task=False,
        video_caption_task=False,
        image_size = 224,
        video_frame_nums = 8,
        audiocaps_data_root='AudioCaps',
        audio_caption_task=False,
        video_processor: CLIPImageProcessor = None,
        # audio_processor=None,
        tokenizer: transformers.PreTrainedTokenizer = None,
    ) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums

        self.samples = []

        if image_caption_task:
            self.add_image_caption_samples(image_annotation_path,video_llava_data_root,max_sample_nums=None)
        
        if video_caption_task:
            self.add_video_caption_samples(video_annotation_path,video_llava_data_root,max_sample_nums=None)
    
        if audio_caption_task:
            self.add_audio_caption_samples(audiocaps_data_root,max_sample_nums=None)


        self.video_processor = video_processor
        self.tokenizer = tokenizer
        

    def add_image_caption_samples(self,image_annotation_path,video_llava_data_root,max_sample_nums=None):
        tot = 0
        with open(image_annotation_path,'r') as f:
            samples = json.load(f)
            for sample in samples:
                image = sample['image']
                image_path = join(video_llava_data_root,image)
                conversations = sample['conversations']
                instruction = conversations[0]['value']
                question = instruction.replace('<image>','')
                question = question.replace('\n','')
                instruction = f'This is an image:\n<image_start><image><image_end>\nPlease answer the question:\n{question}'
                output = conversations[1]['value']
                if output[-1] not in ['.','!','?']:
                    output += '.'
                self.samples.append(
                    {
                        'task_name':'image_caption',
                        'image':image_path,
                        'instruction':instruction,
                        'output':output,
                        'question':question
                    }
                )
                tot+=1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'image caption sample nums: {tot}')

    
    def add_video_caption_samples(self,video_annotation_path,video_llava_data_root,max_sample_nums=None):
        tot = 0
        with open(video_annotation_path,'r') as f:
            samples = json.load(f)
            for sample in samples:
                video = sample['video']
                video_path = join(video_llava_data_root,video)
                conversations = sample['conversations']
                instruction = conversations[0]['value']
                question = instruction.replace('<video>','')
                question = question.replace('\n','')
                instruction = f'This is a video:\n<video_start><video><video_end>\nPlease answer the question:\n{question}'
                output = conversations[1]['value']
                if output[-1] not in ['.','!','?']:
                    output += '.'
                self.samples.append(
                    {
                        'task_name':'video_caption',
                        'video':video_path,
                        'instruction':instruction,
                        'output':output,
                        'question':question
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'video caption sample nums: {tot}')


    def add_audio_caption_samples(self,audiocaps_data_root,max_sample_nums=None):
        '''AudioCaps data'''
        tot = 0
        with open(join(audiocaps_data_root,'train.json'),'r') as f:
            samples = json.load(f)
            for i,sample in enumerate(samples):
                audiocap_id = sample['audiocap_id']
                if audiocap_id == '12347':
                    continue
                start_time = sample['start_time']
                caption = sample['caption']
                # if(len(caption)>100):
                #    caption=caption[:100]
                audio_path = join(audiocaps_data_root,'data',f'{audiocap_id}.wav')
                self.samples.append(
                    {
                        'audio':audio_path,
                        'task_name':'audio_cap',
                        'instruction':'This is an audio:\n<audio_start><audio><audio_end>\nPlease describe this audio.',
                        'output':caption,
                        'question':'Please describe this audio.',
                    }
                )
                tot += 1 
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        with open(join(audiocaps_data_root,'val.json'),'r') as f:
            samples = json.load(f)
            for i,sample in enumerate(samples):
                audiocap_id = sample['audiocap_id']
                start_time = sample['start_time']
                caption = sample['caption']
                # if(len(caption)>100):
                #    caption=caption[:100]
                audio_path = join(audiocaps_data_root,'data',f'{audiocap_id}.wav')
                self.samples.append(
                    {
                        'audio':audio_path,
                        'task_name':'audio_cap',
                        'instruction':'This is an audio:\n<audio_start><audio><audio_end>\nPlease describe this audio.',
                        'output':caption,
                        'question':'Please describe this audio.',
                    }
                )
                tot += 1
                if max_sample_nums is not None and tot >= max_sample_nums:
                    break
        
        print(f'AudioCaps sample nums: {tot}')

    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]

        instruction = sample['instruction']
        output = sample['output']
        task_name = sample['task_name']
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            output = output + '<|eot_id|>'  # qwen2
            #output = output + '  <|endoftext|>'  # llama2

        data = {
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        
        image = sample.get('image',None)
        video = sample.get('video',None)
        audio = sample.get('audio',None)

        if image is not None:
            image = Image.open(image).convert('RGB')
            image = image.resize((self.image_size,self.image_size))
            image = self.video_processor.preprocess([image],return_tensors='pt')
            image = image['pixel_values']  # t,c,h,w
            data['<image>']=image

        if video is not None:
            vr = VideoReader(uri=video, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['<video>']=video

        if audio is not None:
            audio, sr = librosa.load(audio,sr=16000,mono=True)
            # print('duration: ',len(audio)/sr)
            if len(audio) < sr: # < 1s
                sil = np.zeros(sr-len(audio), dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            # audio = audio[: 60 * sr]

            window_size = 1 * sr # 1s
            max_duration = len(audio) // window_size
            if len(audio) % window_size != 0:
                max_duration += 1
                pad_length = window_size - len(audio) % window_size
                sil = np.zeros(pad_length,dtype=float)
                audio = np.concatenate((audio,sil),axis=0)
            step = 1
            audio_feature = []
            # print('max_duration: ',max_duration)
            for i in range(0,max_duration,step):
                start = int(i*sr)
                end = int((i + step)*sr)
                audio_seg = torch.from_numpy(audio[start:end]).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['<audio>'] = audio_feature


        return data



@dataclass
class DataCollatorForPretrainDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer

        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names =[]

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            X_modals = {}
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids
        
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            image = instance.get('<image>',None)
            if image is not None:
                X_modals['<image>'] = image

            video = instance.get('<video>',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('<audio>',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            
            batch_X_modals.append(X_modals)
        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names,
        }



def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train'
):
    if mode == 'train':
        dataset = PretrainDataset(
            image_size=data_args.image_size,
            video_frame_nums=data_args.video_frame_nums,
            image_caption_task=data_args.image_caption_task,
            video_caption_task=data_args.video_caption_task,
            audio_caption_task=data_args.audio_caption_task,
            video_processor=image_processor,
            tokenizer=tokenizer,
        )
        data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    

    return dataset,data_collator


