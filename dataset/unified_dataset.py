import json
import ast
import os
from os.path import join,exists
import numpy as np
import pandas as pd
import cv2,csv
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
from PIL import Image
import torch
import random
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader
from transformers import CLIPImageProcessor
import warnings


warnings.filterwarnings("ignore")
from dataset.audio_processor import preprocess



label_to_idx_path = 'label2idx.json'

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label


class UnifiedDataset(Dataset):
    def __init__(
        self,
        mode='train', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        avvp_task = False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        # audio referred image grounding task
        arig_task = False,
        # av caption task
        avcap_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums

        # if avss_task or ms3_task or s4_task or ref_avs_task:
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)

        self.samples = []
        self.tot = 0

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()
    
        
        if avvp_task:
            self.add_avvp_task_samples()

        
        print(f'tot training sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'music_avqa_valid_train_samples.json'
        avqa_data_root = 'label_file/caption_gemini/avqa_converted_label'
        tot = 0
        my_path='/'
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            _type = sample['type']
            video_path = my_path+sample['video_path'][24:]
            audio_path = my_path+sample['audio_path'][24:]
            question = sample['question']
            answer = sample['answer']
            # label_path = join(avqa_data_root,'converted_label',str(question_id)+'.txt')
            label_path = join(avqa_data_root, str(question_id)+'.txt')
            output = self.read_label(label_path)

            # if(len(output)>600):
            #     drop=len(output)-600
            #     output_new=output[drop:]
            # else:
            
            output_new=output


            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n<question_start>Please answer this question: {question}'+'<question_end>'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'type':_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    # 'question':question,
                    # 'label_path':label_path,
                    'output': output_new,
                    # 'output':simple_output,
                    'task_name':'avqa',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')
        self.tot += tot




    def add_avvp_task_samples(self):
        avvp_annotation_path = 'train_samples.json'
        avvp_data_root = '/'
        tot = 0
        with open(avvp_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']
            audio_path = join(avvp_data_root,'audio_data',vid+'.mp3')
            video_path = join(avvp_data_root,'llp_videos',vid+'.mp4')
            instruction = f'This is an image:\n<video_start><video><video_end>\n<question_start>Please answer this question: {event}'+'<question_end>'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'output': event,
                    'task_name':'avvp',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot




    def read_label(self,label_path):
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):

        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output',None)
        if output is None:
            label_path = sample['label_path']
            output = self.read_label(label_path)
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            output = output + '</s>'
        data = {
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        
        if task_name == 'avqa':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
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
            data['video'] = video
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 60
            nums_per_second = int(length / tot)
            indices = [i for i in range(0,60,6)]
            for indice in indices:
                start_time = max(0, indice - 0.5)
                end_time = min(tot, indice + 1.5)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if indice - 0.5 < 0:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1.5 > tot:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature

        elif task_name == 'avvp':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
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
            data['video'] = video
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)


            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]

                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128

            data['audio'] = audio_feature

        return data


class UnifiedTestDataset(Dataset):
    def __init__(
        self,
        mode='test', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        # avvp task
        avvp_task = False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        ref_avs_task = False,
        test_name = 'test_s',  # for ref-avs: test_s, test_u, test_n
        # audio referred image grounding task
        arig_task = False,
        # avcap task
        avcap_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        self.test_name = test_name

        # if avss_task or ms3_task or s4_task or ref_avs_task:
        token_nums = image_scale_nums * token_nums_per_scale
        mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        self.mask_token = ''.join(mask_token)
        print('mask token: ',self.mask_token)

        self.samples = []
        self.tot = 0

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()

        if avvp_task:
            self.add_avvp_task_samples()

        
        print(f'tot test sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'music_avqa_test_samples.json'
        avqa_data_root = 'music-avqa'
        tot = 0
        my_path='/'
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            questio_type = sample['type']
            video_path = my_path+sample['video_path'][24:]
            audio_path = my_path+sample['audio_path'][24:]
            question = sample['question']
            answer = sample['answer']
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n<question_start>Please answer this question: {question}'+'<question_end>'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'question_type':questio_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    'question':question,
                    'task_name':'avqa',
                    'instruction':instruction,
                    'output': answer,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')



    def add_avvp_task_samples(self):
        avvp_annotation_path = 'test_samples.json'
        avvp_data_root = '/'
        tot = 0
        with open(avvp_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']
            audio_path = join(avvp_data_root,'audio_data',vid+'.mp3')
            video_path = join(avvp_data_root,'llp_videos',vid+'.mp4')
            instruction = f'This is an image:\n<video_start><video><video_end>\n<question_start>Please answer this question: {event}'+'<question_end>'
            self.samples.append(
                {
                    'audio_path':audio_path,
                    'video_path':video_path,
                    'output': event,
                    'task_name':'avvp',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot



    def __len__(self):
        return len(self.samples)


    def read_label(self,label_path):
        if not os.path.exists(label_path):
            return 'no label.'
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __getitem__(self,idx):
        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output',None)
        if output is None:
            label_path = sample['label_path']
            output = self.read_label(label_path)
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            output = output + '</s>'
        
        data = {
            'instruction': instruction,
            'output': output,
            'task_name':task_name,
        }
        
        if task_name=='avqa':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
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
            data['video'] = video
            data['video_path'] = video_path
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 60
            nums_per_second = int(length / tot)
            indices = [i for i in range(0,60,6)]
            for indice in indices:
                start_time = max(0, indice - 0.5)
                end_time = min(tot, indice + 1.5)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if indice - 0.5 < 0:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1.5 > tot:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

            question_type = sample['question_type']
            vid = sample['vid']
            qid = sample['qid']
            data['question_type'] = question_type
            data['vid'] = vid
            data['qid'] = qid


        elif task_name == 'avvp':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
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

            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)


            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(10)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]

                if indice + 1 > tot:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data.update(
                {
                    'audio':audio_feature,
                    'video':video,
                    'audio_path':audio_path,
                    'video_path':video_path,
                }
            )


        return data



@dataclass
class DataCollatorForUnifiedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names = []

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
            
            batch_X_modals.append(X_modals)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names
        }


@dataclass
class DataCollatorForUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            metadata = {
                'instruction': instruction,
                'output': output,
            }
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
                metadata['mask_path'] = instance.get('mask_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train',
    image_scale_nums = 2, token_nums_per_scale = 3, test_name = 'test_s',
):
    if mode == 'train':
        dataset = UnifiedDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task,
            avvp_task = data_args.avvp_task,
            arig_task = data_args.arig_task, 
            avss_task=data_args.avss_task,
            ms3_task=data_args.ms3_task,
            s4_task=data_args.s4_task,
            ref_avs_task=data_args.ref_avs_task,
            avcap_task=data_args.avcap_task,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedDataset(tokenizer=tokenizer)
    
    elif mode == 'test':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task,
            avvp_task = data_args.avvp_task,
            arig_task = data_args.arig_task, 
            avcap_task = data_args.avcap_task,
            avss_task=data_args.avss_task,
            ms3_task=data_args.ms3_task,
            s4_task=data_args.s4_task,
            ref_avs_task=data_args.ref_avs_task,
            test_name=test_name,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    
    return dataset,data_collator


