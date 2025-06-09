import torch
from abc import ABC, abstractmethod
from torch import nn
import pickle

from models.multimodal_encoder import (
    VisualEncoder,
    AudioEncoder,
    VLProjector,
    ALProjector
)

class UnifiedMetaModel:

    def __init__(self, config):
        super(UnifiedMetaModel, self).__init__(config)
        self.config = config


    def init_multimodal_modules(
        self,
        d_model = 3584,
        # visual
        vit_ckpt_path = 'clip-vit-large-patch14',
        select_layer_list = [-11,-1],
        select_feature = 'patch',
        image_size = 224,
        patch_size = 14,
        visual_query_token_nums = 32,
        # audio
        BEATs_ckpt_path = 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.ptt',
        audio_query_token_nums = 32,
        visual_branch = False,
        audio_branch = False,
    ):
        
        if visual_branch:
            image_token_nums = (image_size//patch_size) * (image_size//patch_size)
            self.visual_encoder = VisualEncoder(model_name_or_path=vit_ckpt_path,select_layer_list=select_layer_list,
                                                select_feature=select_feature)
            self.vl_projector = VLProjector(hidden_size=1024, d_model=d_model, depth=2, image_token_nums=image_token_nums,
                                            num_query_token=visual_query_token_nums, num_hidden_layers=2,)
            print('init visual_encoder, vl_projector finished...')

        if audio_branch:
            self.audio_encoder =  AudioEncoder(ckpt_path=BEATs_ckpt_path)
            self.al_projector = ALProjector(hidden_size=768, d_model=d_model, depth=2, num_query_token=audio_query_token_nums,
                                            num_hidden_layers=2)
            print('init audio_encoder, al_projector finished...')


    def encode_video(self,visual,batch_question=None):
        vit_feature_list = self.visual_encoder(visual)  # [(b,t*n,d),(b,t*n,d),...]
        qformer_feature_list = []
        for vit_feature in vit_feature_list:
            qformer_feature = self.vl_projector(vit_feature,batch_question) # b,t*256 -> b,t*32
            qformer_feature_list.append(qformer_feature)
        return vit_feature_list,qformer_feature_list


    def encode_audio(self,audio,batch_qustion=None):
        audio_feature = self.audio_encoder(audio)
        audio_feature = self.al_projector(audio_feature,batch_qustion)
        return audio_feature


    def encode_mask(self,mask):
        return self.mask_encoder(mask)



class UnifiedMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> UnifiedMetaModel:
        pass

    def encode_audio(self,audio,batch_qustion=None, batch_first=True):
        if not batch_first:
            audio = audio.unsqueeze(0)
        audio_feature = self.get_model().encode_audio(audio,batch_qustion=batch_qustion)
        if not batch_first:
            audio_feature = audio_feature.squeeze(0)
        return audio_feature


    def encode_video(self,video,batch_question=None,batch_first=True):
        if not batch_first:
            video = video.unsqueeze(0)
        vit_feature_list, qformer_feature_list = self.get_model().encode_video(video,batch_question=batch_question)
        if not batch_first:
            vit_feature_list = [item.squeeze(0) for item in vit_feature_list]
            qformer_feature_list = [item.squeeze(0) for item in qformer_feature_list]
        return vit_feature_list,qformer_feature_list


    

    def encode_ids(self,ids):
        return self.get_model().embed_tokens(ids)
    
    
    def prepare_multimodal_inputs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
    ):
        device = self.device
        bs = len(batch_input_ids)


        max_length = 0
        new_batch_inputs_embeds = []
        new_batch_attention_mask = []
        new_batch_labels = []
        # keys = ['<image>','<video>','<audio>','<question_start>','<question_end>']
        keys = self.KEYS
        # print(keys)

        ## add my modality mask per batch
        new_batch_inputs_unimodal_mask_video=[]
        new_batch_inputs_unimodal_mask_text=[]
        new_batch_inputs_unimodal_mask_audio=[]

        new_batch_inputs_unimodal_mask_question=[]



        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]

            X_token_indices = torch.where(torch.any(torch.stack([input_ids == self.SPECIAL_TOKEN_2_IDS[key] for key in keys]), dim=0))[0]


            X_token_indices = X_token_indices.tolist()            

            inputs_embeds_seg=[]
            ## my modality mask per sample
            inputs_unimodal_mask_video=[]
            inputs_unimodal_mask_text=[]
            inputs_unimodal_mask_audio=[]

            inputs_unimodal_mask_question=[]


            labels_seg=[]
            pre_indice=0
            for idx,indice in enumerate(X_token_indices):
                special_token = self.IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]

                if special_token == '<question_end>':
                    # token size * emb size
                    tmp=self.encode_ids(input_ids[pre_indice:indice])


                    inputs_embeds_seg.append(tmp)
                    inputs_unimodal_mask_text.append(torch.ones((tmp.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_video.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_audio.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))

                    inputs_unimodal_mask_question.append(torch.ones((tmp.size()[0],1),dtype=torch.int32,device=device))


                    labels_seg.append(labels[pre_indice:indice])
                
                else:
                    # token size * emb size
                    tmp=self.encode_ids(input_ids[pre_indice:indice])


                    inputs_embeds_seg.append(tmp)
                    inputs_unimodal_mask_text.append(torch.ones((tmp.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_video.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_audio.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))

                    inputs_unimodal_mask_question.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))


                    labels_seg.append(labels[pre_indice:indice])


                if special_token == '<audio>':
                    feature = self.encode_audio(batch_X_modals[i][special_token],batch_qustion=None,batch_first=False)
                    inputs_embeds_seg.append(feature)
                    inputs_unimodal_mask_text.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_video.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_audio.append(torch.ones((feature.size()[0],1),dtype=torch.int32,device=device))

                    inputs_unimodal_mask_question.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))


                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                elif special_token == '<video>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature

                    inputs_embeds_seg.append(feature)  
                    inputs_unimodal_mask_text.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_video.append(torch.ones((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_audio.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))

                    inputs_unimodal_mask_question.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))


                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                    # if return_multi_scale_features and is_avs_task: # for ref-avs
                    #     for _scale in range(scale):
                    #         multi_scale_image_features[_scale].append(vit_feature_list[_scale]) # (10*256,1024)
                elif special_token == '<image>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature
                    # token_num x 4096
                    inputs_embeds_seg.append(feature)

                    inputs_unimodal_mask_text.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_video.append(torch.ones((feature.size()[0],1),dtype=torch.int32,device=device))
                    inputs_unimodal_mask_audio.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))

                    inputs_unimodal_mask_question.append(torch.zeros((feature.size()[0],1),dtype=torch.int32,device=device))
                    
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                    # if return_multi_scale_features and is_avs_task:
                    #     for _scale in range(scale):
                    #         multi_scale_image_features[_scale].append(vit_feature_list[_scale])



                pre_indice = indice + 1 # +1,skip special token

            # add last tokens
            # text token
            tmp=self.encode_ids(input_ids[pre_indice:])

            inputs_embeds_seg.append(tmp)
            inputs_unimodal_mask_text.append(torch.ones((tmp.size()[0],1),dtype=torch.int32,device=device))
            inputs_unimodal_mask_video.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))
            inputs_unimodal_mask_audio.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))
            inputs_unimodal_mask_question.append(torch.zeros((tmp.size()[0],1),dtype=torch.int32,device=device))

            labels_seg.append(labels[pre_indice:])




            # concat segs
            inputs_embeds_seg = torch.cat(inputs_embeds_seg,dim=0)
            inputs_unimodal_mask_text=torch.cat(inputs_unimodal_mask_text,dim=0)
            inputs_unimodal_mask_video=torch.cat(inputs_unimodal_mask_video,dim=0)
            inputs_unimodal_mask_audio=torch.cat(inputs_unimodal_mask_audio,dim=0)

            inputs_unimodal_mask_question=torch.cat(inputs_unimodal_mask_question,dim=0)

            assert inputs_unimodal_mask_question.size() == inputs_unimodal_mask_text.size(), "The sizes of inputs_unimodal_mask_question and input_unimodal_mask_text do not match."


            attention_mask_seg = torch.ones(inputs_embeds_seg.shape[0],dtype=torch.int32,device=device)
            labels_seg = torch.cat(labels_seg,dim=0)
            
            new_batch_inputs_embeds.append(inputs_embeds_seg)
            new_batch_inputs_unimodal_mask_text.append(inputs_unimodal_mask_text)
            new_batch_inputs_unimodal_mask_video.append(inputs_unimodal_mask_video)
            new_batch_inputs_unimodal_mask_audio.append(inputs_unimodal_mask_audio)


            new_batch_inputs_unimodal_mask_question.append(inputs_unimodal_mask_question)



            new_batch_attention_mask.append(attention_mask_seg)
            new_batch_labels.append(labels_seg)
            

            max_length = max(max_length,inputs_embeds_seg.shape[0])


        ### left padding
        padding_inputs_embeds = []
        padding_inputs_mask_text=[]
        padding_inputs_mask_video=[]
        padding_inputs_mask_audio=[]
        padding_inputs_mask_question=[]


        padding_attention_mask = []
        padding_labels = []
        padding_mask_token_mask = []

        for i in range(bs):
            embeds = new_batch_inputs_embeds[i]
            mask_text=new_batch_inputs_unimodal_mask_text[i]
            mask_video=new_batch_inputs_unimodal_mask_video[i]
            mask_audio=new_batch_inputs_unimodal_mask_audio[i]
            mask_question=new_batch_inputs_unimodal_mask_question[i]


            mask = new_batch_attention_mask[i]
            labels = new_batch_labels[i]
            
            L,d = embeds.shape
            pad_embeds = self.encode_ids(torch.full((max_length-L,),self.get_model().pad_token_id,dtype=torch.long,device=device))

            padding_inputs_embeds.append(torch.cat([pad_embeds,embeds],dim=0))
            
            temp=torch.zeros((pad_embeds.size()[0],1),dtype=torch.int32,device=device)

            padding_inputs_mask_text.append(torch.cat([temp,mask_text],dim=0))
            padding_inputs_mask_video.append(torch.cat([temp,mask_video],dim=0))
            padding_inputs_mask_audio.append(torch.cat([temp,mask_audio],dim=0))

            padding_inputs_mask_question.append(torch.cat([temp,mask_question],dim=0))
            


            padding_attention_mask.append(torch.cat([torch.zeros((max_length-L),dtype=torch.int32,device=device),mask],dim=0))
            padding_labels.append(torch.cat([torch.full((max_length-L,),-100,dtype=torch.long,device=device),labels],dim=0))
            
    
        padding_inputs_embeds = torch.stack(padding_inputs_embeds,dim=0)
        padding_inputs_mask_text=torch.stack(padding_inputs_mask_text,dim=0)
        padding_inputs_mask_video=torch.stack(padding_inputs_mask_video,dim=0)
        padding_inputs_mask_audio=torch.stack(padding_inputs_mask_audio,dim=0)
        padding_inputs_mask_question=torch.stack(padding_inputs_mask_question,dim=0)


        padding_attention_mask = torch.stack(padding_attention_mask,dim=0)
        padding_labels = torch.stack(padding_labels,dim=0)
        if len(padding_mask_token_mask) > 0:
            padding_mask_token_mask = torch.stack(padding_mask_token_mask,dim=0)

        position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        position_ids[position_ids==-1] = 0


        mask_emb=[padding_inputs_embeds,padding_inputs_mask_text,padding_inputs_mask_video,padding_inputs_mask_audio,padding_inputs_mask_question]

        
        dict_data = {
            'input_ids':None,
            'inputs_embeds':mask_emb,
            'attention_mask':padding_attention_mask,
            'labels':padding_labels,
            'position_ids':position_ids,
        }

        
        return dict_data
    

    def initialize_MM_tokenizer(self, tokenizer):
        vocab_nums=len(tokenizer)
        added_tokens = []
        image_tokens = ['<image>','<image_start>','<image_end>']
        added_tokens += image_tokens
        video_tokens = ['<video>','<video_start>','<video_end>']
        added_tokens += video_tokens
        audio_tokens = ['<audio>','<audio_start>','<audio_end>']
        added_tokens += audio_tokens
        num_new_tokens = tokenizer.add_tokens(added_tokens,special_tokens=True)


        question_tokens= ['<question_start>','<question_end>']
        num_new_tokens += tokenizer.add_tokens(question_tokens,special_tokens=True)
        added_tokens += question_tokens
        

        self.KEYS = ['<image>','<video>','<audio>','<question_start>','<question_end>']
        
        self.SPECIAL_TOKEN_2_IDS={
            token : i + vocab_nums for i,token in enumerate(added_tokens)
        }
        self.IDS_2_SPECIAL_TOKEN={
            i + vocab_nums:token for i,token in enumerate(added_tokens)
        }

        self.resize_token_embeddings(len(tokenizer))


    @property
    def device(self):
        return list(self.parameters())[0].device



