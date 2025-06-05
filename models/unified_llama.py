import json
import torch
from torch import nn
from typing import Optional,List,Tuple
from transformers import AutoModelForCausalLM,LlamaConfig,AutoConfig
from models.modeling_llama import LlamaForCausalLM,LlamaModel
#from transformers import LlamaForCausalLM,LlamaModel
from models.unified_arch import UnifiedMetaModel,UnifiedMetaForCausalLM

class UnifiedConfig(LlamaConfig):
    model_type = "unified_llm"


class UnifiedModel(UnifiedMetaModel,LlamaModel):
    config_class = UnifiedConfig

    def __init__(self, config: LlamaConfig):
        super(UnifiedModel, self).__init__(config)
        self.config = config

def is_avs_task(task_name):
    return task_name in ['ms3','s4','avss','ref-avs']


class UnifiedForCausalLM(LlamaForCausalLM,UnifiedMetaForCausalLM):
    config_class = UnifiedConfig

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config)
        self.config=config
        self.model = UnifiedModel(config,**kwargs)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.is_avs_task = False


    def get_model(self) -> UnifiedModel:
        return self.model


    def forward(
        self,
        batch_input_ids = None,
        batch_labels = None,
        batch_X_modals = None,
        batch_task_names = None,
        # used for inference
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        
        self.is_avs_task = False
        if self.is_avs_task:
            device = self.device
            avs_batch_mask = [is_avs_task(task_name) for task_name in batch_task_names]
            avs_batch_mask = torch.tensor(avs_batch_mask,device=device).bool()
            has_avs_task = avs_batch_mask.int().sum(dim=0) > 0
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_task_names=batch_task_names,
                return_multi_scale_features=True if has_avs_task else False, # vit feature
                return_gt_mask=True if has_avs_task else False,
            )
            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']
            mask_token_mask = inputs.get('mask_token_mask',None)
            multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            gt_mask = inputs.get('gt_mask',None)

            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )
            if not has_avs_task:
                return output
            
            ### avs mask loss
            output_hidden_states = output.hidden_states
            last_hidden_states = output_hidden_states[-1] # last layer
            avs_hidden_states = last_hidden_states[avs_batch_mask]
            avs_nums, seq_len, dim = avs_hidden_states.shape
            pred_embeddings = avs_hidden_states[mask_token_mask]
            pred_embeddings = pred_embeddings.reshape(avs_nums,-1,dim)
            # print('task_names: ',batch_task_names,' avs_nums: ',avs_nums,' gt_mask: ',gt_mask.shape)
            assert avs_nums == gt_mask.shape[0]
            assert avs_nums == multi_scale_image_features[0].shape[0]
            
            pred_embeddings = pred_embeddings.detach().clone()
            seg_output = self.model.postprocess_seg(
                pred_embeddings = pred_embeddings, # avs_nums,N,dim
                multi_scale_image_feature_list = multi_scale_image_features, # [(avs_nums,256,dim), (avs_nums,256,dim), ...]
                gt_mask = gt_mask, # avs_nums,1,224,224
                batch_task_names = batch_task_names,
            )
            mask_loss = seg_output['mask_loss']
            output.loss = output.loss + mask_loss
            return output

        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_task_names=batch_task_names,
                return_multi_scale_features=False, # vit feature
                return_gt_mask=False,
            )
            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']

            mask_token_mask = inputs.get('mask_token_mask',None)
            multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            gt_mask = inputs.get('gt_mask',None)
            
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            # output_hidden_states=False,
        )
        
        return output


    def forward_avs(
        self,
        batch_input_ids = None,
        batch_labels = None,
        batch_X_modals = None,
        batch_task_names = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_task_names=batch_task_names,
                return_multi_scale_features=True, # vit feature
                return_gt_mask=True,
            )
            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']

            mask_token_mask = inputs.get('mask_token_mask',None)
            multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            # print('multi_scale: ',len(multi_scale_image_features), ' shape: ',multi_scale_image_features[0].shape)
            gt_mask = inputs.get('gt_mask',None)
            # print('multi scale feature: ',multi_scale_image_features[0].shape, multi_scale_image_features[1].shape)
            # print('mask token mask: ',mask_token_mask)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        ### avs mask loss
        output_hidden_states = output.hidden_states
        avs_hidden_states = output_hidden_states[-1] # last layer
        # avs_hidden_states = output_hidden_states[avs_task_mask]
        avs_nums, seq_len, dim = avs_hidden_states.shape
        # assert multi_scale_image_features[0].shape[0] == avs_nums
        # assert gt_mask.shape[0] == avs_nums
        # assert mask_token_mask.shape[0] == avs_nums
        pred_embeddings = avs_hidden_states[mask_token_mask]
        pred_embeddings = pred_embeddings.reshape(avs_nums,-1,dim)
        seg_output = self.model.postprocess_seg(
            pred_embeddings = pred_embeddings, # avs_nums,N,dim
            multi_scale_image_feature_list = multi_scale_image_features, # [(avs_nums,256,dim), (avs_nums,256,dim), ...]
            gt_mask = gt_mask, # avs_nums,1,224,224
            batch_task_names = batch_task_names,
        )
        mask_loss = seg_output['mask_loss']
        output.loss = mask_loss
        return output
        

    @torch.no_grad()
    def generate(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_task_names,
        **kwargs
    ):
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids = batch_input_ids,
            batch_labels = batch_labels,
            batch_X_modals = batch_X_modals,
            return_multi_scale_features=False,
            return_gt_mask=False,
            batch_task_names=batch_task_names
        )
        inputs_embeds = inputs['inputs_embeds']
        return super().generate(
            inputs_embeds = inputs_embeds[0],
            output_hidden_states=False,
            return_dict_in_generate=False,
            **kwargs
        )


    @torch.no_grad()
    def generate_avs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_task_names,
        **kwargs
    ):
        # assert len(batch_task_names) == 1
        # task_name = batch_task_names[0]
        self.is_avs_task = True
        
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids = batch_input_ids,
            batch_labels = batch_labels,
            batch_X_modals = batch_X_modals,
            return_multi_scale_features=True,
            return_gt_mask=True,
            batch_task_names=batch_task_names
        )
        input_ids = inputs['input_ids']
        inputs_embeds = inputs['inputs_embeds']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        position_ids = inputs['position_ids']
        mask_token_mask = inputs.get('mask_token_mask',None)
        multi_scale_image_features = inputs.get('multi_scale_image_features',None)
        gt_mask = inputs.get('gt_mask',None)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        avs_hidden_states = output_hidden_states[-1] # last layer
        avs_nums, seq_len, dim = avs_hidden_states.shape
        pred_embeddings = avs_hidden_states[mask_token_mask]
        pred_embeddings = pred_embeddings.reshape(avs_nums,-1,dim)
        seg_output = self.model.postprocess_seg(
            pred_embeddings = pred_embeddings, # avs_nums,N,dim
            multi_scale_image_feature_list = multi_scale_image_features, # [(avs_nums,256,dim), (avs_nums,256,dim), ...]
            gt_mask = None, # avs_nums,1,224,224
            batch_task_names = batch_task_names,
        )
        # pred_masks = seg_output['pred_masks']  # list, [(cls,224,224), (cls,224,224), ...]
        return seg_output
    
        # output_hidden_states = output.hidden_states
        # output_ids = output.sequences
        # seg_token_mask = torch.zeros_like(output_ids[:, 1:]).bool()
        # special_tokens = [f'<mask_{i}>' for i in range(6)]
        # seg_token_idx = [self.SPECIAL_TOKEN_2_IDS[special_token] for special_token in special_tokens]
        # for idx in seg_token_idx:
        #     seg_token_mask = seg_token_mask | (output_ids[:, 1:] == idx)
        # mask_list =  seg_token_mask.int().tolist()[0]  # bs == 1
        # pred_embeddings = []
        # for item, hs in zip(mask_list,output_hidden_states):
        #     if item == 1:
        #         pred_embeddings.append(hs[-1])
        
        # if len(pred_embeddings) == 0:
        #     print('len(pred_embeddings) == 0')
        #     return None
        # pred_embeddings = torch.cat(pred_embeddings,dim=1)
        # if pred_embeddings.shape[1] > 6:
        #     print(f'pred_embeddings.shape[1] > 6, shape: {pred_embeddings.shape}')
        #     pred_embeddings = pred_embeddings[:,-6:]
        # elif pred_embeddings.shape[1] < 6:
        #     print(f'pred_embeddings.shape[1] < 6, shape: {pred_embeddings.shape}')
        #     return None

        # result = self.model.postprocess_seg(
        #     pred_embeddings=pred_embeddings,
        #     multi_scale_image_feature_list=multi_scale_image_features,
        #     gt_mask=None,
        #     batch_task_names=batch_task_names,
        # )
        # result['output_ids'] = output_ids
        # return result

        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # print('into prepare inputs...   input_ids:  ',input_ids,'  past key values:  ',past_key_values is None, '   inputs_emebds: ',inputs_embeds is None)
        # if inputs_embeds is not None:
        #     print(inputs_embeds.shape)
        # if past_key_values is not None:
        #     print(f'past key values:  {past_key_values[10][0].shape}')
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # print(f'_inputs>>>>>  {_inputs.keys()}')
        # if 'input_ids' in _inputs.keys():
        #     print(_inputs['input_ids'])
        # if 'inputs_embeds' in _inputs.keys():
        #     print(_inputs['inputs_embeds'].shape)
        if images is not None:
            _inputs['images'] = images
        return _inputs

    
    @property
    def device(self):
        return list(self.parameters())[0].device
    

AutoConfig.register("unified_llm", UnifiedConfig)
AutoModelForCausalLM.register(UnifiedConfig, UnifiedForCausalLM)


