import re
from typing import Dict,Union,Any,Mapping
import torch
import jsonlines
import transformers


def load_ckpt(path:str):
    if path.endswith('.safetensors'):
        ckpt = {}
        from safetensors import safe_open
        with safe_open(path,framework='pt',device='cpu') as f:
            for k in f.keys():
                ckpt[k] = f.get_tensor(k)
        return ckpt
    elif (path.endswith('.bin')) or (path.endswith('.pth')):
        return torch.load(path,map_location='cpu')
    


def rank0_print(local_rank,*args):
    if local_rank == 0:
        print(*args)


def rank0write2txt(local_rank,fp,info,mode='a'):
    if local_rank == 0:
        with open(fp,mode=mode) as f:
            f.write(info)
            f.write('\n')


def prepare_sample(data: Union[torch.Tensor, Any], device='cuda', dtype=None) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_sample(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_sample(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if dtype is not None:
            # kwargs.update({"dtype": dtype})
            data = data.to(dtype=dtype)
        return data.to(**kwargs)
    return data


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def write2txt(fp,info,mode='a'):
    with open(fp,mode=mode) as f:
        f.write(info)
        f.write('\n')


def write2json(fp,dict_data,mode='a'):
    with jsonlines.open(fp,mode=mode) as f:
        f.write(dict_data)
    

def get_mask_from_string(s):
    pattern = r"<vqgan_(\d+)>"
    numbers = re.findall(pattern, s)
    numbers = [int(item) for item in numbers]
    return numbers

