# MokA


<p align="center">
    <img src="assets/moka.png" width="150" style="margin-bottom: 0.2;"/>
<p>


<h3 align="center"> <a href="https://arxiv.org/abs/2506.05191"> MokA: Multimodal Low-Rank Adaptation for MLLMs</a></h3>



<table width="100%" border="0" frame="void" rules="none" style="border-collapse: collapse;">
    <tr>
        <td align="center" style="border: none;">
            <img src='assets/moka.svg ' width="120%" height="120%" valign="center">
        </td>
        <td align="center" style="border: none;">
            <img src='assets/radar.svg ' width="80%" height="80%" valign="center">
        </td>
    </tr>
</table>


## 🌴 Update
We have updated the organized code, checkpoint and predicted results of visual-text case!

Please refer to MokA_VisualText/README.md.


## 🚀 Quick Start

### 🛠️ Requirements and Installation
Basic Dependencies:
* Python == 3.9
* Pytorch == 2.1.0
* transformers == 4.37.2
* deepspeed == 0.12.6

### 🥑 Used pre-trained weights:
Multi-modal Encoder Weights:
- download visual encoder [openai-clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- download audio encoder [Fine-tuned BEATs_iter3+ (AS2M)](https://github.com/microsoft/unilm/blob/master/beats/README.md)

LLM Weights:
- download [LLaMA-2-Chat-HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

### 🌴 Prepare datasets
In this repo, we take the audio-visual-text case as an example. Pretrain based on llama2-7b-chat-hf model.

- Download image and video pretrain dataset from [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md);
- Download audio pretrain dataset from [AudioCaps](https://github.com/cdjkim/audiocaps);
- The used fine-tuning dataset is MUSIC-AVQA. Prepare the corresponding data and annotation [Here](https://github.com/GeWu-Lab/Crab?tab=readme-ov-file).
- In addition, get data and annotation of the AVE dataset [Here](https://github.com/GeWu-Lab/Crab?tab=readme-ov-file).

Set the path of pretrain dataset at:
```
dataset/pretrain_dataset.py
```
Set the path of finetuning dataset at
```
dataset/unified_dataset.py
```


## 🔑 Training

replace necessary path of google-bert-base-uncased, clip-vit-large-patch14 and BEATs in:
```
models/multimodal_encoder.py
models/unified_arch.py
```

### 🔥 Stage 1: pre-train projectors
It takes about 24h to pre-train the visual projector, using 20 A100 40g GPUs:
```
sh scripts/pretrain/pretrain_visual.sh
```
It takes about 1h to pre-train the audio projector, using 16 A100 40g GPUs:
```
sh scripts/pretrain/pretrain_audio.sh
```
We also release our pre-trained projectors for llama2-7b-chat-hf: Download [audio projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/audio_pretrain.bin), [visual projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/visual_pretrain.bin).

### 🔥 Stage 2: fine-tuning
Set the path of pre-trained projectors of line 134-135 at:
```
sh scripts/finetune/finetune.py

```

Here we take MUSIC-AVQA as an example, and it takes about 5-6h, using 16 A100 40g GPUs:
```
sh scripts/finetune/ft.sh
```

## 🤖 Inference
Here we take MUSIC-AVQA as an example, run
```
sh scripts/finetune/infer.sh
```

## 🤖 Evaluation
Here we take MUSIC-AVQA as an example, run
```
python evaluation.py
```



## 📃 BibTeX
```bibtex
@article{wei2025moka,
  title={MokA: Multimodal Low-Rank Adaptation for MLLMs},
  author={Wei, Yake and Miao, Yu and Zhou, Dongzhan and Hu, Di},
  journal={arXiv preprint arXiv:2506.05191},
  year={2025}
}
```
