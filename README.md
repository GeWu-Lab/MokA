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




## 🚀 Quick Start

### 🛠️ Requirements and Installation
Basic Dependencies:
* Python == 3.9
* Pytorch == 2.1.0
* transformers == 4.37.2
* deepspeed == 0.12.6

### 🌴 Prepare datasets
In this repo, we take the audio-visual-text case as an example. Pretrain based on llama2-7b-chat-hf model.

- Download image and video pretrain dataset from [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md);
- Download audio pretrain dataset from [AudioCaps](https://github.com/cdjkim/audiocaps);
- The used fine-tuning dataset is MUSIC-AVQA. Prepare the corresponding data and annotation [Here](https://github.com/GeWu-Lab/Crab?tab=readme-ov-file).

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
pre-train visual projector, run:
```
sh scripts/pretrain/pretrain_visual.sh
```
pre-train audio projector, run:
```
sh scripts/pretrain/pretrain_audio.sh
```

### 🔥 Stage 2: fine-tuning
Set the path of pre-trained projectors of line 134-135 at:
```
sh scripts/finetune/finetune.py

```

Here we take MUSIC-AVQA as an example, run:
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
