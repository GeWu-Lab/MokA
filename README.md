# MokA

<p align="center">
    <img src="assets/moka.png" width="150" style="margin-bottom: 0.2;"/>
<p>


<h3 align="center"> <a href="https://arxiv.org/abs/2506.05191"> MokA: Multimodal Low-Rank Adaptation for MLLMs</a></h3>


<table width="100%" frame=void>
    <tr>
        <td align="center">
            <img src='assets/moka.svg ' width="120%" height="120%" valign="center">
        </td>
        <td align="center">
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
Set pretrain dataset at
```
dataset/pretrain_dataset.py
```
Set finetuning dataset at
```
dataset/unified_dataset.py
```

## 🔑 Training

### 🔥 Stage 1: pre-train projectors
Here we take visual projector as an example, run:
```
sh scripts/pretrain/pretrain_visual.sh
```

### 🔥 Stage 2: fine-tuning
Here we take visual-text case as an example, run:
```
sh scripts/finetune/ft.sh
```

## 🤖 Inference
Here we take visual-text case as an example, run:
```
sh scripts/finetune/infer_0.sh
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
