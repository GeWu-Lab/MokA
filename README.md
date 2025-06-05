# MokA

<p align="center">
    <img src="assets/moka.png" width="150" style="margin-bottom: 0.2;"/>
<p>


<h3 align="center"> 
MokA: Multimodal Low-Rank Adaptation for MLLMs</a></h3>



<h5 align="center">

## 🚀 Quick Start
### prepare dataset
1. 'dataset/pretrain_dataset.py': set pretrain dataset
2. 'dataset/unified_dataset.py': set fine-tuning dataset

### pre-train projectors
Here we take visual projector as an example:
```
sh scripts/pretrain/pretrain_visual.sh
```

### fine-tuning
Here we take visual-text case as an example:
```
sh scripts/finetune/ft.sh
```

### inference
```
sh scripts/finetune/infer_0.sh
```