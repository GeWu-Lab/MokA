# Training and Inference Instruction

❗️❗️❗️The root path of all the following scirpts is `AudioVisualText`. 
```
cd AudioVisualText
```

## 🔑 Training

### 🛠️ Basic Dependencies
* Python == 3.9
* Pytorch == 2.1.0
* transformers == 4.37.2
* deepspeed == 0.12.6



### Set dataset path and pretrained weight path

Set the path of the pretrain dataset at:
```
dataset/pretrain_dataset.py
```
Set the path of finetuning dataset at (line 59, 92, 93, 283, 315, 316):
```
dataset/unified_dataset.py
```

It needs to replace the `"audio_path"` and `video_path` in the JSON files of the MUSIC-AVQA dataset, and put the raw video & audio into the data folders, like:

```
├── AVE_data
│   ├── converted_label
│   │        ├──sample_id.txt
│   │        └──.....
│   ├── audio_data
│   │        ├──sample_id.mp3
│   │        └──.....
│   ├── AVE
│   │    ├──sample_id.mp4
│   │    └──.....
│   ├── train_samples_ave.json
│   ├── test_samples_ave.json
│   └── Annotations.txt
└── MUSIC_AVQA_data
    ├── train_samples_with_reasoning_avqa.json
    ├── test_samples_avqa.json
    ├── audio_data
    │        ├──sample_id.mp3
    │        └──.....
    └── video_data
             ├──sample_id.mp4
             └──.....

```


Replace necessary path of google-bert-base-uncased, clip-vit-large-patch14 and BEATs in:
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

❗️❗️ Set the ckpt path of all used pre-trained weights in the `.sh` files, e.g., llama_ckpt_path, vit_ckpt_path, BEATs_ckpt_path.


🥥🥥 We also release our pre-trained projectors for llama2-7b-chat-hf: Download [audio projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/audio_pretrain.bin), [visual projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/visual_pretrain.bin).

### 🔥 Stage 2: fine-tuning
Set the path of pre-trained projectors of line 134-135 at:
```
scripts/finetune/finetune.py
```
❗️❗️ Please make sure the visual/audio projectors and other pre-trained weights are loaded correctly!


❗️❗️ Set the ckpt path of all used pre-trained weights in the `ft_musicavqa.sh` and `ft_ave.sh` files, e.g., llama_ckpt_path, vit_ckpt_path, BEATs_ckpt_path.

Finetuning on MUSIC-AVQA dataset:
```
sh scripts/finetune/ft_musicavqa.sh
```
Finetuning on AVE dataset:
```
sh scripts/finetune/ft_ave.sh
```

The model checkpoints will be saved at `output_dir` of the scripts. We need to use `adapter_model.bin` and `non_lora_trainables.bin` in the checkpoint folders for inference.

## 🤖 Inference
❗️❗️ Set the ckpt path (`$YOUR_CKPT_PARH`) of fine-tuned models in the `infer_ave.sh` and `infer_avqa.sh`.

❗️❗️ Set the ckpt path of all used pre-trained weights in the `ft_musicavqa.sh` and `ft_ave.sh` files, e.g., llama_ckpt_path, vit_ckpt_path, BEATs_ckpt_path.

Make sure all the weights are loaded correctly.

Inference on MUSIC-AVQA dataset:
```
sh scripts/finetune/infer_ave.sh
```

Inference on AVE dataset:
```
sh scripts/finetune/infer_ave.sh
```

Then, the prediced results will be saved at `$YOUR_CKPT_PARH/inference_results/inference_avqa.jsonl` or `$YOUR_CKPT_PARH/inference_results/inference_ave.jsonl` 

## 🤖 Evaluation

Set your JSONL path and dataset path at:
```
scripts/evaluation/ave_eval.py
scripts/evaluation/avqa_eval.py
```

Our predicted results are provided at:
```
scripts/evaluation/inference_ave.jsonl
scripts/evaluation/inference_avqa.jsonl
```

Then, get the accuracy:
```
# MUSIC-AVQA:
python scripts/evaluation/ave_eval.py

# AVE:
python scripts/evaluation/avqa_eval.py
```

Our results are:
| Dataset | Total samples| Samples with correct prediction format  | Acc (Right answer/Samples with correct prediction format)|  Acc (Right answer/Total samples)| 
| --- | --- | --- |--- |--- |
| AVE |402|398| 77.21 | 76.44 |
| MUSIC-AVQA | 9129|9035|77.26 |76.46|



## 🍃 Our checkpints

- [MUSIC-AVQA](https://huggingface.co/yake0409/MokA_AudioVisualText/tree/main/AVQA_checkpoint)
- [AVE](https://huggingface.co/yake0409/MokA_AudioVisualText/tree/main/AVE_data)
