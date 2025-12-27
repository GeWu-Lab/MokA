# Training and Inference Instruction

❗️❗️❗️The root path of all the following scripts is `VisualText`. 
```
cd VisualText
```

## 🔑 Training

### 🛠️ Basic Dependencies
* transformers == 4.53.2


### 🔥 Stage 1: pre-train projectors

Stage 1 is the same as the audio-visual-text case. And the used visual projector is also the same: [visual projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/visual_pretrain.bin).

### 🔥 Stage 2: fine-tuning
Set the path of the dataset, the visual projector, and the pretrained weights at (line 75, 80, 420, 421, 436, 468, 495)
```
scripts/train/train.py
```
❗️❗️ Please make sure the visual projectors and other pre-trained weights are loaded correctly!



Then, train the model:
```
sh shell/train.sh
```

The model checkpoints will be saved at `OUTPUT_DIR` of the scripts.

## 🤖 Inference
Take MMBench as an example, set the dataset path, model path, and the used pretrained weights at (line 102, 464, 465, 480, 513)
```
scripts/eval_benchmarks/mmbench/mmbench.py
```

Set your checkpoint path at (  `--model_path`):
```
sh eval_benchmarks/mmbench/mmbench.sh
```

Then, the predicted results will be saved at: `'eval_results/MMBench/'+name_tag`.


## 🤖 Evaluation
Set your JSONL path and dataset path at:
```
eval_benchmarks/mmbench/eval_mmbench.py
```

❗️❗️ Here we use `DistributedSampler` during inference, and there will be multiple JSONL files if there are multiple GPUs. Hence, we first merge these JSONL and get the final `merged.jsonl`.


Then, calculate the Acc:
```
python eval_benchmarks/mmbench/eval_mmbench.py
```


Our predicted results are available in the `merged.jsonl` files located in the corresponding benchmark directories. Our results:

| Dataset | Acc/Score |
| --- | --- |
| MME_perception | 1105.51 |
| MMBench | 56.02 |
| POPE | 77.07 |
| SEED-Bench | 40.80 |

## 🍃 Our checkpints
[[HERE]](https://huggingface.co/yake0409/MokA_VisualText/tree/main)
