## Visual-Text case


### Train

Stage 1 is the same as the audio-visual-text case. And the used visual projector is also the same: [visual projector checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/visual_pretrain.bin).

Stage 2: SFT

```
sh MokA_VisualText/shell/train.sh
```

Download the used json of train data [HERE](https://drive.google.com/file/d/1EEM3AsO6da_Hbb9I4YnhTN2V29mzS-rp/view?usp=drive_link). A small set of multiple-choice type instructions is integrated besides the original LLaVA-Instruct-150K.

### Inference
Use the last checkpoint. Take MMBench as an example, the command would be:
```
sh MokA_VisualText/eval_benchmarks/mmbench/mmbench.sh
```

Then, calculate the Acc/Score:
```
python MokA_VisualText/eval_benchmarks/mmbench/eval_mmbench.py
```

Our predicted results of specific benchmarks are shown in the corresponding `merged.jsonl` files.


### Results

| Dataset | Acc/Score |
| --- | --- |
| MME_perception | 1105.51 |
| MMBench | 56.02 |
| POPE | 77.07 |
| SEED-Bench | 40.80 |

### Our checkpoint
https://huggingface.co/yake0409/MokA_VisualText/tree/main