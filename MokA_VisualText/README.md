## Visual-Text case


### Train
```
sh MokA_VisualText/shell/train.sh
```
Download the used json of train data [HERE](https://drive.google.com/file/d/1EEM3AsO6da_Hbb9I4YnhTN2V29mzS-rp/view?usp=drive_link).

### Inference
Use the last checkpoint. Take MMBench as an example, the command would be:
```
sh MokA_VisualText/eval_benchmarks/mmbench/mmbench.sh
```

Then, calculate the Acc/Score:
```
python MokA_VisualText/eval_benchmarks/mmbench/eval_mmbench.py
```

Our predcted results of specific benchmarks are shown in corresponding `merged.jsonl` files.


### Results

| Dataset | Acc/Score |
| --- | --- |
| MME_perception | 1105.51 |
| MMBench | 56.02 |
| POPE | 77.07 |
| SEED-Bench | 40.80 |

### Our checkpoint
https://huggingface.co/yake0409/MokA_VisualText/tree/main