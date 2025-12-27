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
[2025-12-27] We have updated the organized code, checkpoint, and predicted results!



## 🚀 Quick Start


### 🥑 Used pre-trained weights:
Multi-modal Encoder Weights:
- download visual encoder [openai-clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- download audio encoder [Fine-tuned BEATs_iter3+ (AS2M)](https://github.com/microsoft/unilm/blob/master/beats/README.md)

LLM Weights:
- download [LLaMA-2-Chat-HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

### 🌴 Prepare datasets
In this repo, we take the audio-visual-text and visual-text case as an example. Pretrain based on llama2-7b-chat-hf model.

#### Stage 1 dataset:

- Download image and video pretrain dataset from [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md);
- Download audio pretrain dataset from [AudioCaps](https://github.com/cdjkim/audiocaps).
  

#### Stage 2 dataset:
##### Audio-Visual-Text: 
- AVE annotation & JSON: [HERE](https://huggingface.co/yake0409/MokA_AudioVisualText/tree/main/AVE_data)
- AVE raw video: [HERE](https://github.com/YapengTian/AVE-ECCV18)
- MUSIC-AVQA annotation & JSON: [HERE](https://huggingface.co/yake0409/MokA_AudioVisualText/tree/main/MUSIC_AVQA_data)
- MUSIC-AVQA raw video: [HERE](https://github.com/GeWu-Lab/MUSIC-AVQA)


##### Visual-Text: 
- Download the used JSON of train data [HERE](https://drive.google.com/file/d/1EEM3AsO6da_Hbb9I4YnhTN2V29mzS-rp/view?usp=drive_link). A small set of multiple-choice type instructions is integrated with the original LLaVA-Instruct-150K.



## 🔑 Training

### Audio-Visual-Text case

Read [AudioVisualText/README_AVT.md](https://github.com/GeWu-Lab/MokA/blob/main/AudioVisualText/README_AVT.md) for the detailed information.

### Visual-Text case
Read [VisualText/README_VT.md](https://github.com/GeWu-Lab/MokA/blob/main/VisualText/README_VT.md) for the detailed information.


## 📃 BibTeX
```bibtex
@article{wei2025moka,
  title={MokA: Multimodal Low-Rank Adaptation for MLLMs},
  author={Wei, Yake and Miao, Yu and Zhou, Dongzhan and Hu, Di},
  journal={arXiv preprint arXiv:2506.05191},
  year={2025}
}
```
