import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

class AudioProcessor:

    def __init__(
        self,
        sr = 16000,
        mono = True,
        duration = 60,
    ) -> None:
        self.sr = sr
        self.mono = mono
        self.duration = duration
    

    def forward(self,audio):
        audio, sr = librosa.load(audio,sr=self.sr,mono=self.mono,duration=self.duration)
        if len(audio) < sr:
            sil = np.zeros(sr-len(audio), dtype=float)
            audio = np.concatenate((audio,sil),axis=0)
        audio = audio[: 60 * sr]
        audio = torch.from_numpy(audio).to(torch.float32) # L,
        return audio


def preprocess(
    source: torch.Tensor,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank

