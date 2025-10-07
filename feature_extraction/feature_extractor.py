import torchaudio
import torch

def extract_mfcc(waveform, sample_rate, n_mfcc=13):
    """
    MFCC özelliklerini çıkarır.
    
    Args:
        waveform (Tensor): Ses dalgası.
        sample_rate (int): Örnekleme frekansı.
        n_mfcc (int): MFCC katsayı sayısı.
        
    Returns:
        mfcc (Tensor): MFCC özellikleri.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

def extract_mel_spectrogram(waveform, sample_rate):
    """
    Mel-spectrogram özelliklerini çıkarır.
    
    Args:
        waveform (Tensor): Ses dalgası.
        sample_rate (int): Örnekleme frekansı.
        
    Returns:
        mel_spec (Tensor): Mel-spectrogram özellikleri.
    """
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=23
    )
    mel_spec = mel_spectrogram_transform(waveform)
    return mel_spec