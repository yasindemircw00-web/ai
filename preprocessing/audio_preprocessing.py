import torchaudio
import torch

def preprocess_audio(file_path, target_sr=16000):
    """
    Ses dosyasını önişler: 16kHz'e yeniden örnekleme ve mono hale getirme.
    
    Args:
        file_path (str): Ses dosyasının yolu.
        target_sr (int): Hedef örnekleme frekansı (varsayılan: 16000).
        
    Returns:
        waveform (Tensor): Önişlenmiş ses dalgası.
        sample_rate (int): Yeni örnekleme frekansı.
    """
    # Ses dosyasını yükle
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Mono hale getir (stereo ise)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Yeniden örnekleme
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr

def segment_audio(waveform, sample_rate, segment_length=1.0):
    """
    Ses dosyasını belirtilen uzunlukta segmentlere böler.
    
    Args:
        waveform (Tensor): Ses dalgası.
        sample_rate (int): Örnekleme frekansı.
        segment_length (float): Segment uzunluğu (saniye cinsinden).
        
    Returns:
        segments (list): Segmentlerin listesi.
    """
    segment_samples = int(segment_length * sample_rate)
    segments = []
    
    for i in range(0, waveform.shape[1], segment_samples):
        segment = waveform[:, i:i + segment_samples]
        if segment.shape[1] == segment_samples:  # Tam segment
            segments.append(segment)
    
    return segments