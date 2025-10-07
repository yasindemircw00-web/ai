import torch
import torchaudio
import os
from speechbrain.pretrained import EncoderClassifier
import numpy as np

def load_embedding_model():
    """
    Önceden eğitilmiş konuşmacı tanıma modelini yükler.
    
    Returns:
        model (EncoderClassifier): Eğitilmiş model.
    """
    save_dir = "pretrained_models/embedding"
    os.makedirs(save_dir, exist_ok=True)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=save_dir,
        run_opts={"device": "cpu"}
    )
    return model

def extract_embeddings(model, segments):
    """
    Segmentlerden embedding vektörlerini çıkarır.
    
    Args:
        model (EncoderClassifier): Eğitilmiş model.
        segments (list): Ses segmentleri.
        
    Returns:
        embeddings (list): Embedding vektörleri.
    """
    embeddings = []
    
    # Her segment için ayrı ayrı işle
    for i, segment in enumerate(segments):
        try:
            # Segment çok kısa ise atla
            if segment.numel() < 1000:  # Çok küçük segmentleri işleme
                embeddings.append(np.zeros(192))
                continue
                
            # Segmenti doğru formata getir
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)  # (samples) -> (1, samples)
            if segment.dim() == 2:
                if segment.shape[0] > 1:
                    # Stereo ise mono'ya çevir
                    segment = torch.mean(segment, dim=0, keepdim=True)
            
            # Segmentin uzunluğunu kontrol et ve sabit uzunlukta yap
            target_length = 16000  # 1 saniye @ 16kHz
            if segment.shape[1] < target_length:
                # Zero padding ile uzat
                padding_length = target_length - segment.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding_length), mode='constant', value=0)
            elif segment.shape[1] > target_length:
                # Kırp
                segment = segment[:, :target_length]
            
            # SpeechBrain için doğru tensör boyutlarını sağla
            if segment.dim() == 2:
                segment = segment.unsqueeze(0)  # (batch, channels, time)
            
            # Embedding çıkarma - hata durumunda daha iyi geri dönüş
            with torch.no_grad():
                try:
                    embedding = model.encode_batch(segment)
                    if embedding is not None and embedding.numel() > 0:
                        embeddings.append(embedding.squeeze().detach().cpu().numpy())
                    else:
                        embeddings.append(np.zeros(192))
                except Exception as e:
                    # SpeechBrain hata verirse, varyasyonlu zero embedding kullan
                    # Bu, tüm segmentlerin aynı embedding'e sahip olmasının önüne geçer
                    print(f"SpeechBrain error for segment {i}: {e}")
                    # Segment indeksine göre farklı zero embedding oluştur
                    zero_embedding = np.zeros(192)
                    # Küçük bir varyasyon ekle (tüm segmentlerin aynı olmasın)
                    zero_embedding[0] = (i % 10) * 0.01
                    embeddings.append(zero_embedding)
                    
        except Exception as e:
            print(f"Preprocessing error for segment {i}: {e}")
            # Herhangi bir hata durumunda varyasyonlu zero embedding
            zero_embedding = np.zeros(192)
            zero_embedding[0] = (i % 10) * 0.01
            embeddings.append(zero_embedding)
    
    return embeddings