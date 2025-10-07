import sys
import os
from TTS.api import TTS
import torch
import uuid

# XTTS güvenli global ayarlar
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

def prepare_speaker_wav():
    """Konuşmacı için WAV dosyası hazırlar"""
    base_dir = r"C:\Users\yasin\Desktop\ai\sps-corpus-1.0-2025-09-05-tr"
    audios_dir = os.path.join(base_dir, "audios")
    corpus_file = os.path.join(base_dir, "ss-corpus-tr.tsv")
    reported_file = os.path.join(base_dir, "ss-reported-audios-tr.tsv")
    
    # Speaker WAV dizininin var olduğundan emin ol
    speaker_wav_dir = r"C:\Users\yasin\Desktop\ai\sesclon"
    if not os.path.exists(speaker_wav_dir):
        os.makedirs(speaker_wav_dir)
    
    speaker_wav_path = os.path.join(speaker_wav_dir, "speaker_wav.wav")
    
    # Hatalı dosyaları oku
    reported_files = set()
    if os.path.exists(reported_file):
        with open(reported_file, "r", encoding="utf-8") as f:
            for line in f:
                reported_files.add(line.strip())
    
    # TTS metin ve dosya seç
    if os.path.exists(corpus_file):
        import csv
        with open(corpus_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                file_name = row["audio_file"]
                if file_name not in reported_files:
                    mp3_path = os.path.join(audios_dir, file_name)
                    if os.path.exists(mp3_path):
                        try:
                            # İlk 10 saniyeyi kes
                            from pydub import AudioSegment
                            sound = AudioSegment.from_file(mp3_path)
                            short_sound = sound[:10000]
                            short_sound.export(speaker_wav_path, format="wav")
                            print(f"Konuşmacı dosyası seçildi: {file_name}")
                            return speaker_wav_path
                        except Exception as e:
                            print(f"Dosya işlenirken hata oluştu {file_name}: {e}")
                            continue
    return None

def generate_tts(text, output_path):
    """TTS ses dosyası oluşturur"""
    try:
        # CUDA GPU kullanılabilirliğini kontrol et
        if torch.cuda.is_available():
            gpu = True
            print("CUDA GPU kullanılabilir, TTS modeli GPU üzerinde çalışacak.")
        else:
            gpu = False
            print("CUDA GPU kullanılamıyor, TTS modeli CPU üzerinde çalışacak.")
        
        # TTS modelini yükle
        tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=gpu)
        print("TTS modeli yüklendi.")
        
        # Konuşmacı dosyasını hazırla
        selected_wav = prepare_speaker_wav()
        if not selected_wav:
            print("Konuşmacı dosyası hazırlanamadı.")
            return False
        
        # Lisans sözleşmesini kabul et
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # Ses dosyası oluştur
        tts_model.tts_to_file(
            text=text,
            speaker_wav=selected_wav,
            language="tr",
            file_path=output_path
        )
        
        print(f"Ses dosyası oluşturuldu: {output_path}")
        return True
    except Exception as e:
        print(f"TTS sırasında hata oluştu: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Kullanım: python tts_generate.py <metin> <çıkış_dosyası>")
        sys.exit(1)
    
    text = sys.argv[1]
    output_path = sys.argv[2]
    
    success = generate_tts(text, output_path)
    if success:
        print("TTS işlemi başarıyla tamamlandı.")
        sys.exit(0)
    else:
        print("TTS işlemi başarısız oldu.")
        sys.exit(1)