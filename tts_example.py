import csv
from pydub import AudioSegment
import os
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
import logging
import warnings

# XTTS güvenli global ayar
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])

# Uyarıları bastır
warnings.filterwarnings("ignore")

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dizini ayarla
base_dir = r"C:\Users\yasin\Desktop\ai\sps-corpus-1.0-2025-09-05-tr"
audios_dir = os.path.join(base_dir, "audios")
corpus_file = os.path.join(base_dir, "ss-corpus-tr.tsv")
reported_file = os.path.join(base_dir, "ss-reported-audios-tr.tsv")

# Kullanıcıdan alınacak metin
user_text = "Değerli dostlar, Yahudi mahallesinde sık sık tuhaf bir topluluğa rastlanır."

# Hatalı dosyaları oku
reported_files = set()
if os.path.exists(reported_file):
    with open(reported_file, "r", encoding="utf-8") as f:
        for line in f:
            reported_files.add(line.strip())

# TTS metin ve dosya seç
selected_wav = None

if os.path.exists(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            file_name = row["audio_file"]
            if file_name not in reported_files:
                mp3_path = os.path.join(audios_dir, file_name)
                if os.path.exists(mp3_path):
                    try:
                        # İlk 10 saniyeyi kes
                        sound = AudioSegment.from_file(mp3_path)
                        short_sound = sound[:10000]
                        short_sound.export("speaker_wav.wav", format="wav")
                        selected_wav = "speaker_wav.wav"
                        print(f"Konuşmacı dosyası seçildi: {file_name}")
                        break
                    except Exception as e:
                        logger.error(f"Dosya işlenirken hata oluştu {file_name}: {e}")
                        continue

# XTTS ile sesi üret
if selected_wav and user_text:
    try:
        # Lisans sözleşmesini kabul et
        os.environ['COQUI_TOS_AGREED'] = '1'
        
        # Modeli yükle
        print("XTTS v2 modeli yükleniyor...")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
        print("Model yüklendi.")
        
        # Ses dosyası oluştur
        print("Ses dosyası oluşturuluyor...")
        tts.tts_to_file(
            text=user_text,
            speaker_wav=selected_wav,
            language="tr",
            file_path="deneme_xtts_v2.wav"
        )
        print("XTTS v2 ile ses dosyası oluşturuldu: deneme_xtts_v2.wav")
    except Exception as e:
        logger.error(f"XTTS ile ses üretilemedi: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Konuşmacı dosyası bulunamadı veya metin girilmedi.")