@echo off
echo Sanal ortam oluşturuluyor...
python -m venv .venv

echo Sanal ortam etkinleştiriliyor...
call .venv\Scripts\activate.bat

echo Gerekli paketler yükleniyor...
pip install -r requirements.txt

echo Kurulum tamamlandı.
echo Uygulamayı başlatmak için aşağıdaki komutu çalıştırın:
echo .venv\Scripts\activate.bat && python app.py

pause