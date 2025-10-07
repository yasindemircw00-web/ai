document.addEventListener('DOMContentLoaded', function() {
    // Menü butonlarını ve bölümleri al
    const menuButtons = document.querySelectorAll('.menu-btn');
    const sections = document.querySelectorAll('.section');
    
    // Değişkenler
    let recordedAudioBlob;
    let audioFileName;

    // DOM elementleri
    const textInput = document.getElementById('textInput');
    const audioOutput = document.getElementById('audioOutput');
    const convertToSpeechBtn = document.getElementById('convertToSpeech');
    const clearTextBtn = document.getElementById('clearText');
    const playAudioBtn = document.getElementById('playAudio'); // Yeni eklenen buton
    
    const audioFile = document.getElementById('audioFile');
    const recordedAudio = document.getElementById('recordedAudio');
    const convertToTextBtn = document.getElementById('convertToText');
    const textOutput = document.getElementById('textOutput');
    const copyTextBtn = document.getElementById('copyText');
    
    // Video to text elements
    const videoFile = document.getElementById('videoFile');
    const uploadedVideo = document.getElementById('uploadedVideo');
    const convertVideoToTextBtn = document.getElementById('convertVideoToText');
    const videoTextOutput = document.getElementById('videoTextOutput');
    const copyVideoTextBtn = document.getElementById('copyVideoText');
    
    // Durum ve ilerleme çubukları
    const ttsStatus = document.getElementById('tts-status');
    const sttStatus = document.getElementById('stt-status');
    const vttStatus = document.getElementById('vtt-status');
    const ttsProgressFill = document.getElementById('tts-progress-fill');
    const sttProgressFill = document.getElementById('stt-progress-fill');
    const vttProgressFill = document.getElementById('vtt-progress-fill');

    // Audio elementi için event listener'lar
    audioOutput.addEventListener('loadstart', function() {
        console.log("Audio load started");
    });
    
    audioOutput.addEventListener('loadeddata', function() {
        console.log("Audio data loaded");
    });
    
    audioOutput.addEventListener('canplay', function() {
        console.log("Audio can play");
    });
    
    audioOutput.addEventListener('error', function(e) {
        console.log("Audio error:", e);
        console.log("Error code:", audioOutput.error.code);
        console.log("Error message:", audioOutput.error.message);
    });

    // Menü navigasyonu
    menuButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            
            // Aktif menü butonunu güncelle
            menuButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Aktif bölümü güncelle
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });
        });
    });

    // Metin → Ses butonları
    convertToSpeechBtn.addEventListener('click', convertTextToSpeech);
    clearTextBtn.addEventListener('click', clearText);
    playAudioBtn.addEventListener('click', playAudio); // Yeni eklenen event listener

    // Ses dosyası yükleme
    audioFile.addEventListener('change', handleAudioFileChange);

    // Ses → Metin çevirme
    convertToTextBtn.addEventListener('click', transcribeAudio);

    // Metni kopyala
    copyTextBtn.addEventListener('click', copyText);

    // Video dosyası yükleme
    videoFile.addEventListener('change', handleVideoFileChange);

    // Video → Metin çevirme
    convertVideoToTextBtn.addEventListener('click', transcribeVideo);

    // Video metnini kopyala
    copyVideoTextBtn.addEventListener('click', copyVideoText);

    // Metin seslendirme fonksiyonu
    function convertTextToSpeech() {
        const text = textInput.value.trim();
        
        if (!text) {
            ttsStatus.textContent = 'Lütfen dönüştürmek için metin girin.';
            ttsStatus.style.color = 'red';
            return;
        }

        // Butonu devre dışı bırak
        convertToSpeechBtn.disabled = true;
        ttsStatus.textContent = 'İşlem başlatılıyor...';
        ttsStatus.style.color = 'orange';
        ttsProgressFill.style.width = '0%';

        // Önceki ses dosyasını temizle
        audioOutput.style.display = 'none';

        // Python Flask API'sini çağır
        callTextToSpeechAPI(text);
    }

    // Metni temizle
    function clearText() {
        textInput.value = '';
        audioOutput.style.display = 'none';
        playAudioBtn.style.display = 'none'; // Play butonunu gizle
        ttsStatus.textContent = 'Metni yazın ve "Sese Çevir" butonuna tıklayın';
        ttsStatus.style.color = '#666';
    }

    // Ses dosyasını oynat
    function playAudio() {
        // Butonu devre dışı bırak ve metni değiştir
        playAudioBtn.disabled = true;
        const originalText = playAudioBtn.innerHTML;
        playAudioBtn.innerHTML = '<span>▶️</span><span>Oynatılıyor...</span>';
        
        console.log("Attempting to play audio:", audioOutput.src);
        console.log("Audio element:", audioOutput);
        console.log("Audio network state:", audioOutput.networkState);
        console.log("Audio ready state:", audioOutput.readyState);
        
        if (audioOutput.src) {
            // Önce ses seviyesini kontrol edelim
            audioOutput.volume = 1.0;
            
            // Ses dosyasının yüklenmesini bekle
            if (audioOutput.readyState === 0) {
                console.log("Audio not loaded, waiting for load...");
                audioOutput.load();
            }
            
            // Ses dosyasını oynatmaya çalış
            var playPromise = audioOutput.play();
            
            if (playPromise !== undefined) {
                playPromise.then(function() {
                    // Oynatma başarılı
                    console.log("Audio playback started successfully");
                    // Butonu tekrar etkinleştir
                    playAudioBtn.disabled = false;
                    playAudioBtn.innerHTML = originalText;
                }).catch(function(error) {
                    // Oynatma başarısız
                    console.log("Audio playback failed:", error);
                    ttsStatus.textContent = 'Ses oynatılamadı: ' + error.message;
                    ttsStatus.style.color = 'red';
                    // Butonu tekrar etkinleştir
                    playAudioBtn.disabled = false;
                    playAudioBtn.innerHTML = originalText;
                });
            } else {
                // Promise desteklenmiyorsa
                console.log("Play promise not supported");
                playAudioBtn.disabled = false;
                playAudioBtn.innerHTML = originalText;
            }
        } else {
            console.log("No audio source available");
            ttsStatus.textContent = 'Ses dosyası bulunamadı';
            ttsStatus.style.color = 'red';
            // Butonu tekrar etkinleştir
            playAudioBtn.disabled = false;
            playAudioBtn.innerHTML = originalText;
        }
    }

    // Ses dosyası yükleme işlemi
    function handleAudioFileChange(e) {
        const file = e.target.files[0];
        if (file) {
            const audioUrl = URL.createObjectURL(file);
            recordedAudio.src = audioUrl;
            recordedAudio.style.display = 'block';
            recordedAudioBlob = file;
            sttStatus.textContent = 'Ses dosyası yüklendi. "Metne Çevir" butonuna tıklayın.';
            sttStatus.style.color = '#666';
        }
    }

    // Video dosyası yükleme işlemi
    function handleVideoFileChange(e) {
        const file = e.target.files[0];
        if (file) {
            const videoUrl = URL.createObjectURL(file);
            uploadedVideo.src = videoUrl;
            uploadedVideo.style.display = 'block';
            recordedAudioBlob = file;
            vttStatus.textContent = 'Video dosyası yüklendi. "Metne Çevir" butonuna tıklayın.';
            vttStatus.style.color = '#666';
        }
    }

    // Ses dosyasını metne dönüştürme
    function transcribeAudio() {
        if (!recordedAudioBlob) {
            sttStatus.textContent = 'Lütfen önce ses kaydı yapın veya ses dosyası yükleyin.';
            sttStatus.style.color = 'red';
            return;
        }

        // Butonu devre dışı bırak
        convertToTextBtn.disabled = true;
        sttStatus.textContent = 'Ses dosyası işleniyor...';
        sttStatus.style.color = 'orange';
        sttProgressFill.style.width = '0%';

        // Önceki çıktıları temizle
        textOutput.value = '';

        // Python Flask API'sini çağır
        callSpeechToTextAPI(recordedAudioBlob);
    }

    // Video dosyasını metne dönüştürme
    function transcribeVideo() {
        if (!recordedAudioBlob) {
            vttStatus.textContent = 'Lütfen önce video dosyası yükleyin.';
            vttStatus.style.color = 'red';
            return;
        }

        // Butonu devre dışı bırak
        convertVideoToTextBtn.disabled = true;
        vttStatus.textContent = 'Video dosyası işleniyor...';
        vttStatus.style.color = 'orange';
        vttProgressFill.style.width = '0%';

        // Önceki çıktıları temizle
        videoTextOutput.value = '';

        // Python Flask API'sini çağır (video için özel endpoint)
        callVideoToTextAPI(recordedAudioBlob);
    }

    // Metni kopyala
    function copyText() {
        if (textOutput.value.trim()) {
            textOutput.select();
            document.execCommand('copy');
            const originalText = copyTextBtn.innerHTML;
            copyTextBtn.innerHTML = '<span>✓</span><span>Kopyalandı!</span>';
            setTimeout(() => {
                copyTextBtn.innerHTML = originalText;
            }, 2000);
        } else {
            alert('Kopyalanacak metin yok.');
        }
    }

    // Video metnini kopyala
    function copyVideoText() {
        if (videoTextOutput.value.trim()) {
            videoTextOutput.select();
            document.execCommand('copy');
            const originalText = copyVideoTextBtn.innerHTML;
            copyVideoTextBtn.innerHTML = '<span>✓</span><span>Kopyalandı!</span>';
            setTimeout(() => {
                copyVideoTextBtn.innerHTML = originalText;
            }, 2000);
        } else {
            alert('Kopyalanacak metin yok.');
        }
    }

    // Python Flask API'sini çağır (metin -> ses)
    function callTextToSpeechAPI(text) {
        ttsStatus.textContent = 'Metin işleniyor...';
        
        // İlerleme çubuğunu animasyonla doldur
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            ttsProgressFill.style.width = progress + '%';
        }, 100);
        
        // API çağrısı
        fetch('/convert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({text: text})
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            ttsProgressFill.style.width = '100%';
            
            if (data.success) {
                finishConversion(data.filename);
            } else {
                ttsStatus.textContent = 'Hata: ' + data.error;
                ttsStatus.style.color = 'red';
                convertToSpeechBtn.disabled = false;
            }
        })
        .catch(error => {
            clearInterval(interval);
            ttsStatus.textContent = 'Bağlantı hatası: ' + error;
            ttsStatus.style.color = 'red';
            convertToSpeechBtn.disabled = false;
        });
    }

    // Python Flask API'sini çağır (ses -> metin)
    function callSpeechToTextAPI(file) {
        sttStatus.textContent = 'Ses metne dönüştürülüyor...';
        vttStatus.textContent = 'Video metne dönüştürülüyor...';
        
        // İlerleme çubuğunu animasyonla doldur
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            sttProgressFill.style.width = progress + '%';
            vttProgressFill.style.width = progress + '%';
        }, 100);
        
        // FormData oluştur
        const formData = new FormData();
        formData.append('audio', file);
        
        // API çağrısı
        fetch('/transcribe', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            sttProgressFill.style.width = '100%';
            vttProgressFill.style.width = '100%';
            
            if (data.success) {
                finishTranscription(data.text, data.language, data.detailed_transcription);
                // Video metin çıktısını da güncelle
                videoTextOutput.value = data.text;
            } else {
                sttStatus.textContent = 'Hata: ' + data.error;
                sttStatus.style.color = 'red';
                vttStatus.textContent = 'Hata: ' + data.error;
                vttStatus.style.color = 'red';
                convertToTextBtn.disabled = false;
                convertVideoToTextBtn.disabled = false;
            }
        })
        .catch(error => {
            clearInterval(interval);
            sttStatus.textContent = 'Bağlantı hatası: ' + error;
            sttStatus.style.color = 'red';
            vttStatus.textContent = 'Bağlantı hatası: ' + error;
            vttStatus.style.color = 'red';
            convertToTextBtn.disabled = false;
            convertVideoToTextBtn.disabled = false;
        });
    }

    // Python Flask API'sini çağır (video -> metin)
    function callVideoToTextAPI(file) {
        vttStatus.textContent = 'Video metne dönüştürülüyor...';
        
        // İlerleme çubuğunu animasyonla doldur
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            vttProgressFill.style.width = progress + '%';
        }, 100);
        
        // FormData oluştur
        const formData = new FormData();
        formData.append('audio', file);
        
        // API çağrısı (video için özel endpoint)
        fetch('/transcribe_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            vttProgressFill.style.width = '100%';
            
            if (data.success) {
                vttStatus.textContent = 'Metne dönüştürme tamamlandı!';
                vttStatus.style.color = 'green';
                convertVideoToTextBtn.disabled = false;
                
                // Sonucu göster
                videoTextOutput.value = data.text;
                
                // Detaylı transkripsiyonu göster
                if (data.detailed_transcription && data.detailed_transcription.length > 0) {
                    showDetailedTranscription(data.detailed_transcription);
                }
            } else {
                vttStatus.textContent = 'Hata: ' + data.error;
                vttStatus.style.color = 'red';
                convertVideoToTextBtn.disabled = false;
            }
        })
        .catch(error => {
            clearInterval(interval);
            vttStatus.textContent = 'Bağlantı hatası: ' + error;
            vttStatus.style.color = 'red';
            convertVideoToTextBtn.disabled = false;
        });
    }

    // Detaylı transkripsiyonu göster
    function showDetailedTranscription(detailed_transcription) {
        // Detaylı transkripsiyon bilgilerini konsola yaz
        console.log('Detaylı transkripsiyon:', detailed_transcription);
        
        // İsterseniz bu bilgileri bir tablo veya başka bir formatta da gösterebilirsiniz
        // Örneğin, konuşmacıların ne zaman konuştuğunu gösteren bir zaman çizelgesi
        let schedule = "Konuşma Zaman Çizelgesi:\n";
        const speakers = new Set();
        
        detailed_transcription.forEach(segment => {
            speakers.add(segment.speaker);
            // Başlangıç ve bitiş zamanlarını MM:SS formatına çevir
            const startMinutes = Math.floor(segment.start / 60);
            const startSeconds = Math.floor(segment.start % 60);
            const endMinutes = Math.floor(segment.end / 60);
            const endSeconds = Math.floor(segment.end % 60);
            
            // Zaman damgası formatı: [MM:SS-MM:SS] SpeakerX: metin
            schedule += `[${startMinutes.toString().padStart(2, '0')}:${startSeconds.toString().padStart(2, '0')}-${endMinutes.toString().padStart(2, '0')}:${endSeconds.toString().padStart(2, '0')}] ${segment.speaker}: ${segment.text}\n`;
        });
        
        console.log(schedule);
        
        // Konuşmacı sayısı bilgisini göster
        const speakerCount = speakers.size;
        vttStatus.textContent += ` (${speakerCount} farklı konuşmacı tespit edildi)`;
    }

    // Dönüştürme tamamlandığında (ses -> metin)
    function finishTranscription(text, language, detailed_transcription) {
        sttStatus.textContent = 'Metne dönüştürme tamamlandı!';
        sttStatus.style.color = 'green';
        vttStatus.textContent = 'Metne dönüştürme tamamlandı!';
        vttStatus.style.color = 'green';
        convertToTextBtn.disabled = false;
        convertVideoToTextBtn.disabled = false;
        
        // Sonucu göster
        textOutput.value = text;
        
        // Detaylı transkripsiyon varsa, video bölümüne daha fazla bilgi ekleyin
        if (detailed_transcription && detailed_transcription.length > 0) {
            // Konuşmacı sayısı bilgisini göster
            const speakers = new Set();
            detailed_transcription.forEach(segment => {
                speakers.add(segment.speaker);
            });
            
            const speakerCount = speakers.size;
            sttStatus.textContent += ` (${speakerCount} farklı konuşmacı tespit edildi)`;
            
            // Detaylı transkripsiyonu göster
            showDetailedTranscription(detailed_transcription);
        }
    }

    // Dönüştürme tamamlandığında (metin -> ses)
    function finishConversion(filename) {
        ttsStatus.textContent = 'Seslendirme tamamlandı!';
        ttsStatus.style.color = 'green';
        convertToSpeechBtn.disabled = false;
        
        // Zaman damgası ekleyerek önbellekten kaçın
        const timestamp = new Date().getTime();
        const audioSrc = '/audio/' + filename + '?t=' + timestamp;
        
        // Ses dosyasını ayarla
        console.log("Setting audio source to:", audioSrc);
        audioOutput.src = audioSrc;
        audioOutput.style.display = 'block';
        playAudioBtn.style.display = 'block'; // Play butonunu göster
        audioOutput.load();
        
        // Ses dosyası adını sakla
        audioFileName = filename;
    }

    // Örnek metin ekle
    textInput.value = "Değerli dostlar, Yahudi mahallesinde sık sık tuhaf bir topluluğa rastlanır.";
});