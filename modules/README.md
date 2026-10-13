# 🚗 ADAS - Akıllı Sürücü İzleme Sistemi

Gerçek zamanlı bilgisayarlı görü teknikleri kullanarak sürücünün yorgunluğunu, dikkat dağınıklığını ve telefon kullanımını tespit eden bir Gelişmiş Sürücü Yardım Sistemi (Advanced Driver Assistance System) prototipi.

![Demo](demo.png)


## 📌 Proje Hakkında

Bu proje, otomotiv sektöründe giderek önem kazanan **mobilite ve otonom sürüş** teknolojilerine küçük ölçekli bir giriş olarak geliştirildi. Sürücünün anlık durumunu standart bir webcam üzerinden analiz ederek üç farklı tehlike türünü gerçek zamanlı olarak tespit eder:

- 😴 **Yorgunluk / Uyuklama** — Göz kapanma süresine dayalı tespit
- 👀 **Dikkat Dağınıklığı** — Baş pozisyonu (yana/yukarı/aşağı dönme) tespiti
- 📱 **Telefon Kullanımı** — Elde tutulan cep telefonu tespiti

Herhangi bir tehlike tespit edildiğinde sistem hem görsel (ekran uyarısı, renkli çerçeve) hem de sesli alarm üretir.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| **Python 3.12** | Ana programlama dili |
| **OpenCV** | Kamera erişimi, görüntü işleme, arayüz çizimi |
| **MediaPipe Face Mesh** | Yüz ve göz landmark (referans nokta) tespiti |
| **YOLOv8 (Ultralytics)** | Gerçek zamanlı nesne (telefon) tespiti |
| **NumPy** | Matematiksel hesaplamalar (EAR, head pose) |

## ⚙️ Nasıl Çalışır?

### 1. Yorgunluk Tespiti (EAR - Eye Aspect Ratio)
MediaPipe ile tespit edilen göz çevresi noktaları kullanılarak gözün açıklık oranı hesaplanır. Bu oran belirli bir eşiğin altına düşüp 2 saniyeden uzun süre öyle kalırsa, sistem yorgunluk alarmı verir.

### 2. Dikkat Dağınıklığı Tespiti (Head Pose Estimation)
Yüzdeki referans noktaları (burun, çene, göz ve ağız köşeleri) ile OpenCV'nin `solvePnP` algoritması kullanılarak kafanın yatay (yaw) ve dikey (pitch) açısı hesaplanır. Belirli bir açı eşiği aşılıp belirli bir süre korunursa dikkat dağınıklığı alarmı tetiklenir.

### 3. Telefon Tespiti (YOLOv8)
Önceden eğitilmiş YOLOv8n modeli, her karede (performans için optimize edilmiş aralıklarla) çalıştırılarak "cep telefonu" sınıfını tespit eder. Tespit anında görsel kutu ve alarm üretilir.

### 4. Canlı Durum Paneli
Tüm metrikler (EAR, Yaw/Pitch, FPS) ve tehlike durumları (Normal/Tehlike) tek bir dashboard panelinde gerçek zamanlı olarak gösterilir.

## 🚀 Kurulum

```bash
# 1. Projeyi klonla
git clone https://github.com/Asyaipek0/adas-project.git
cd adas-project

# 2. Sanal ortam oluştur ve aktif et
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# 4. Çalıştır
python adas_main.py
```

> İlk çalıştırmada YOLOv8n modeli (~6MB) otomatik olarak indirilecektir.

**Çıkmak için:** Kamera penceresi açıkken `q` tuşuna basın.

## 📁 Proje Yapısı

```
adas-project/
├── adas_main.py              # Ana program, kamera döngüsü ve dashboard
├── requirements.txt          # Bağımlılıklar
├── modules/
│   ├── drowsiness.py         # EAR hesaplama ve yorgunluk mantığı
│   ├── distraction.py        # Head pose hesaplama ve dikkat dağınıklığı mantığı
│   ├── phone_detection.py    # YOLOv8 ile telefon tespiti
│   └── alarm.py              # Sesli alarm yönetimi
└── yolov8n.pt                 # YOLO model ağırlıkları
```

## ⚡ Performans

Sistem, başlangıçta ~10 FPS ile çalışırken, aşağıdaki optimizasyonlarla **40+ FPS**'e çıkarılmıştır:
- YOLO modeli her karede değil, belirli aralıklarla çalıştırılır
- YOLO'ya verilen görüntü çözünürlüğü küçültülür (320px genişlik)
- FPS, ani dalgalanmaları önlemek için son 15 karenin ortalaması alınarak hesaplanır

## 🔮 Gelecek Geliştirmeler

- [ ] Sigara/yeme-içme gibi ek dikkatsizlik davranışlarının tespiti (özel model eğitimi)
- [ ] Trafik tabelası (DUR, hız sınırı vb.) tespiti
- [ ] Kafa pozisyonu hesaplamasında aşırı açılarda yaşanan kararsızlığın smoothing ile giderilmesi
- [ ] Mobil/gömülü sistemlere taşınabilirlik için model optimizasyonu (quantization)

## 👤 İletişim

ASYA İPEK SERİN 
GitHub: [@Asyaipek0](https://github.com/Asyaipek0)
LinkedIn: [Profilin](https://www.linkedin.com/in/asyaipekserin)

---

*Bu proje, otomotiv ve mobilite sektöründeki bilgisayarlı görü uygulamalarını keşfetmek amacıyla eğitim amaçlı geliştirilmiştir.*
