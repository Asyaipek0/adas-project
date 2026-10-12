"""
phone_detection.py
------------------
YOLOv8 (nano) modeliyle gerçek zamanlı telefon tespiti.

Önceden eğitilmiş yolov8n.pt modeli, COCO veri setindeki
80 nesne sınıfını tanır — bunlar arasında "cell phone" da vardır.
Ek model eğitimi gerekmez.

Performans optimizasyonları:
- Model her karede değil, belirli aralıklarla (skip_frames) çalışır
- YOLO'ya küçültülmüş görüntü (320px genişlik) verilir
- Tespit edilen kutu, orijinal görüntü boyutuna geri ölçeklenir
"""

from ultralytics import YOLO
import cv2

# COCO veri setindeki telefon sınıfının adı
PHONE_CLASS_NAME = "cell phone"

# Bu güven skorunun altındaki tespitler görmezden gelinir
CONFIDENCE_THRESHOLD = 0.5

# YOLO'ya verilecek görüntünün genişliği (px) — küçük = hızlı
INFERENCE_WIDTH = 320


class PhoneDetector:
    """
    YOLOv8 modelini yükler ve her güncelleme çağrısında
    görüntüde telefon olup olmadığını kontrol eder.
    """

    def __init__(self, skip_frames=2):
        # skip_frames=2: YOLO her 3 karede 1 çalışır (1 çalış, 2 atla)
        # Ara karelerde son tespit sonucu korunur, kutu ekrandan kaybolmaz
        self.model = YOLO("yolov8n.pt")
        self.phone_detected = False
        self.last_box = None
        self.skip_frames = skip_frames
        self.frame_counter = 0

    def update(self, frame):
        """
        Yeni kare için telefon tespiti yapar.
        skip_frames aralığına göre YOLO'yu atlar veya çalıştırır.
        Döndürür: phone_detected (bool)
        """
        self.frame_counter += 1

        # Belirlenen aralık dolmadıysa son sonucu koru, YOLO çalıştırma
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return self.phone_detected

        # Görüntüyü küçült, YOLO'ya ver
        h, w, _ = frame.shape
        scale = INFERENCE_WIDTH / w
        small_frame = cv2.resize(frame, (INFERENCE_WIDTH, int(h * scale)))

        results = self.model(small_frame, verbose=False)[0]

        self.phone_detected = False
        self.last_box = None

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            if class_name == PHONE_CLASS_NAME and confidence >= CONFIDENCE_THRESHOLD:
                self.phone_detected = True

                # Küçültülmüş görüntüdeki kutu koordinatlarını
                # orijinal boyuta geri ölçekle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.last_box = (
                    int(x1 / scale), int(y1 / scale),
                    int(x2 / scale), int(y2 / scale)
                )
                break  # İlk bulunan telefonla yetinilir

        return self.phone_detected

    def draw(self, frame):
        """Tespit edilen telefonun etrafına kutu ve etiket çizer."""
        if self.phone_detected and self.last_box:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "TELEFON TESPIT EDILDI", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)