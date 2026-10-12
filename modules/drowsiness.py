"""
drowsiness.py
-------------
EAR (Eye Aspect Ratio) algoritmasıyla sürücü yorgunluğu tespiti.

EAR Formülü:
    EAR = (dikey_1 + dikey_2) / (2 * yatay)

Göz açıkken EAR ~0.25-0.35 civarındadır.
Göz kapandıkça EAR 0'a yaklaşır.
EAR belirli bir eşiğin (0.21) altında kalıp
2 saniyeyi aşarsa yorgunluk alarmı tetiklenir.
"""

import numpy as np
import time
import cv2

# Sol göz çevresi için MediaPipe landmark indeksleri
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Sağ göz çevresi için MediaPipe landmark indeksleri
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Bu değerin altı "göz kapalı" sayılır — literatürde yaygın kabul gören eşik
EAR_THRESHOLD = 0.21

# Gözler bu kadar süre (saniye) kapalı kalırsa alarm tetiklenir
CLOSED_EYES_DURATION = 2.0


def _euclidean_distance(p1, p2):
    """İki 2D nokta arasındaki Öklid mesafesini hesaplar."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _calculate_ear(landmarks, eye_indices, w, h):
    """
    Verilen göz noktaları için EAR değerini hesaplar.
    6 nokta kullanılır: 2 yatay (köşeler), 4 dikey (üst-alt kenarlar).
    """
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = _euclidean_distance(p2, p6)
    vertical_2 = _euclidean_distance(p3, p5)
    horizontal = _euclidean_distance(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class DrowsinessDetector:
    """
    Her kare için EAR değerini hesaplar,
    süre takibi yaparak yorgunluk alarmını yönetir.
    """

    def __init__(self):
        self.eyes_closed_start = None  # Gözlerin kapandığı an
        self.alarm_active = False
        self.avg_ear = 0.0             # Son hesaplanan ortalama EAR değeri

    def update(self, landmarks, w, h):
        """
        Yeni bir kare için EAR hesaplar ve alarm durumunu günceller.
        Döndürür: alarm_active (bool)
        """
        left_ear = _calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = _calculate_ear(landmarks, RIGHT_EYE, w, h)
        self.avg_ear = (left_ear + right_ear) / 2.0

        if self.avg_ear < EAR_THRESHOLD:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = time.time()
            elif time.time() - self.eyes_closed_start >= CLOSED_EYES_DURATION:
                self.alarm_active = True
        else:
            # Göz açıldı, sayacı ve alarmı sıfırla
            self.eyes_closed_start = None
            self.alarm_active = False

        return self.alarm_active

    def draw(self, frame, w, h):
        """Yorgunluk alarmı aktifse ekrana uyarı yazar."""
        if self.alarm_active:
            cv2.putText(frame, "UYARI: YORGUNLUK TESPIT EDILDI!",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)