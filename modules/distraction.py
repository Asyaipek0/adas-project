"""
distraction.py
--------------
Baş pozisyonu tahmini (Head Pose Estimation) ile dikkat dağınıklığı tespiti.

Yüzdeki 6 referans noktası (burun, çene, göz ve ağız köşeleri),
OpenCV'nin solvePnP algoritmasıyla 3D bir yüz modeline eşleştirilerek
kafanın yatay (yaw) ve dikey (pitch) açısı hesaplanır.

Yaw veya pitch belirli eşikleri aşıp 1.5 saniye boyunca korunursa
dikkat dağınıklığı alarmı tetiklenir.
"""

import numpy as np
import time
import cv2

# Yatay dönüş eşiği (derece) — bu açıdan fazla sağa/sola dönerse dikkat dağınık say
YAW_THRESHOLD = 20.0

# Dikey dönüş eşiği (derece) — bu açıdan fazla yukarı/aşağı bakarsa dikkat dağınık say
PITCH_THRESHOLD = 18.0

# Bu kadar süre (saniye) boyunca eşik aşılırsa alarm tetiklenir
DISTRACTION_DURATION = 1.5

# Head pose hesaplamasında kullanılan yüz landmark indeksleri
# Sırasıyla: burun ucu, çene, sol göz dış köşe, sağ göz dış köşe,
# sol ağız köşe, sağ ağız köşe
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Yukarıdaki noktaların gerçek dünyadaki yaklaşık 3D karşılıkları (mm cinsinden)
# Standart bir insan yüzü modeline dayanır
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Burun ucu
    (0.0, -330.0, -65.0),     # Çene
    (-225.0, 170.0, -135.0),  # Sol göz dış köşe
    (225.0, 170.0, -135.0),   # Sağ göz dış köşe
    (-150.0, -150.0, -125.0), # Sol ağız köşe
    (150.0, -150.0, -125.0)   # Sağ ağız köşe
], dtype=np.float64)


def _calculate_head_pose(landmarks, w, h):
    """
    solvePnP algoritmasıyla kafanın yaw ve pitch açılarını hesaplar.
    Kamera matrisi, görüntü boyutlarından yaklaşık olarak türetilir.
    Döndürür: (yaw, pitch) veya tespit başarısızsa (None, None)
    """
    # 2D görüntü noktalarını landmark koordinatlarından çıkar
    image_points = np.array([
        (landmarks[idx].x * w, landmarks[idx].y * h)
        for idx in HEAD_POSE_LANDMARKS
    ], dtype=np.float64)

    # Basit pinhole kamera modeli (lens distorsiyonu yok varsayılır)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    # Rodrigues dönüşümü: rotasyon vektörünü rotasyon matrisine çevirir
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

    return yaw, pitch


class DistractionDetector:
    """
    Her kare için baş açısını hesaplar,
    süre takibi yaparak dikkat dağınıklığı alarmını yönetir.
    """

    def __init__(self):
        self.distraction_start = None  # Dikkat dağınıklığının başladığı an
        self.distraction_active = False
        self.yaw = None
        self.pitch = None

    def update(self, landmarks, w, h):
        """
        Yeni bir kare için baş açısını hesaplar ve alarm durumunu günceller.
        Döndürür: distraction_active (bool)
        """
        self.yaw, self.pitch = _calculate_head_pose(landmarks, w, h)

        if self.yaw is None:
            return False

        is_distracted = (abs(self.yaw) > YAW_THRESHOLD or
                         abs(self.pitch) > PITCH_THRESHOLD)

        if is_distracted:
            if self.distraction_start is None:
                self.distraction_start = time.time()
            elif time.time() - self.distraction_start >= DISTRACTION_DURATION:
                self.distraction_active = True
        else:
            # Kafa normale döndü, sayacı ve alarmı sıfırla
            self.distraction_start = None
            self.distraction_active = False

        return self.distraction_active

    def draw(self, frame, w, h):
        """Dikkat dağınıklığı alarmı aktifse ekrana uyarı yazar."""
        if self.yaw is not None:
            cv2.putText(frame, f"Yaw: {self.yaw:.1f}  Pitch: {self.pitch:.1f}",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

        if self.distraction_active:
            cv2.putText(frame, "UYARI: DIKKATINI YOLA VER!",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 140, 255), 3)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 140, 255), 10)