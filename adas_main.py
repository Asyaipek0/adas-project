"""
adas_main.py
------------
ADAS (Advanced Driver Assistance System) — Ana Program

Webcam görüntüsünü her kare için işleyerek üç tehlike türünü
gerçek zamanlı olarak tespit eder:
    1. Yorgunluk/uyuklama  → drowsiness.py (EAR algoritması)
    2. Dikkat dağınıklığı  → distraction.py (Head Pose Estimation)
    3. Telefon kullanımı   → phone_detection.py (YOLOv8)

Herhangi bir tehlike tespit edildiğinde alarm.py üzerinden
sesli alarm tetiklenir, ekranda uyarı banner'ı ve renkli çerçeve gösterilir.
"""

import cv2
import mediapipe as mp
import time

from modules.alarm import AlarmManager
from modules.drowsiness import DrowsinessDetector
from modules.distraction import DistractionDetector
from modules.phone_detection import PhoneDetector

mp_face_mesh = mp.solutions.face_mesh

# Göz noktaları — draw için main'de kullanılıyor
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Dashboard panel boyutları
PANEL_WIDTH = 300
PANEL_HEIGHT = 220
PANEL_MARGIN = 15


def draw_dashboard(frame, ear, yaw, pitch, fps, drowsy, distracted, phone):
    """
    Sol üst köşeye yarı saydam bir bilgi paneli çizer.
    EAR, Yaw/Pitch, FPS değerlerini ve her modülün
    anlık durumunu (NORMAL / TEHLİKE) gösterir.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (PANEL_MARGIN, PANEL_MARGIN),
                  (PANEL_MARGIN + PANEL_WIDTH, PANEL_MARGIN + PANEL_HEIGHT),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x = PANEL_MARGIN + 15
    y = PANEL_MARGIN + 30

    cv2.putText(frame, "ADAS - DURUM PANELI", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 30

    cv2.putText(frame, f"EAR: {ear:.3f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 28

    yaw_text = f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}" if yaw is not None else "Yaw: --  Pitch: --"
    cv2.putText(frame, yaw_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 28

    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 35

    def status_row(label, is_alert):
        nonlocal y
        color = (0, 0, 255) if is_alert else (0, 200, 0)
        status_text = "TEHLIKE" if is_alert else "NORMAL"
        cv2.circle(frame, (x + 8, y - 6), 7, color, -1)
        cv2.putText(frame, f"{label}: {status_text}", (x + 25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 30

    status_row("Yorgunluk", drowsy)
    status_row("Dikkat", distracted)
    status_row("Telefon", phone)


def draw_alert_banner(frame, w, drowsy, distracted, phone):
    """
    Tehlike varsa ekranın üst kısmına ortalanmış uyarı banner'ı çizer.
    Aynı anda birden fazla tehlike varsa hepsi üst üste listelenir.
    Ekranın etrafına da renkli bir çerçeve eklenir.
    """
    messages = []
    if drowsy:
        messages.append(("YORGUNLUK TESPIT EDILDI!", (0, 0, 255)))
    if distracted:
        messages.append(("DIKKATINI YOLA VER!", (0, 140, 255)))
    if phone:
        messages.append(("TELEFONU BIRAK!", (0, 0, 255)))

    if not messages:
        return

    banner_height = 50 * len(messages) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 35
    for text, color in messages:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
        x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        y += 50

    border_color = (0, 0, 255) if drowsy or phone else (0, 140, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, frame.shape[0] - 1), border_color, 8)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("HATA: Kamera açılamadı.")
        return

    print("Kamera açıldı. Çıkmak için 'q' tuşuna bas.")

    # Modülleri başlat
    alarm_manager = AlarmManager()
    drowsiness_detector = DrowsinessDetector()
    distraction_detector = DistractionDetector()
    phone_detector = PhoneDetector()

    # FPS hesaplama için değişkenler
    prev_time = time.time()
    fps = 0
    fps_history = []  # Ani dalgalanmaları önlemek için son 15 karenin ortalaması

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()

            # FPS hesapla (son 15 karenin ortalaması)
            current_time = time.time()
            instant_fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            fps_history.append(instant_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)

            if not ret:
                print("HATA: Görüntü alınamadı.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            h, w, _ = frame.shape

            ear = 0.0
            yaw, pitch = None, None
            drowsy, distracted = False, False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Göz noktalarını yeşil nokta olarak çiz
                    for idx in LEFT_EYE + RIGHT_EYE:
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Yorgunluk tespiti
                    drowsy = drowsiness_detector.update(landmarks, w, h)
                    ear = drowsiness_detector.avg_ear

                    # Dikkat dağınıklığı tespiti
                    distracted = distraction_detector.update(landmarks, w, h)
                    yaw, pitch = distraction_detector.yaw, distraction_detector.pitch

            # Telefon tespiti (yüz olmasa bile her karede çalışır)
            phone_detected = phone_detector.update(frame)
            phone_detector.draw(frame)

            # Herhangi bir tehlike varsa sesli alarm tetikle
            if drowsy or distracted or phone_detected:
                alarm_manager.trigger()

            # Dashboard ve uyarı banner'ını çiz
            draw_dashboard(frame, ear, yaw if yaw else 0, pitch if pitch else 0,
                           fps, drowsy, distracted, phone_detected)
            draw_alert_banner(frame, w, drowsy, distracted, phone_detected)

            cv2.imshow("ADAS - Surucu Izleme Sistemi", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()