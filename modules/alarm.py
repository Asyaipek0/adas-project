"""
alarm.py
--------
Sesli alarm yönetimi.
Yorgunluk, dikkat dağınıklığı veya telefon tespitinde
winsound ile kısa bir bip sesi çalar.
Sesin üst üste binmemesi için cooldown (bekleme süresi) mekanizması içerir.
"""

import winsound
import threading


class AlarmManager:
    def __init__(self, cooldown=1.0):
        # cooldown: iki alarm arasındaki minimum bekleme süresi (saniye)
        self.alarm_sound_playing = False
        self.cooldown = cooldown

    def _play_alarm_sound(self):
        # 1500 Hz frekansta, 700 ms süren bir bip sesi
        winsound.Beep(1500, 700)

    def _reset_flag(self):
        # cooldown süresi dolunca bayrağı sıfırla, yeni alarm çalabilsin
        self.alarm_sound_playing = False

    def trigger(self):
        # Eğer şu an çalmıyorsa yeni bir alarm thread'i başlat
        if not self.alarm_sound_playing:
            self.alarm_sound_playing = True
            threading.Thread(target=self._play_alarm_sound, daemon=True).start()
            threading.Timer(self.cooldown, self._reset_flag).start()