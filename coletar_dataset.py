from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import os

SAVE_DIR = "capturas/dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# False = salva imagem bruta da câmera.
# O processar_dataset.py já faz a rotação 180°.
ROTATE_180 = False

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Coletando imagens. Pressione Ctrl+C para parar.")

try:
    while True:
        frame_rgb = picam2.capture_array()

        if ROTATE_180:
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{SAVE_DIR}/{timestamp}.jpg"

        cv2.imwrite(filename, frame_bgr)
        print("Salvo:", filename)

        time.sleep(1)

except KeyboardInterrupt:
    print("Coleta encerrada.")

finally:
    picam2.stop()
