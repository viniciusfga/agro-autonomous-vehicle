from picamera2 import Picamera2
import cv2
import numpy as np
import time
from datetime import datetime

# =========================
# CONFIGURAÇÕES
# =========================

WIDTH = 1280
HEIGHT = 720

# Região inferior da imagem usada para navegação
ROI_START = 0.45

# Faixa inicial para detectar vegetação em HSV
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])

MIN_CONTOUR_AREA = 800

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)

frame_rgb = picam2.capture_array()
picam2.stop()

frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

height, width, _ = frame.shape

# =========================
# 1. RECORTE DA REGIÃO DE INTERESSE
# =========================

roi_y = int(height * ROI_START)
roi = frame[roi_y:height, 0:width]

# =========================
# 2. CONVERSÃO PARA HSV
# =========================

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# =========================
# 3. MÁSCARA DE VEGETAÇÃO
# =========================

mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

# Limpeza de ruído
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

# =========================
# 4. CONTORNOS
# =========================

contours, _ = cv2.findContours(
    mask_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

valid_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area >= MIN_CONTOUR_AREA:
        valid_contours.append(contour)

debug = roi.copy()

centers = []

for contour in valid_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2

    centers.append(cx)

    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

# =========================
# 5. ESTIMATIVA DO CENTRO DO CAMINHO
# =========================

image_center = width // 2
path_center = image_center
command = "INDEFINIDO"

if len(centers) >= 2:
    left_points = [c for c in centers if c < image_center]
    right_points = [c for c in centers if c > image_center]

    if left_points and right_points:
        left_line = max(left_points)
        right_line = min(right_points)

        path_center = (left_line + right_line) // 2

        cv2.line(debug, (left_line, 0), (left_line, debug.shape[0]), (255, 0, 0), 2)
        cv2.line(debug, (right_line, 0), (right_line, debug.shape[0]), (255, 0, 0), 2)

elif len(centers) == 1:
    # Caso detecte vegetação só de um lado
    only_line = centers[0]

    if only_line < image_center:
        command = "AJUSTAR DIREITA"
    else:
        command = "AJUSTAR ESQUERDA"

error = path_center - image_center

if command == "INDEFINIDO":
    if error < -80:
        command = "VIRAR ESQUERDA"
    elif error > 80:
        command = "VIRAR DIREITA"
    else:
        command = "SEGUIR EM FRENTE"

# =========================
# 6. DESENHO DO RESULTADO
# =========================

cv2.line(debug, (image_center, 0), (image_center, debug.shape[0]), (0, 255, 255), 2)
cv2.line(debug, (path_center, 0), (path_center, debug.shape[0]), (0, 0, 255), 2)

cv2.putText(
    debug,
    f"Comando: {command}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.putText(
    debug,
    f"Erro: {error}",
    (30, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

# =========================
# 7. SALVAR RESULTADOS
# =========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

cv2.imwrite(f"capturas/{timestamp}_original.jpg", frame)
cv2.imwrite(f"capturas/{timestamp}_roi.jpg", roi)
cv2.imwrite(f"capturas/{timestamp}_mask.jpg", mask_clean)
cv2.imwrite(f"capturas/{timestamp}_debug.jpg", debug)

print("Resultado salvo.")
print("Comando:", command)
print("Erro:", error)
print("Contornos detectados:", len(valid_contours))
