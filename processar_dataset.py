import cv2
import numpy as np
import os
from glob import glob

# =========================
# CONFIGURAÇÕES
# =========================

INPUT_DIR = "capturas/dataset"
OUTPUT_DIR = "capturas/processadas"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Faixa HSV ajustada para vegetação real em campo
LOWER_GREEN = np.array([38, 45, 35])
UPPER_GREEN = np.array([85, 255, 220])

# Usa metade inferior da imagem para focar no solo/corredor
ROI_START = 0.60

# Remove ruídos pequenos
MIN_CONTOUR_AREA = 2500

# Remove objetos verdes muito grandes, como camisa, sombra colorida ou grandes blocos
MAX_CONTOUR_AREA = 150000

# Tolerância do erro lateral para considerar "seguir em frente"
ERROR_TOLERANCE = 60

images = sorted(glob(os.path.join(INPUT_DIR, "*.jpg")))

print(f"Imagens encontradas: {len(images)}")

for image_path in images:
    frame = cv2.imread(image_path)

    if frame is None:
        print("Erro ao ler:", image_path)
        continue

    # Corrige câmera montada de cabeça para baixo
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    height, width, _ = frame.shape
    image_center = width // 2

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

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # =========================
    # 4. DETECÇÃO DE CONTORNOS
    # =========================

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        if area > MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 20 or h < 20:
            continue

        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.15 or aspect_ratio > 8.0:
            continue

        valid_contours.append(contour)

    debug = roi.copy()
    centers = []

    # =========================
    # 5. DESENHAR CONTORNOS VÁLIDOS
    # =========================

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        centers.append(cx)

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

    # =========================
    # 6. ESTIMATIVA DO CENTRO DO CORREDOR
    # =========================

    path_center = image_center
    command = "INDEFINIDO"

    left_line = None
    right_line = None

    if len(centers) == 0:
        command = "PARAR - SEM REFERENCIA"

    if len(centers) >= 2:
        left_points = [c for c in centers if c < image_center]
        right_points = [c for c in centers if c > image_center]

        if left_points and right_points:
            # Pega a vegetação mais próxima do centro em cada lado
            left_line = max(left_points)
            right_line = min(right_points)

            path_center = (left_line + right_line) // 2

    elif len(centers) == 1:
        only_line = centers[0]

        if only_line < image_center:
            command = "AJUSTAR DIREITA"
        else:
            command = "AJUSTAR ESQUERDA"

    error = path_center - image_center

    if command == "INDEFINIDO":
        if error < -ERROR_TOLERANCE:
            command = "VIRAR ESQUERDA"
        elif error > ERROR_TOLERANCE:
            command = "VIRAR DIREITA"
        else:
            command = "SEGUIR EM FRENTE"

    # =========================
    # 7. DESENHAR GUIAS VISUAIS
    # =========================

    # Centro da imagem: amarelo
    cv2.line(
        debug,
        (image_center, 0),
        (image_center, debug.shape[0]),
        (0, 255, 255),
        2
    )

    # Centro estimado do caminho: vermelho
    cv2.line(
        debug,
        (path_center, 0),
        (path_center, debug.shape[0]),
        (0, 0, 255),
        2
    )

    # Linhas laterais estimadas: azul
    if left_line is not None:
        cv2.line(
            debug,
            (left_line, 0),
            (left_line, debug.shape[0]),
            (255, 0, 0),
            2
        )

    if right_line is not None:
        cv2.line(
            debug,
            (right_line, 0),
            (right_line, debug.shape[0]),
            (255, 0, 0),
            2
        )

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

    cv2.putText(
        debug,
        f"Contornos: {len(valid_contours)}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # =========================
    # 8. SALVAR RESULTADOS
    # =========================

    base_name = os.path.basename(image_path)

    output_debug = os.path.join(
        OUTPUT_DIR,
        base_name.replace(".jpg", "_debug.jpg")
    )

    output_mask = os.path.join(
        OUTPUT_DIR,
        base_name.replace(".jpg", "_mask.jpg")
    )

    cv2.imwrite(output_debug, debug)
    cv2.imwrite(output_mask, mask_clean)

    print(
        f"{base_name} -> {command} | "
        f"erro={error} | "
        f"contornos={len(valid_contours)}"
    )

print("Processamento finalizado.")
print(f"Resultados em: {OUTPUT_DIR}")
