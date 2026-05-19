import cv2
import time
import os
import numpy as np
from datetime import datetime

from picamera2 import Picamera2

from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

# =========================================================
# CONFIGURAÇÕES DE DIRETÓRIOS
# =========================================================
BASE_DIR = "capturas"

DIRS = {
    "frame": os.path.join(BASE_DIR, "frame"),
    "mask": os.path.join(BASE_DIR, "mask"),
    "overlay": os.path.join(BASE_DIR, "overlay"),
    "logs": os.path.join(BASE_DIR, "logs")
}

for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

SAVE_IMAGES = True
SAVE_EVERY_N_FRAMES = 1

# =========================================================
# INICIALIZAÇÃO DA CÂMERA
# =========================================================
print("[INFO] Inicializando Picamera2...")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)
print("[INFO] Câmera iniciada!")

# =========================================================
# LOOP PRINCIPAL
# =========================================================
prev = time.time()
frame_count = 0

try:
    while True:
        # 1. Captura
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_count += 1

        # 2. Definição do Filename (Garantido no topo do loop)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{frame_count:05d}_{timestamp}.jpg"

        # 3. Processamento e Inferência
        tensor = preprocess_frame(frame)

        start_inf = time.time()
        output = predict(tensor)
        inference_time = time.time() - start_inf

        # 4. Pós-Processamento com Ajuste de Confiança (Threshold)
        mask_raw = output.squeeze().cpu().numpy()
        print(f"[DEBUG MÁSCARA] Valor Máx: {mask_raw.max():.4f} | Valor Mín: {mask_raw.min():.4f}")
        
        # Cortando em 0.15 já que o modelo está respondendo por volta de 0.23
        THRESHOLD = 0.15
        mask = (mask_raw > THRESHOLD).astype(np.uint8) * 255

        mask_resized = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        # 5. Criação do Overlay
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = mask_resized

        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # 6. Métricas de Tempo e Cópia Visual
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        overlay_visualizacao = overlay.copy()
        
        cv2.putText(overlay_visualizacao, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_visualizacao, f"Inferencia: {inference_time*1000:.1f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Exibição
        cv2.imshow("Overlay", overlay_visualizacao)

        # 7. Salvamento Seguro (Todas as linhas usam a variável já criada acima)
        if SAVE_IMAGES and frame_count % SAVE_EVERY_N_FRAMES == 0:
            cv2.imwrite(os.path.join(DIRS["frame"], filename), frame)
            cv2.imwrite(os.path.join(DIRS["mask"], f"raw_{filename}"), mask)
            cv2.imwrite(os.path.join(DIRS["mask"], filename), mask_resized)
            cv2.imwrite(os.path.join(DIRS["overlay"], filename), overlay)

        # 8. Logs
        log_message = f"[FRAME {frame_count}] Inferência: {inference_time*1000:.1f} ms | FPS: {fps:.2f}"
        print(log_message)

        with open(os.path.join(DIRS["logs"], "runtime.log"), "a") as log_file:
            log_file.write(log_message + "\n")

        key = cv2.waitKey(1)
        if key == 27:
            break

except KeyboardInterrupt:
    print("\n[INFO] Encerrando...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados.")
