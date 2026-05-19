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
# 1. CONFIGURAÇÕES DE DIRETÓRIOS
# =========================================================
BASE_DIR = "capturas"

DIRS = {
    "frame": os.path.join(BASE_DIR, "frame"),
    "mask": os.path.join(BASE_DIR, "mask"),
    "overlay": os.path.join(BASE_DIR, "overlay"),
    "logs": os.path.join(BASE_DIR, "logs")
}

# Cria diretórios automaticamente caso não existam
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

SAVE_IMAGES = True
SAVE_EVERY_N_FRAMES = 1

# =========================================================
# 2. INICIALIZAÇÃO DA CÂMERA
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
# 3. LOOP PRINCIPAL
# =========================================================
prev = time.time()
frame_count = 0

try:
    while True:
        # -------------------------------------------------
        # CAPTURA
        # -------------------------------------------------
        frame_rgb = picam2.capture_array()
        
        # Converte para BGR para manter compatibilidade com o preprocess.py
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_count += 1

        # -------------------------------------------------
        # DEFINIÇÃO DO FILENAME (防 - À prova de falhas)
        # -------------------------------------------------
        # Criado logo no início para garantir que NENHUM cv2.imwrite quebre por NameError
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{frame_count:05d}_{timestamp}.jpg"

        # -------------------------------------------------
        # PRÉ-PROCESSAMENTO & INFERÊNCIA
        # -------------------------------------------------
        tensor = preprocess_frame(frame)

        start_inf = time.time()
        output = predict(tensor)
        inference_time = time.time() - start_inf

        # -------------------------------------------------
        # PÓS-PROCESSAMENTO (Ajustado para o sinal de 0.20)
        # -------------------------------------------------
        mask_raw = output.squeeze().cpu().numpy()
        
        # Monitoramento no terminal
        print(f"[DEBUG MÁSCARA] Valor Máx: {mask_raw.max():.4f} | Valor Mín: {mask_raw.min():.4f}")
        
        # Usando 0.15 como corte já que seu modelo bateu máx de 0.2099
        THRESHOLD = 0.15
        mask = (mask_raw > THRESHOLD).astype(np.uint8) * 255

        # Redimensiona mantendo os valores cravados (0 ou 255)
        mask_resized = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_resized = mask_resized.astype(np.uint8)

        # -------------------------------------------------
        # GERAÇÃO DO OVERLAY (MÁSCARA VERDE)
        # -------------------------------------------------
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = mask_resized  # Canal Verde

        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # -------------------------------------------------
        # CÁLCULO DE TELEMETRIA & CÓPIA DE VISUALIZAÇÃO
        # -------------------------------------------------
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        # Copia o overlay para desenhar o texto, mantendo o arquivo final limpo
        overlay_visualizacao = overlay.copy()
        
        cv2.putText(
            overlay_visualizacao,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            overlay_visualizacao,
            f"Inferencia: {inference_time*1000:.1f} ms",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Mostra na tela (Se houver monitor conectado ao Rasp)
        cv2.imshow("Overlay", overlay_visualizacao)

        # -------------------------------------------------
        # SALVAMENTO EM DISCO
        # -------------------------------------------------
        if SAVE_IMAGES and frame_count % SAVE_EVERY_N_FRAMES == 0:
            
            # 1. Salva o frame original limpo (sem textos por cima)
            cv2.imwrite(os.path.join(DIRS["frame"], filename), frame)

            # 2. Salva a máscara pequena original 256x256
            cv2.imwrite(os.path.join(DIRS["mask"], f"raw_{filename}"), mask)

            # 3. Salva a máscara redimensionada de 640x480 limpa
            cv2.imwrite(os.path.join(DIRS["mask"], filename), mask_resized)

            # 4. Salva o overlay limpo
            cv2.imwrite(os.path.join(DIRS["overlay"], filename), overlay)

        # -------------------------------------------------
        # SALVAMENTO DE LOGS
        # -------------------------------------------------
        log_message = (
            f"[FRAME {frame_count}] "
            f"Inferência: {inference_time*1000:.1f} ms | "
            f"FPS: {fps:.2f}"
        )
        print(log_message)

        with open(os.path.join(DIRS["logs"], "runtime.log"), "a") as log_file:
            log_file.write(log_message + "\n")

        # Tecla ESC para fechar manualmente
        key = cv2.waitKey(1)
        if key == 27:
            break

# =========================================================
# LIBERAÇÃO DE RECURSOS
# =========================================================
except KeyboardInterrupt:
    print("\n[INFO] Encerrando...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados.")
