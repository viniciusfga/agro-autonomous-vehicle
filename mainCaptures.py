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
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = "capturas"

DIRS = {
    "frame": os.path.join(BASE_DIR, "frame"),
    "mask": os.path.join(BASE_DIR, "mask"),
    "overlay": os.path.join(BASE_DIR, "overlay"),
    "logs": os.path.join(BASE_DIR, "logs")
}

# Cria diretórios automaticamente
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

        # -------------------------------------------------
        # CAPTURA
        # -------------------------------------------------
        frame_rgb = picam2.capture_array()

        # Converte para BGR pois o preprocess_frame espera BGR para converter internamente
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_count += 1

        # -------------------------------------------------
        # NOME PADRONIZADO (Corrigido para evitar conflitos)
        # -------------------------------------------------
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
        # PÓS-PROCESSAMENTO (Correção do Resize)
        # -------------------------------------------------
        mask_raw = output.squeeze().cpu().numpy()

        THRESHOLD = 0.15
        mask = (mask_raw > THRESHOLD).astype(np.uint8) * 255

        # Redimensiona mantendo os valores binários cravados (0 ou 255)
        mask_resized = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_resized = mask_resized.astype(np.uint8)

        # -------------------------------------------------
        # OVERLAY PURA
        # -------------------------------------------------
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = mask_resized  # Canal Verde

        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # -------------------------------------------------
        # CÁLCULO DE FPS & TELEMETRIA VISUAL
        # -------------------------------------------------
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        # Criamos uma cópia do overlay apenas para a janela de exibição.
        # Isso impede que o texto do FPS seja gravado no arquivo final das imagens de dataset.
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

        # -------------------------------------------------
        # VISUALIZAÇÃO (Apenas se houver interface gráfica aberta)
        # -------------------------------------------------
        # Nota: Se rodar via SSH puro sem X11 encaminhado, o imshow pode dar erro. 
        # Caso vá rodar no campo sem tela, você pode comentar a linha abaixo.
        cv2.imshow("Overlay", overlay_visualizacao)

        # -------------------------------------------------
        # SALVAMENTO ORGANIZADO
        # -------------------------------------------------
        if SAVE_IMAGES and frame_count % SAVE_EVERY_N_FRAMES == 0:
            
            # Salva o frame original limpo (sem textos por cima)
            cv2.imwrite(os.path.join(DIRS["frame"], filename), frame)

            # Salva a máscara pequena original 256x256
            cv2.imwrite(os.path.join(DIRS["mask"], f"raw_{filename}"), mask)

            # Salva a máscara redimensionada de 640x480 limpa
            cv2.imwrite(os.path.join(DIRS["mask"], filename), mask_resized)

            # Salva o overlay limpo
            cv2.imwrite(os.path.join(DIRS["overlay"], filename), overlay)

        # -------------------------------------------------
        # LOGS DE EXECUÇÃO
        # -------------------------------------------------
        log_message = (
            f"[FRAME {frame_count}] "
            f"Inferência: {inference_time*1000:.1f} ms | "
            f"FPS: {fps:.2f}"
        )
        print(log_message)

        with open(os.path.join(DIRS["logs"], "runtime.log"), "a") as log_file:
            log_file.write(log_message + "\n")

        # -------------------------------------------------
        # TECLA ESC PARA PARAR CONTROLE LOCAL
        # -------------------------------------------------
        key = cv2.waitKey(1)
        if key == 27:
            break

# =========================================================
# FINALIZAÇÃO DE RECURSOS
# =========================================================
except KeyboardInterrupt:
    print("\n[INFO] Encerrando...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados.")
