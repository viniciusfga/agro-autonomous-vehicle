import cv2
import time
import os
import numpy as np

from picamera2 import Picamera2

from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

# --- Configuração de Caminhos ---
BASE_DIR = "capturas"
DIRS = {
    "frame": os.path.join(BASE_DIR, "frame"),
    "mask": os.path.join(BASE_DIR, "mask"),
    "overlay": os.path.join(BASE_DIR, "overlay"),
    "logs": os.path.join(BASE_DIR, "logs")
}

# Cria as pastas caso não existam
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

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

prev = time.time()
frame_count = 0

try:
    while True:
        # Captura frame RGB
        frame_rgb = picam2.capture_array()

        # RGB -> BGR (OpenCV)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_count += 1

        # Pré-processamento
        tensor = preprocess_frame(frame)

        # Inferência
        start_inf = time.time()
        output = predict(tensor)
        inference_time = time.time() - start_inf

        # Pós-processamento (Máscara binária 256x256)
        mask = process_output(output)

	# Salva uma versão da máscara SEM o resize para teste (ela terá tamanho 256x256)
        cv2.imwrite(os.path.join(DIRS["mask"], f"raw_{filename}"), mask)
       
	# Redimensiona a máscara de volta para o tamanho do frame original (640x480)
        # para que o overlay funcione corretamente
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

	# Garante que a matriz seja do tipo correto para o OpenCV salvar
        mask_resized = mask_resized.astype(np.uint8)

        # --- Criação do Overlay ---
        # Converte máscara para 3 canais (colorida) aplicando uma cor verde, por exemplo
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = mask_resized  # Canal G (Verde) preenchido pela máscara

        # Transparência: 70% frame original, 30% máscara verde
        overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

        # FPS
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        # Adiciona FPS no frame de visualização
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # --- Salvamento Organizado ---
        # Criando um nome padronizado (Ex: frame_00001.jpg)
        filename = f"img_{frame_count:05d}.jpg"

        cv2.imwrite(os.path.join(DIRS["frame"], filename), frame)
        cv2.imwrite(os.path.join(DIRS["mask"], filename), mask_resized)
        cv2.imwrite(os.path.join(DIRS["overlay"], filename), overlay)

        # Print de Log no console
        print(f"\n--- Frame {frame_count} ---")
        print(f"Inferência: {inference_time*1000:.1f} ms")
        print(f"FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("\n[INFO] Encerrando...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados.")
