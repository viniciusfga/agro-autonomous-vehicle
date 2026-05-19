import cv2
import time
import os

# Importações do seu projeto
from camera.capture import *
from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

print("[INFO] Inicializando a câmera...")
cap = cv2.VideoCapture(0)

# Verifica se a câmera realmente abriu
if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera. Verifique a conexão do cabo/USB.")
    exit()

print("[INFO] Câmera iniciada com sucesso. Iniciando loop de inferência...")
print("[DICA] Como você está no modo Lite (CLI), pressione 'Ctrl + C' no terminal para encerrar o script.")

prev = time.time()
frame_count = 0

try:
    while True:
        # 1. Captura do Frame
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("[AVISO] Falha ao capturar o frame da câmera. Pulando...")
            continue

        frame_count += 1
        t_capture = time.time() - start_time

        # 2. Pré-processamento
        start_time = time.time()
        tensor = preprocess_frame(frame)
        t_preprocess = time.time() - start_time

        # 3. Inferência do Modelo (Predict)
        start_time = time.time()
        output = predict(tensor)
        t_inference = time.time() - start_time

        # 4. Pós-processamento (Máscara de segmentação)
        start_time = time.time()
        mask = process_output(output)
        t_postprocess = time.time() - start_time

        # 5. Cálculo de FPS (Geral do Loop)
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        # 6. Desenhar FPS no Frame original
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 7. Salvar saídas no disco (Substituindo os arquivos continuamente)
        cv2.imwrite("frame.jpg", frame)
        cv2.imwrite("mask.jpg", mask)

        # 8. Print de Diagnóstico no Terminal (A cada frame)
        print(f"\n--- Frame {frame_count} ---")
        print(f"| Captura:       {t_capture*1000:.1f}ms")
        print(f"| Pré-proc:      {t_preprocess*1000:.1f}ms")
        print(f"| Inferência:    {t_inference*1000:.1f}ms")
        print(f"| Pós-proc:      {t_postprocess*1000:.1f}ms")
        print(f"| FPS Atual:     {fps:.2f}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupção pelo usuário recebida. Encerrando o script...")

finally:
    # Garante que os recursos serão liberados mesmo se houver erro
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recursos da câmera liberados. Execução finalizada.")
