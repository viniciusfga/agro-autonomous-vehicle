import cv2
import time

from picamera2 import Picamera2

from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

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
        frame = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        frame_count += 1

        # Pré-processamento
        tensor = preprocess_frame(frame)

        # Inferência
        start_inf = time.time()

        output = predict(tensor)

        inference_time = time.time() - start_inf

        # Pós-processamento
        mask = process_output(output)

        # FPS
        now = time.time()

        fps = 1 / (now - prev)

        prev = now

        # Texto
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # Salva continuamente
        cv2.imwrite("frame.jpg", frame)

        cv2.imwrite("mask.jpg", mask)

        print(f"\n--- Frame {frame_count} ---")
        print(f"Inferência: {inference_time*1000:.1f} ms")
        print(f"FPS: {fps:.2f}")

except KeyboardInterrupt:

    print("\n[INFO] Encerrando...")

finally:

    picam2.stop()

    cv2.destroyAllWindows()

    print("[INFO] Recursos liberados.")
