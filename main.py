import cv2
import time

from camera.capture import *
from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

cap = cv2.VideoCapture(0)

prev = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    tensor = preprocess_frame(frame)

    output = predict(tensor)

    mask = process_output(output)

    now = time.time()

    fps = 1 / (now - prev)

    prev = now

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
