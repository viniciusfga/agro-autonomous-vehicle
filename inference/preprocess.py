import cv2
import torch
import numpy as np

IMG_SIZE = 256

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(frame):

    # Resize
    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # float32 e escala
    image = image.astype(np.float32) / 255.0

    # Normalização IGUAL ao treinamento
    image = (image - MEAN) / STD

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor
