import cv2
import torch
import numpy as np

IMG_SIZE = 256

def preprocess_frame(frame):

    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor
