import cv2
import numpy as np

def process_output(output):

    mask = output.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)

    mask = mask * 255

    return mask
