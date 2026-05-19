import cv2
import numpy as np

def process_output(output):

    mask = output.squeeze().cpu().numpy()

    mask = (mask * 255).astype(np.uint8)

    return mask
