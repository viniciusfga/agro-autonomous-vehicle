import cv2
import numpy as np

THRESHOLD = 0.35

def process_output(output):

    mask = output.squeeze().cpu().numpy()

    print(
        f"[DEBUG] "
        f"min={mask.min():.4f} "
        f"max={mask.max():.4f} "
        f"mean={mask.mean():.4f}"
    )

    mask = (mask > THRESHOLD).astype(np.uint8)

    mask = mask * 255

    return mask
