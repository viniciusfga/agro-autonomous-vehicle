# inference/postprocess.py
import cv2
import numpy as np

def process_output(output):
    mask = output.squeeze().cpu().numpy()
    
    # --- LOG DE DEPURAÇÃO ---
    print(f"[DEBUG MÁSCARA] Valor Máx: {mask.max():.4f} | Valor Mín: {mask.min():.4f} | Média: {mask.mean():.4f}")
    
    # Se o valor máximo for menor que 0.5, o threshold atual vai apagar tudo.
    # Como teste temporário, você pode baixar para 0.3 ou usar uma binarização simples
    mask = (mask > 0.5).astype(np.uint8)
    mask = mask * 255
    
    return mask
