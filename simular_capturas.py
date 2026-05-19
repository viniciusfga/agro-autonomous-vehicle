import cv2
import os
import numpy as np
import time

from inference.preprocess import preprocess_frame
from inference.predict import predict
from inference.postprocess import process_output

# --- Configuração de Caminhos ---
PASTA_ORIGEM = "capturas/frame"
PASTA_DESTINO = "capturas_teste_offline"

DIRS = {
    "mask": os.path.join(PASTA_DESTINO, "mask"),
    "overlay": os.path.join(PASTA_DESTINO, "overlay")
}

# Cria as novas pastas de teste para não misturar com as antigas
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

# Lista todas as imagens originais salvas na pasta frame e coloca em ordem
lista_imagens = sorted([f for f in os.listdir(PASTA_ORIGEM) if f.endswith('.jpg')])

if not lista_imagens:
    print(f"❌ Nenhuma imagem encontrada em '{PASTA_ORIGEM}'. Garanta que você coletou fotos antes.")
    exit()

print(f"🤖 Encontradas {len(lista_imagens)} imagens para simulação. Iniciando processamento offline...\n")

for nome_arquivo in lista_imagens:
    caminho_imagem = os.path.join(PASTA_ORIGEM, nome_arquivo)
    
    # 1. Carrega a imagem simulando o frame da câmera
    frame = cv2.imread(caminho_imagem)
    if frame is None:
        continue

    # 2. Pré-processamento
    tensor = preprocess_frame(frame)

    # 3. Inferência
    output = predict(tensor)

    # 4. Pós-processamento (Verificando valores internos para depuração)
    mask_raw = output.squeeze().cpu().numpy()
    
    # 📝 Print de depuração para você ver o comportamento do modelo em cada imagem
    print(f"[{nome_arquivo}] Máx: {mask_raw.max():.4f} | Mín: {mask_raw.min():.4f} | Média: {mask_raw.mean():.4f}")

    # --- Ajuste do Threshold ---
    # Se os valores máximos acima forem menores que 0.5, altere o valor abaixo para 0.2 ou 0.1
    THRESHOLD = 0.5 
    mask = (mask_raw > THRESHOLD).astype(np.uint8) * 255

    # 5. Redimensiona para o tamanho original com Interpolação por Vizinho Próximo
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.uint8)

    # 6. Criação do Overlay (Máscara Verde)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 1] = mask_resized  # Aplica o canal verde

    # Transparência: 70% imagem original, 30% máscara verde
    overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

    # 7. Salva os novos resultados obtidos na simulação
    cv2.imwrite(os.path.join(DIRS["mask"], nome_arquivo), mask_resized)
    cv2.imwrite(os.path.join(DIRS["overlay"], nome_arquivo), overlay)

print("\n✅ Simulação concluída! Verifique os resultados na pasta 'capturas_teste_offline/'")
