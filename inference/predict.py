import torch
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.ccrdnet import CustomCCRDNet

MODEL_PATH = "models/ccrdnet_light_weight.pth"

device = torch.device("cpu")

# Reconstrói arquitetura
model = CustomCCRDNet().to(device)

# Carrega pesos treinados
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

# Modo inferência
model.eval()

print("✅ Modelo CCRDNet carregado com sucesso!")

def predict(tensor):

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

    return output
