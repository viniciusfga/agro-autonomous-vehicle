import torch

MODEL_PATH = "models/ccrdnet_light_weight.pth"

device = torch.device("cpu")

model = torch.load(
    MODEL_PATH,
    map_location=device
)

model.eval()

def predict(tensor):

    with torch.no_grad():
        output = model(tensor)

    return output
