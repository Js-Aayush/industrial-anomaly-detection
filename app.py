from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


app = FastAPI()


# =========================
# LOAD SAVED DATA
# =========================

saved_data = torch.load(
    "bottle_anomaly_data.pth",
    map_location="cpu"
)

train_features = saved_data["train_features"]
threshold = saved_data["threshold"]


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD RESNET18
# =========================

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(weights=weights)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()

for param in feature_extractor.parameters():
    param.requires_grad = False


# =========================
# PREDICTION
# =========================

def predict_image(image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        features = feature_extractor(image)

        features = features.flatten(1)

        distances = torch.cdist(
            features,
            train_features
        )

        anomaly_score = torch.min(
            distances
        ).item()

    prediction = (
        "Defective"
        if anomaly_score > threshold
        else "Normal"
    )

    return prediction, anomaly_score


# =========================
# API
# =========================

@app.get("/")
def home():
    return {
        "message": "Industrial Bottle Anomaly Detection API"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    prediction, anomaly_score = predict_image(
        image
    )

    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "threshold": float(threshold)
    }