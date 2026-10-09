import torch
import os

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
saved_data = torch.load(
    "bottle_anomaly_data.pth",
    map_location="cpu"
)

train_features = saved_data["train_features"]
threshold = saved_data["threshold"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
weights = ResNet18_Weights.DEFAULT

resnet = resnet18(
    weights=weights
)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()

for param in feature_extractor.parameters():
    param.requires_grad = False

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

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


    if anomaly_score > threshold:
        prediction = "Defective"
    else:
        prediction = "Normal"


    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "threshold": float(threshold)
    }
if __name__ == "__main__":

    image_path = "bottle/test/broken_large/000.png"

    result = predict_image(
        image_path
    )

    print(result)