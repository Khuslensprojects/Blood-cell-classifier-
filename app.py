"""Upload-and-predict web app for the blood cell classifier (Gradio)."""
import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

# Same order as the BloodMNIST label indices used in training
CLASSES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(img):
    if img is None:
        return None
    x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {name: float(p) for name, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a single blood cell image"),
    outputs=gr.Label(num_top_classes=4, label="Predicted cell type"),
    title="Blood Cell Classifier",
    description=(
        "Fine-tuned ResNet18 trained on BloodMNIST (8 normal blood cell types). "
        "Works best on a single, centered cell from a stained blood smear, similar "
        "to the training images. It cannot detect disease, and it will still give "
        "an answer for images it was not trained on (like random photos), so treat "
        "low-confidence results with caution. Research demo only, not for diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch()
