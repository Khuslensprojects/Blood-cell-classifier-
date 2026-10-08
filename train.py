"""Blood cell classifier: baseline vs. fine-tuned ResNet18 on BloodMNIST.

Setup:
    pip install torch torchvision medmnist scikit-learn matplotlib
Run:
    python train.py --epochs 5
"""
import argparse

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from medmnist import INFO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

NAME = "bloodmnist"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_baseline(DataClass):
    """Logistic regression on raw 28x28 pixels. The number the CNN must beat."""
    tr = DataClass(split="train", download=True, size=28)
    te = DataClass(split="test", download=True, size=28)
    Xtr = tr.imgs.reshape(len(tr), -1) / 255.0
    Xte = te.imgs.reshape(len(te), -1) / 255.0
    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, tr.labels.ravel())
    acc = clf.score(Xte, te.labels.ravel())
    print(f"Baseline (logistic regression, raw pixels) test accuracy: {acc:.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    info = INFO[NAME]
    DataClass = getattr(medmnist, info["python_class"])
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    device = get_device()
    print(f"Device: {device} | classes: {class_names}")

    base_acc = run_baseline(DataClass)

    # --- Data (224px so pretrained ImageNet weights make sense) ---
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(15), T.ToTensor(), norm])
    eval_tf = T.Compose([T.ToTensor(), norm])
    train_ds = DataClass(split="train", transform=train_tf, download=True, size=224)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, size=224)
    test_ds = DataClass(split="test", transform=eval_tf, download=True, size=224)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    # --- Model ---
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(dl):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in dl:
                preds.append(model(x.to(device)).argmax(1).cpu())
                labels.append(y.squeeze(1))
        return torch.cat(preds).numpy(), torch.cat(labels).numpy()

    # --- Train ---
    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.squeeze(1).long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        p, l = evaluate(val_dl)
        val_acc = (p == l).mean()
        print(f"Epoch {epoch + 1}/{args.epochs} | val acc {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # --- Test + error analysis ---
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    p, l = evaluate(test_dl)
    print(f"\nFine-tuned ResNet18 test accuracy: {(p == l).mean():.3f} (baseline: {base_acc:.3f})\n")
    print(classification_report(l, p, target_names=class_names))

    cm = confusion_matrix(l, p, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved best_model.pt and confusion_matrix.png")


if __name__ == "__main__":
    main()
