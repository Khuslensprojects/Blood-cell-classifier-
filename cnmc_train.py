"""Leukemic (ALL) vs normal cell classifier on the C-NMC 2019 dataset.

Key design choice: train/val/test are split BY PATIENT, so cells from the same
person never appear in both training and test. Splitting by image would leak
patient-specific features and inflate the results.

Run (in Colab, after downloading the dataset):
    python cnmc_train.py --data_dir "<path to extracted dataset>" --epochs 5
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES = ["normal", "ALL (leukemic)"]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_images(root):
    """Return (path, label, patient_id) for every training image. label 1 = ALL."""
    items = []
    for p in Path(root).rglob("*.bmp"):
        in_fold = any(part.startswith("fold_") for part in p.parts)
        if in_fold and p.parent.name in ("all", "hem"):
            label = 1 if p.parent.name == "all" else 0
            patient = f"{label}_{p.name.split('_')[1]}"  # e.g. UID_57_29_1_all.bmp -> 57
            items.append((str(p), label, patient))
    if not items:
        raise SystemExit(f"No fold_*/all|hem/*.bmp images found under {root}")
    return items


def split_by_patient(items, seed, frac=(0.70, 0.15, 0.15)):
    """Stratified split at the patient level (each patient has a single label)."""
    rng = random.Random(seed)
    patients = {0: set(), 1: set()}
    for _, y, pid in items:
        patients[y].add(pid)
    split_of = {}
    for y, pids in patients.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        n_tr = int(round(frac[0] * len(pids)))
        n_va = int(round(frac[1] * len(pids)))
        for i, pid in enumerate(pids):
            split_of[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[split_of[it[2]]].append(it)
    # Safety check: no patient appears in more than one split
    ids = {k: {it[2] for it in v} for k, v in out.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    return out


class CellDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y, _ = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def compute_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "balanced_accuracy": float((sens + spec) / 2),
        "roc_auc": float(roc_auc_score(y, prob)),
        "sensitivity_ALL": float(sens),
        "specificity_normal": float(spec),
        "f1_ALL": float(f1_score(y, pred)),
        "accuracy": float((pred == y).mean()),
    }


def predict_probs(model, dl, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def load_small(items, size=32):
    X = np.stack(
        [np.asarray(Image.open(p).convert("L").resize((size, size))).ravel() / 255.0 for p, _, _ in items]
    )
    return X, np.array([y for _, y, _ in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    items = find_images(args.data_dir)
    splits = split_by_patient(items, args.seed)
    for name, its in splits.items():
        n_pat = len({it[2] for it in its})
        n_all = sum(it[1] for it in its)
        print(f"{name:5s}: {len(its):5d} images, {n_pat:3d} patients, {n_all} ALL / {len(its) - n_all} normal")

    # --- Baseline: logistic regression on 32x32 grayscale pixels ---
    Xtr, ytr = load_small(splits["train"])
    Xte, yte = load_small(splits["test"])
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xtr, ytr)
    base = compute_metrics(yte, clf.predict_proba(Xte)[:, 1])
    print("\nBaseline (logistic regression, raw pixels):", json.dumps(base, indent=2))

    # --- Data ---
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20), T.ToTensor(), norm]
    )
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
    train_dl = DataLoader(CellDS(splits["train"], train_tf), batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(CellDS(splits["val"], eval_tf), batch_size=args.batch, num_workers=2)
    test_dl = DataLoader(CellDS(splits["test"], eval_tf), batch_size=args.batch, num_workers=2)

    # --- Model (class-weighted loss because ALL images outnumber normal ~2:1) ---
    n1 = sum(it[1] for it in splits["train"])
    n0 = len(splits["train"]) - n1
    weights = torch.tensor([len(splits["train"]) / (2 * n0), len(splits["train"]) / (2 * n1)], dtype=torch.float)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        yv, pv = predict_probs(model, val_dl, device)
        auc = roc_auc_score(yv, pv)
        print(f"Epoch {epoch + 1}/{args.epochs} | val ROC-AUC {auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_cnmc_model.pt")

    # --- Test ---
    model.load_state_dict(torch.load("best_cnmc_model.pt", map_location=device))
    y, p = predict_probs(model, test_dl, device)
    res = compute_metrics(y, p)
    print("\nFine-tuned ResNet18 (patient-level test set):", json.dumps(res, indent=2))
    json.dump({"baseline": base, "resnet18": res}, open("cnmc_metrics.json", "w"), indent=2)

    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("C-NMC test set (split by patient)")
    plt.tight_layout()
    plt.savefig("cnmc_confusion_matrix.png", dpi=150)

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"ResNet18 (AUC {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnmc_roc.png", dpi=150)
    print("Saved best_cnmc_model.pt, cnmc_metrics.json, cnmc_confusion_matrix.png, cnmc_roc.png")


if __name__ == "__main__":
    main()
