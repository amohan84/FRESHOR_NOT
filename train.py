"""
train.py — Fine-tune MobileNetV2 on Swoyam2609 Fresh-and-Stale dataset
──────────────────────────────────────────────────────────────────────
Dataset:  archive/dataset/Train  (Swoyam2609 Fresh-and-Stale)

Setup:
  1. pip install torch torchvision
  2. python train.py

The script will:
  • Read training data from archive/dataset/Train (80/20 train-val split)
  • Fine-tune MobileNetV2 (ImageNet pretrained) for fresh/stale classification
  • Save the best model to model/freshor_not.pt
"""

import os
import pathlib
import sys

# ── Verify dependencies ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import datasets, models
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch / torchvision not found.  Install with:\n"
          "  pip install torch torchvision")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
ARCHIVE_DIR     = pathlib.Path("archive/dataset")
TRAIN_DIR       = ARCHIVE_DIR / "Train"
VAL_SPLIT       = 0.2
MODEL_DIR       = pathlib.Path("model")
MODEL_PATH      = MODEL_DIR / "freshor_not.pt"
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 3
EPOCHS_FINETUNE = 2
LR_HEAD         = 1e-3
LR_FINETUNE     = 1e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ── Data transforms ───────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(n_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )
    return m.to(DEVICE)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print(f"Loading dataset from: {TRAIN_DIR}")

    # Load full dataset with training transforms; we'll override val subset transforms below
    full_ds   = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    n_classes = len(full_ds.classes)
    print(f"Classes ({n_classes}): {full_ds.classes}")

    # 80 / 20 train-val split (reproducible)
    n_total   = len(full_ds)
    n_val     = int(n_total * VAL_SPLIT)
    n_train   = n_total - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=generator)

    # Apply val transforms to the validation subset
    val_ds.dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)

    print(f"Split: {n_train} train / {n_val} val  (total {n_total})")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_model(n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_acc = 0.0
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"\n── Phase 1: training head ({EPOCHS_FROZEN} epochs) ──")
    for epoch in range(1, EPOCHS_FROZEN + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)
        print(f"  Epoch {epoch:2d}  loss={tr_loss:.4f}/{vl_loss:.4f}  acc={tr_acc:.3f}/{vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model, MODEL_PATH)
            print(f"           ✅ saved (val_acc={vl_acc:.3f})")

    print(f"\n── Phase 2: fine-tuning ({EPOCHS_FINETUNE} epochs) ──")
    for layer in list(model.features.children())[-4:]:
        for p in layer.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_FINETUNE)

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        print(f"  Epoch {epoch:2d}  loss={tr_loss:.4f}/{vl_loss:.4f}  acc={tr_acc:.3f}/{vl_acc:.3f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model, MODEL_PATH)
            print(f"           ✅ saved (val_acc={vl_acc:.3f})")

    print(f"\n✅  Best val accuracy: {best_val_acc:.3f}")
    print(f"   Model saved → {MODEL_PATH}")
    print("   Launch app:  streamlit run app.py")


if __name__ == "__main__":
    train()
