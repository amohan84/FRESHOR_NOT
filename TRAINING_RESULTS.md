# Training Results — FreshOrNot MobileNetV2

## Configuration

| Parameter        | Value                          |
|-----------------|-------------------------------|
| Model            | MobileNetV2 (ImageNet pretrained) |
| Dataset          | `archive/dataset/Train` (local) |
| Total images     | 23,619                        |
| Train split      | 18,896 (80%)                  |
| Val split        | 4,723 (20%)                   |
| Classes          | 18 (9 fresh + 9 stale)        |
| Image size       | 224 × 224                     |
| Batch size       | 32                            |
| Device           | CPU                           |
| Random seed      | 42                            |

---

## Phase 1 — Head Training (Frozen Backbone)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Saved |
|-------|-----------|---------|----------|--------|-------|
| 1     | 0.3738    | 0.1524  | 0.879    | 0.949  | ✅     |
| 2     | 0.1833    | 0.0997  | 0.935    | 0.969  | ✅     |
| 3     | 0.1526    | 0.1442  | 0.945    | 0.949  |       |

- Learning rate: `1e-3` (Adam)
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Best val accuracy after Phase 1: **96.9%**

---

## Phase 2 — Fine-Tuning (Last 4 Backbone Layers Unfrozen)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Saved |
|-------|-----------|---------|----------|--------|-------|
| 1     | 0.1071    | 0.0641  | 0.963    | 0.978  | ✅     |
| 2     | 0.0445    | 0.0399  | 0.983    | 0.987  | ✅     |

- Learning rate: `1e-4` (Adam)
- Scheduler: CosineAnnealingLR
- Best val accuracy after Phase 2: **98.7%**

---

## Output

| Item         | Path                    |
|-------------|------------------------|
| Model file   | `model/freshor_not.pt` |

---

## Augmentation (Training)

- Random resized crop (scale 0.7–1.0)
- Random horizontal flip
- Color jitter (brightness 0.3, contrast 0.2, saturation 0.2)
- Normalize (ImageNet mean/std)
