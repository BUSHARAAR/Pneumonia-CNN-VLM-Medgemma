import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import timm
from sklearn.metrics import roc_auc_score
from medmnist import PneumoniaMNIST

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Pneumo28(Dataset):
    """
    PneumoniaMNIST returns PIL grayscale images (28x28).
    We convert to torch tensor [1,28,28] in [0,1], apply (optional) augmentation,
    then normalize with train mean/std.
    """
    def __init__(self, base_ds, mean: float, std: float, augment: bool = False):
        self.ds = base_ds
        self.mean = float(mean)
        self.std = float(std)
        self.augment = augment

    def __len__(self):
        return len(self.ds)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Brightness/contrast (CXR-safe mild)
        if random.random() < 0.8:
            b = random.uniform(0.85, 1.15)
            c = random.uniform(0.85, 1.15)
            x = TF.adjust_brightness(x, b)
            x = TF.adjust_contrast(x, c)

        # Small rotation (±7°)
        if random.random() < 0.5:
            angle = random.uniform(-7, 7)
            x = TF.rotate(x, angle=angle)

        # Small translation (≤2 px)
        if random.random() < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            x = TF.affine(x, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])

        # Mild Gaussian noise
        if random.random() < 0.5:
            x = torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)

        return x

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0) / 255.0
        y = int(np.asarray(label).squeeze())

        if self.augment:
            x = self._augment(x)

        x = (x - self.mean) / (self.std + 1e-8)
        return x, y

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)

def build_model(name: str, num_classes: int = 2) -> nn.Module:
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return timm.create_model("resnet18", pretrained=True, num_classes=num_classes, in_chans=1)
    if name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes, in_chans=1)
    if name == "vit_tiny":
        return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=num_classes, in_chans=1)
    raise ValueError(f"Unknown model: {name}")

def maybe_resize(x: torch.Tensor, model_name: str) -> torch.Tensor:
    # ViT expects 224x224; resize only for ViT
    if model_name.lower().startswith("vit"):
        return torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x

def compute_mean_std(ds) -> tuple[float, float]:
    s1, s2 = 0.0, 0.0
    n = len(ds)
    for i in range(n):
        img, _ = ds[i]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        s1 += float(arr.mean())
        s2 += float((arr ** 2).mean())
    mean = s1 / n
    var = (s2 / n) - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), float(std)

@torch.no_grad()
def eval_auc(model: nn.Module, loader: DataLoader, model_name: str, device: str) -> float:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = maybe_resize(x.to(device), model_name)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return float(roc_auc_score(ys, probs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["simplecnn", "resnet18", "efficientnet_b0", "vit_tiny"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", type=int, default=1)
    args, _ = ap.parse_known_args()  # ✅ fixes Colab/Jupyter "-f kernel.json" issue

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_raw = PneumoniaMNIST(split="train", download=True, root=args.data_root)
    val_raw   = PneumoniaMNIST(split="val", download=True, root=args.data_root)

    mean, std = compute_mean_std(train_raw)

    train_ds = Pneumo28(train_raw, mean, std, augment=True)
    val_ds   = Pneumo28(val_raw, mean, std, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory)
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_weights_path = out_dir / f"best_{args.model}.pt"
    best_meta_path = out_dir / f"best_{args.model}_meta.json"

    history = {"train_loss": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = maybe_resize(x.to(device), args.model)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)

        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        val_auc = eval_auc(model, val_loader, args.model, device)

        history["train_loss"].append(float(train_loss))
        history["val_auc"].append(float(val_auc))

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # ✅ save weights ONLY (avoids torch.load pickle errors)
            torch.save(model.state_dict(), best_weights_path)
            # ✅ save metadata separately as JSON (safe)
            meta = {
                "model_name": args.model,
                "mean": mean,
                "std": std,
                "best_val_auc": best_auc,
                "hyperparams": vars(args),
                "history": history,
            }
            best_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved best weights:", best_weights_path)
    print("Saved metadata:", best_meta_path)

if __name__ == "__main__":
    main()
