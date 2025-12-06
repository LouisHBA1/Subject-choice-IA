# Subject-choice-IA
Automatic Crack Detection on Aircraft Wings

import os
import cv2
import numpy as np

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x/img_w, y/img_h, w/img_w, h/img_h

def convert_masks_folder_to_yolo(images_dir, masks_dir, labels_out_dir, class_id=0):
    os.makedirs(labels_out_dir, exist_ok=True)
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(('.jpg','.png','.jpeg')):
            continue

        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bbox = mask_to_bbox(mask)

        label_file = os.path.join(labels_out_dir, fname.replace('.jpg', '.txt').replace('.png','.txt'))
        if bbox is None:
            open(label_file, 'w').close()
            continue

        xmin, ymin, xmax, ymax = bbox
        yolo = bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)

        with open(label_file, 'w') as f:
            f.write(f"{class_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}\n")

if __name__ == "__main__":
    convert_masks_folder_to_yolo(
        "data/images/train",
        "data/masks/train",
        "yolov8/labels/train"
    )

    path: ../data
train: images/train
val: images/val
test: images/test

names:
  0: crack

  from ultralytics import YOLO
import os

model = YOLO("runs/yolov8/crack_exp/weights/best.pt")

def infer_image(img_path, save_folder="yolov8/results"):
    os.makedirs(save_folder, exist_ok=True)
    results = model.predict(
        source=img_path,
        imgsz=640,
        conf=0.25,
        save=True,
        project=save_folder,
        name="detections"
    )
    return results

if __name__ == "__main__":
    infer_image("data/images/test/example.jpg")

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.enc4 = DoubleConv(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.masks_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = img.astype("float32") / 255.0
        img = np.transpose(img, (2,0,1))

        return torch.tensor(img), torch.tensor(mask).unsqueeze(0)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from unet.model import UNet
from unet.datasets import CrackDataset

def dice_coeff(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + eps)).item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = A.Compose([A.Resize(512,512), A.RandomRotate90(),
                          A.Flip(), A.RandomBrightnessContrast()])
    val_tf = A.Compose([A.Resize(512,512)])

    train_ds = CrackDataset("data/images/train", "data/masks/train", train_tf)
    val_ds = CrackDataset("data/images/val", "data/masks/val", val_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = UNet().to(device)
    optim_ = optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    best_dice = 0

    for epoch in range(1, 101):
        model.train()
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            pred = model(imgs)
            loss = bce(pred, masks)

            optim_.zero_grad()
            loss.backward()
            optim_.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = torch.sigmoid(model(imgs))
                dices.append(dice_coeff(pred, masks))

        avg_dice = sum(dices)/len(dices)

        print(f"Epoch {epoch} — Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "unet/best.pth")

if __name__ == "__main__":
    main()

    import torch
from torch.utils.data import DataLoader
from unet.model import UNet
from unet.datasets import CrackDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet/best.pth", map_location=device))
    model.eval()

    test_ds = CrackDataset("data/images/test", "data/masks/test")
    loader = DataLoader(test_ds, batch_size=1)

    dices = []

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).float()

            inter = (pred * mask).sum()
            dice = (2 * inter) / (pred.sum() + mask.sum() + 1e-6)
            dices.append(dice.item())

    print("Mean Dice:", sum(dices)/len(dices))

if __name__ == "__main__":
    main()
import time
from yolov8.infer_yolo import infer_image
from unet.infer_unet import infer_image as infer_unet

img = "data/images/test/example.jpg"

t0 = time.time()
infer_image(img)
print("YOLOv8 time:", (time.time() - t0)*1000, "ms")

t0 = time.time()
infer_unet(img)
print("U-Net time:", (time.time() - t0)*1000, "ms")

