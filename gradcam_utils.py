import os
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from dataset import load_metadata, build_image_dataframe
from model import build_resnet18_regressor


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XLSX_PATH = "Physical_properties_Alphonso_Images.xlsx"
IMAGE_DIR = "."
CHECKPOINT_PATH = os.path.join("checkpoints", "best_full_finetune_fold5.pth")

IMG_SIZE = 224


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_transform_targets(values, mean, std):
    return np.array(values) * std + mean


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_display_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])


def create_sample_folds(image_df, n_splits=5, random_state=42):
    sample_df = (
        image_df[["sample_id"]]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for _, (train_idx, val_idx) in enumerate(kf.split(sample_df), start=1):
        train_sample_ids = sample_df.iloc[train_idx]["sample_id"].tolist()
        val_sample_ids = sample_df.iloc[val_idx]["sample_id"].tolist()

        train_df = image_df[image_df["sample_id"].isin(train_sample_ids)].reset_index(drop=True)
        val_df = image_df[image_df["sample_id"].isin(val_sample_ids)].reset_index(drop=True)

        folds.append((train_df, val_df))

    return folds


def load_fold_data(target_fold=5):
    metadata_df = load_metadata(XLSX_PATH)
    image_df = build_image_dataframe(IMAGE_DIR, metadata_df)

    folds = create_sample_folds(image_df, n_splits=5, random_state=SEED)
    train_df, val_df = folds[target_fold - 1]

    train_df = train_df.copy()
    val_df = val_df.copy()

    target_mean = train_df["weight_g"].mean()
    target_std = train_df["weight_g"].std()
    if target_std == 0:
        target_std = 1.0

    train_df["weight_scaled"] = (train_df["weight_g"] - target_mean) / target_std
    val_df["weight_scaled"] = (val_df["weight_g"] - target_mean) / target_std

    return train_df, val_df, target_mean, target_std


def load_model(checkpoint_path):
    model = build_resnet18_regressor(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_validation_set(model, val_df, target_mean, target_std):
    transform = get_val_transform()

    records = []
    for _, row in val_df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(x).item()

        pred_weight = pred_scaled * target_std + target_mean
        actual_weight = float(row["weight_g"])
        abs_error = abs(pred_weight - actual_weight)

        records.append({
            "image_path": row["image_path"],
            "file_name": row["file_name"],
            "sample_id": int(row["sample_id"]),
            "view_id": int(row["view_id"]),
            "actual_weight": actual_weight,
            "predicted_weight": pred_weight,
            "abs_error": abs_error
        })

    pred_df = pd.DataFrame(records).sort_values("abs_error").reset_index(drop=True)
    return pred_df


class GradCAMRegressor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)  # shape: [1, 1]
        score = output.squeeze()
        score.backward(retain_graph=True)

        gradients = self.gradients[0]         # [C, H, W]
        activations = self.activations[0]     # [C, H, W]

        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = torch.zeros(activations.shape[1:], device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, score.item()


def overlay_heatmap_on_image(image_np, cam, alpha=0.4):
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(
        (image_np.shape[1], image_np.shape[0]),
        resample=Image.BILINEAR
    )
    cam_resized = np.array(cam_resized) / 255.0

    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return cam_resized, heatmap, overlay


def visualize_gradcam_for_row(model, row, target_mean, target_std):
    val_transform = get_val_transform()
    display_transform = get_display_transform()

    raw_image = Image.open(row["image_path"]).convert("RGB")

    input_tensor = val_transform(raw_image).unsqueeze(0).to(DEVICE)
    display_image = display_transform(raw_image).permute(1, 2, 0).numpy()

    gradcam = GradCAMRegressor(model, model.layer4[-1])
    cam, pred_scaled = gradcam.generate(input_tensor)

    pred_weight = pred_scaled * target_std + target_mean
    actual_weight = float(row["actual_weight"])

    cam_resized, heatmap, overlay = overlay_heatmap_on_image(display_image, cam)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(display_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(display_image)
    plt.imshow(cam_resized, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"Actual: {actual_weight:.1f} g\n"
        f"Pred: {pred_weight:.1f} g\n"
        f"Abs Error: {abs(pred_weight - actual_weight):.1f} g"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed(SEED)

    print("DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_df, val_df, target_mean, target_std = load_fold_data(target_fold=5)
    model = load_model(CHECKPOINT_PATH)

    pred_df = predict_validation_set(model, val_df, target_mean, target_std)
    pred_df.to_csv("fold5_val_predictions.csv", index=False)

    print("\n=== Best predicted samples ===")
    print(pred_df.head(5)[["file_name", "actual_weight", "predicted_weight", "abs_error"]])

    print("\n=== Worst predicted samples ===")
    print(pred_df.tail(5)[["file_name", "actual_weight", "predicted_weight", "abs_error"]])

    # 잘 맞춘 샘플 2개
    print("\nVisualizing best samples...")
    for i in range(2):
        visualize_gradcam_for_row(model, pred_df.iloc[i], target_mean, target_std)

    # 못 맞춘 샘플 2개
    print("\nVisualizing worst samples...")
    for i in range(1, 3):
        visualize_gradcam_for_row(model, pred_df.iloc[-i], target_mean, target_std)