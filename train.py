import os
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import load_metadata, build_image_dataframe, MangoImageDataset
from model import build_resnet18_regressor, freeze_backbone, unfreeze_all

from tqdm import tqdm

# =========================================================
# 1. 설정
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XLSX_PATH = "Physical_properties_Alphonso_Images.xlsx"
IMAGE_DIR = "."

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0   # Windows 환경이면 0 권장

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 2. 재현성 고정
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. sample 기준 shared fold 생성
# =========================================================
def create_sample_folds(image_df, n_splits=5, random_state=42):
    sample_df = (
        image_df[["sample_id"]]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(sample_df), start=1):
        train_sample_ids = sample_df.iloc[train_idx]["sample_id"].tolist()
        val_sample_ids = sample_df.iloc[val_idx]["sample_id"].tolist()

        train_df = image_df[image_df["sample_id"].isin(train_sample_ids)].reset_index(drop=True)
        val_df = image_df[image_df["sample_id"].isin(val_sample_ids)].reset_index(drop=True)

        folds.append((train_df, val_df))

        print(
            f"Fold {fold_idx}: "
            f"train_samples={len(train_sample_ids)}, val_samples={len(val_sample_ids)}, "
            f"train_images={len(train_df)}, val_images={len(val_df)}"
        )

    return folds


# =========================================================
# 4. transform
# =========================================================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# =========================================================
# 5. dataloader
# =========================================================
def create_dataloaders(train_df, val_df, batch_size=8, target_col="weight_scaled"):
    train_transform, val_transform = get_transforms()

    train_dataset = MangoImageDataset(train_df, transform=train_transform, target_col=target_col)
    val_dataset = MangoImageDataset(val_df, transform=val_transform, target_col=target_col)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader

def inverse_transform_targets(values, mean, std):
    return np.array(values) * std + mean

# =========================================================
# 6. 한 epoch 학습
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, target_mean, target_std, fold_idx=None, epoch_idx=None):
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_targets = []

    desc = "Training"
    if fold_idx is not None and epoch_idx is not None:
        desc = f"Fold {fold_idx} Epoch {epoch_idx}"

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["target"].float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        running_loss += loss.item() * images.size(0)

        all_preds.extend(outputs.detach().cpu().numpy().flatten())
        all_targets.extend(targets.detach().cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets_original = inverse_transform_targets(all_targets, target_mean, target_std)
    all_preds_original = inverse_transform_targets(all_preds, target_mean, target_std)

    epoch_mae = mean_absolute_error(all_targets_original, all_preds_original)
    epoch_rmse = np.sqrt(mean_squared_error(all_targets_original, all_preds_original))
    epoch_r2 = r2_score(all_targets_original, all_preds_original)

    return epoch_loss, epoch_mae, epoch_rmse, epoch_r2


# =========================================================
# 7. 검증
# =========================================================
def validate_one_epoch(model, loader, criterion, device, target_mean, target_std):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].float().to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets_original = inverse_transform_targets(all_targets, target_mean, target_std)
    all_preds_original = inverse_transform_targets(all_preds, target_mean, target_std)

    epoch_mae = mean_absolute_error(all_targets_original, all_preds_original)
    epoch_rmse = np.sqrt(mean_squared_error(all_targets_original, all_preds_original))
    epoch_r2 = r2_score(all_targets_original, all_preds_original)

    return (
        epoch_loss,
        epoch_mae,
        epoch_rmse,
        epoch_r2,
        all_targets_original,
        all_preds_original
    )


# =========================================================
# 8. 한 fold 학습
# =========================================================
def run_fold(train_df, val_df, training_mode="head_only", fold_idx=1):
    train_df = train_df.copy()
    val_df = val_df.copy()

    target_mean = train_df["weight_g"].mean()
    target_std = train_df["weight_g"].std()

    if target_std == 0:
        target_std = 1.0

    train_df["weight_scaled"] = (train_df["weight_g"] - target_mean) / target_std
    val_df["weight_scaled"] = (val_df["weight_g"] - target_mean) / target_std

    train_loader, val_loader = create_dataloaders(train_df, val_df, batch_size=BATCH_SIZE, target_col="weight_scaled")

    model = build_resnet18_regressor(pretrained=True)

    if training_mode == "head_only":
        model = freeze_backbone(model)
    elif training_mode == "full_finetune":
        model = unfreeze_all(model)
    else:
        raise ValueError("training_mode must be 'head_only' or 'full_finetune'")

    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    history = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mae, train_rmse, train_r2 = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            target_mean=target_mean,
            target_std=target_std,
            fold_idx=fold_idx,
            epoch_idx=epoch + 1
        )

        val_loss, val_mae, val_rmse, val_r2, y_true, y_pred = validate_one_epoch(
            model, val_loader, criterion, DEVICE,
            target_mean=target_mean,
            target_std=target_std
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        })

        print(
            f"[Fold {fold_idx}][{training_mode}] Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    # best model로 다시 검증
    val_loss, val_mae, val_rmse, val_r2, y_true, y_pred = validate_one_epoch(
        model, val_loader, criterion, DEVICE, target_mean=target_mean, target_std=target_std
    )

    save_path = os.path.join(SAVE_DIR, f"best_{training_mode}_fold{fold_idx}.pth")
    torch.save(model.state_dict(), save_path)

    fold_result = {
        "fold": fold_idx,
        "training_mode": training_mode,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "save_path": save_path
    }

    history_df = pd.DataFrame(history)
    return fold_result, history_df


# =========================================================
# 9. 전체 CV 실행
# =========================================================
def run_cross_validation(training_mode="head_only"):
    print("DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    set_seed(SEED)

    metadata_df = load_metadata(XLSX_PATH)
    image_df = build_image_dataframe(IMAGE_DIR, metadata_df)

    print("image_df shape:", image_df.shape)
    print(image_df.head())

    folds = create_sample_folds(image_df, n_splits=5, random_state=SEED)

    all_fold_results = []
    all_histories = []

    for fold_idx, (train_df, val_df) in enumerate(folds, start=1):
        fold_result, history_df = run_fold(
            train_df=train_df,
            val_df=val_df,
            training_mode=training_mode,
            fold_idx=fold_idx
        )
        all_fold_results.append(fold_result)
        history_df["fold"] = fold_idx
        history_df["training_mode"] = training_mode
        all_histories.append(history_df)

    results_df = pd.DataFrame(all_fold_results)
    history_df = pd.concat(all_histories, axis=0).reset_index(drop=True)

    print("\n===== Cross Validation Results =====")
    print(results_df)

    print("\n===== Mean Performance =====")
    print(results_df[["val_mae", "val_rmse", "val_r2"]].mean())

    return results_df, history_df

# =========================================================
# 10. 실행
# =========================================================
if __name__ == "__main__":
    # 1) head-only
    # print("\n================ HEAD ONLY ================\n")
    # head_results, head_history = run_cross_validation(training_mode="head_only")
    # head_results.to_csv("head_only_results.csv", index=False)
    # head_history.to_csv("head_only_history.csv", index=False)

    # 2) full fine-tuning
    print("\n================ FULL FINE-TUNING ================\n")
    full_results, full_history = run_cross_validation(training_mode="full_finetune")
    full_results.to_csv("full_finetune_results.csv", index=False)
    full_history.to_csv("full_finetune_history.csv", index=False)