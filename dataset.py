import os
import re
from glob import glob

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def normalize_sample_id(x):
    x = str(x).strip()
    m = re.search(r'AW0*([0-9]+)', x)
    return int(m.group(1)) if m else None


def load_metadata(xlsx_path):
    raw = pd.read_excel(xlsx_path, header=None)

    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    df = df[[
        "Sample No",
        "fruit diameter (mm)",
        "width across schoulder (mm)",
        "Actual Weight (gms)"
    ]].copy()

    df = df.rename(columns={
        "Sample No": "sample_no",
        "fruit diameter (mm)": "diameter_mm",
        "width across schoulder (mm)": "shoulder_width_mm",
        "Actual Weight (gms)": "weight_g"
    })

    for col in ["diameter_mm", "shoulder_width_mm", "weight_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample_id"] = df["sample_no"].apply(normalize_sample_id)
    return df


def extract_sample_id_from_filename(fname):
    m = re.search(r'AW0*([0-9]+)_([12])\.jpg$', fname, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_view_id_from_filename(fname):
    m = re.search(r'AW0*([0-9]+)_([12])\.jpg$', fname, re.IGNORECASE)
    return int(m.group(2)) if m else None


def build_image_dataframe(image_dir, metadata_df):
    image_paths = glob(os.path.join(image_dir, "AW*.jpg"))

    img_df = pd.DataFrame({"image_path": image_paths})
    img_df["file_name"] = img_df["image_path"].apply(os.path.basename)
    img_df["sample_id"] = img_df["file_name"].apply(extract_sample_id_from_filename)
    img_df["view_id"] = img_df["file_name"].apply(extract_view_id_from_filename)

    merged_df = img_df.merge(metadata_df, on="sample_id", how="left")

    # 정렬
    merged_df = merged_df.sort_values(["sample_id", "view_id"]).reset_index(drop=True)

    return merged_df


class MangoImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_col="weight_g"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        target = float(row[self.target_col])

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "target": target,
            "sample_id": int(row["sample_id"]),
            "view_id": int(row["view_id"]),
            "file_name": row["file_name"]
        }
        return sample