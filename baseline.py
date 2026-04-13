import pandas as pd
import numpy as np
import re

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. 데이터 로드
# =========================
xlsx_path = "Physical_properties_Alphonso_Images.xlsx"

raw = pd.read_excel(xlsx_path, header=None)

df = raw.iloc[1:].reset_index(drop=True)
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)
df.columns = [str(c).strip() for c in df.columns]

df = df[["Sample No", "fruit diameter (mm)", "width across schoulder (mm)", "Actual Weight (gms)"]].copy()

df = df.rename(columns={
    "Sample No": "sample_no",
    "fruit diameter (mm)": "diameter_mm",
    "width across schoulder (mm)": "shoulder_width_mm",
    "Actual Weight (gms)": "weight_g"
})

for col in ["diameter_mm", "shoulder_width_mm", "weight_g"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def normalize_sample_id(x):
    x = str(x).strip()
    m = re.search(r'AW0*([0-9]+)', x)
    return int(m.group(1)) if m else None

df["sample_id"] = df["sample_no"].apply(normalize_sample_id)

print("Data shape:", df.shape)
print(df.head())

# =========================
# 2. 입력 / 타깃
# =========================
feature_cols = ["diameter_mm", "shoulder_width_mm"]
target_col = "weight_g"

X = df[feature_cols]
y = df[target_col]

# =========================
# 3. sample 기준 fold 생성
# =========================
# baseline은 행=sample 1개라서 일반 KFold로 sample fold를 만들면 됨
sample_df = df[["sample_id"]].drop_duplicates().sort_values("sample_id").reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(sample_df), start=1):
    train_sample_ids = sample_df.iloc[train_idx]["sample_id"].tolist()
    val_sample_ids = sample_df.iloc[val_idx]["sample_id"].tolist()

    train_row_idx = df.index[df["sample_id"].isin(train_sample_ids)].tolist()
    val_row_idx = df.index[df["sample_id"].isin(val_sample_ids)].tolist()

    folds.append((train_row_idx, val_row_idx))

    print(f"Fold {fold_idx}: train={len(train_row_idx)}, val={len(val_row_idx)}")

# 이제 folds를 baseline과 CNN이 같이 쓸 수 있음

# =========================
# 4. 평가 지표
# =========================
def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scoring = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse_func, greater_is_better=False),
    "r2": make_scorer(r2_score)
}

# =========================
# 5. 모델 정의
# =========================
models = {
    "LinearRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])
}

# =========================
# 6. fold 공유 방식으로 평가
# =========================
results = []

for model_name, model in models.items():
    cv_result = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=folds,   # 여기 핵심: 미리 만든 sample 기준 fold 사용
        scoring=scoring,
        return_train_score=False
    )

    results.append({
        "model": model_name,
        "MAE_mean": -cv_result["test_mae"].mean(),
        "MAE_std": cv_result["test_mae"].std(),
        "RMSE_mean": -cv_result["test_rmse"].mean(),
        "RMSE_std": cv_result["test_rmse"].std(),
        "R2_mean": cv_result["test_r2"].mean(),
        "R2_std": cv_result["test_r2"].std()
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE_mean")
print("\nBaseline Results")
print(results_df)