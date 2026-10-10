"""Student Performance Prediction - ML pipeline.
Run:  python pipeline.py            (uses data/student-mat.csv)
      python pipeline.py --data data/student-por.csv
"""
import argparse, os, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "G3"  # final grade (0-20)
OUT, MODELS = "outputs", "models"


def load_data(path):
    """UCI files are ';' separated."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:  # fallback if comma separated
        df = pd.read_csv(path)
    return df


def clean(df):
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    df = df.dropna(subset=[TARGET])  # target must exist
    return df


def eda(df):
    os.makedirs(OUT, exist_ok=True)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print(df[[TARGET, "G1", "G2", "studytime", "absences"]].describe().round(2))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(df[TARGET], bins=20, color="#4C78A8"); ax[0, 0].set_title("Final grade (G3) distribution")
    df.groupby("studytime")[TARGET].mean().plot.bar(ax=ax[0, 1], color="#F58518")
    ax[0, 1].set_title("Avg G3 by study time (1=<2h ... 4=>10h)")
    ax[1, 0].scatter(df["absences"], df[TARGET], alpha=.5, color="#54A24B")
    ax[1, 0].set_title("Absences vs G3"); ax[1, 0].set_xlabel("absences")
    ax[1, 1].scatter(df["G2"], df[TARGET], alpha=.5, color="#E45756")
    ax[1, 1].set_title("Previous grade G2 vs G3"); ax[1, 1].set_xlabel("G2")
    plt.tight_layout(); plt.savefig(f"{OUT}/eda_overview.png", dpi=130); plt.close()

    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(9, 7)); plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr)), corr.columns, fontsize=7)
    plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=130); plt.close()
    print("Top correlations with G3:\n", corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(6).round(3))


def build_preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def main(path):
    df = clean(load_data(path))
    eda(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=5.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", m)])
        pipe.fit(Xtr, ytr)
        pred = np.clip(pipe.predict(Xte), 0, 20)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        r2 = float(r2_score(yte, pred))
        cv = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
        rows.append({"Model": name, "RMSE": rmse, "R2": r2, "CV_R2": cv})
        fitted[name] = (pipe, pred)
        print(f"{name:20s} RMSE={rmse:.3f}  R2={r2:.3f}  CV R2={cv:.3f}")

    res = pd.DataFrame(rows).sort_values("RMSE").round(4)
    res.to_csv(f"{OUT}/model_comparison.csv", index=False)
    best = res.iloc[0]["Model"]
    pipe, pred = fitted[best]
    print("\nBest model:", best)

    plt.figure(figsize=(5.5, 5)); plt.scatter(yte, pred, alpha=.6)
    plt.plot([0, 20], [0, 20], "r--"); plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.title(f"{best}: actual vs predicted"); plt.tight_layout()
    plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=130); plt.close()

    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": X.columns.tolist(), "best": best,
                 "defaults": X.median(numeric_only=True).to_dict(),
                 "cats": {c: sorted(X[c].dropna().unique().tolist()) for c in X.select_dtypes(exclude="number")}},
                f"{MODELS}/best_model.joblib")
    print("Saved model + outputs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/student-mat.csv")
    main(ap.parse_args().data)
