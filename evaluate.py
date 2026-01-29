import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Utils
def compute_errors(model, loader):
    errors = []
    feature_errors = []
    with torch.no_grad():
        for (x,) in loader:
            recon = model(x)
            diff = (recon - x) ** 2
            batch_err = torch.mean(diff, dim=1)
            errors.extend(batch_err.numpy())
            feature_errors.append(diff.numpy())
    return np.array(errors), np.vstack(feature_errors)


# Eval
def main(args):
    df = pd.read_csv(args.data)

    # Keep only numeric columns and drop label
    X = df.select_dtypes(include=[np.number]).drop(columns=["suspicious"], errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    feature_names = X.columns.tolist()

    # Load scaler
    scaler = MinMaxScaler()
    scaler.scale_ = np.load(args.scaler_scale)
    scaler.min_ = np.load(args.scaler_min)
    scaler.data_min_ = np.zeros_like(scaler.scale_)
    scaler.data_max_ = np.ones_like(scaler.scale_)

    X_scaled = scaler.transform(X)

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = Autoencoder(X.shape[1])
    model.load_state_dict(torch.load(args.model))
    model.eval()

    errors, feature_errors = compute_errors(model, loader)

    # Percentile Threshold 
    threshold = np.percentile(errors, args.percentile)
    anomalies = errors > threshold

    print(f"Percentile threshold ({args.percentile}%): {threshold:.6f}")
    print(f"Anomalies: {anomalies.sum()} / {len(errors)} ({anomalies.mean()*100:.2f}%)")

    df["reconstruction_error"] = errors
    df["anomaly"] = anomalies
    df.to_csv(args.output, index=False)
    print(f"Saved row-level results to {args.output}")

    # ROC Curve 
    if "suspicious" in df.columns:
        y_true = df["suspicious"].values
        y_scores = errors

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
        plt.savefig("roc_curve.png")
        plt.close()

        print(f"ROC AUC: {roc_auc:.4f}")
        print("Saved ROC plot: roc_curve.png")

    # Per-user Scoring 
    if "user" in df.columns:
        user_summary = df.groupby("user").agg(
            mean_error=("reconstruction_error", "mean"),
            max_error=("reconstruction_error", "max"),
            anomaly_count=("anomaly", "sum"),
            total_records=("anomaly", "count")
        )
        user_summary["anomaly_rate"] = (
            user_summary["anomaly_count"] / user_summary["total_records"]
        )
        user_summary = user_summary.sort_values("anomaly_rate", ascending=False)
        user_summary.to_csv("user_anomaly_summary.csv")
        print("Saved per-user summary: user_anomaly_summary.csv")

    # Feature-wise Error 
    mean_feature_error = feature_errors.mean(axis=0)
    feature_error_df = pd.DataFrame({
        "feature": feature_names,
        "mean_reconstruction_error": mean_feature_error
    }).sort_values("mean_reconstruction_error", ascending=False)

    feature_error_df.to_csv("feature_error_summary.csv", index=False)
    print("Saved feature-wise error summary: feature_error_summary.csv")

    plt.figure(figsize=(10, 5))
    plt.bar(feature_error_df["feature"], feature_error_df["mean_reconstruction_error"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Mean Reconstruction Error per Feature")
    plt.xlabel("Feature")
    plt.ylabel("Mean Reconstruction Error")
    plt.tight_layout()
    plt.savefig("feature_error_bar.png")
    plt.close()

    # Plots 
    plt.figure()
    plt.hist(errors, bins=50)
    plt.axvline(threshold)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.savefig("error_histogram.png")
    plt.close()

    plt.figure()
    plt.scatter(range(len(errors)), errors)
    plt.axhline(threshold)
    plt.title("Reconstruction Error per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.savefig("error_scatter.png")
    plt.close()

    print("Saved plots: error_histogram.png, error_scatter.png, feature_error_bar.png")


#    parser = argparse.ArgumentParser()
#    parser.add_argument("data", default="behavioral_features.csv")
#    parser.add_argument("epochs", type=int, default=50)
#    parser.add_argument("batch_size", type=int, default=128)
#    parser.add_argument("lr", type=float, default=1e-3)
#    parser.add_argument("model_out", default="autoencoder.pt")
#    parser.add_argument("scaler_out", default="scaler_scale.npy")
#    parser.add_argument("min_out", default="scaler_min.npy")
#    args = parser.parse_args()
