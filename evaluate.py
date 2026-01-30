import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

class UserAwareSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.sequences, self.labels = [], []
        for _, group in df.groupby("user"):
            feat = group[feature_cols].values.astype(np.float32)
            targ = group[target_col].values.astype(np.float32)
            if len(feat) > seq_len:
                for i in range(len(feat) - seq_len):
                    self.sequences.append(feat[i : i + seq_len])
                    self.labels.append(targ[i + seq_len])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

class LargeLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def main(args):
    df = pd.read_csv(args.data)
    df['day'] = pd.to_datetime(df['day'])
    df = df.sort_values(['user', 'day'])

    feature_cols = ['logins', 'off_hour_logins', 'file_ops', 'usb_events', 'http_uploads', 'emails_sent']
    
    scaler = MinMaxScaler()
    scaler.scale_ = np.load("scaler_scale.npy")
    scaler.min_ = np.load("scaler_min.npy")
    scaler.n_features_in_ = len(feature_cols)
    df[feature_cols] = scaler.transform(df[feature_cols])

    dataset = UserAwareSequenceDataset(df, feature_cols, "suspicious", args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = LargeLSTMClassifier(len(feature_cols))
    model.load_state_dict(torch.load(args.model))
    model.eval()

    y_true, y_scores = [], []
    with torch.no_grad():
        for xb, yb in loader:
            y_true.extend(yb.numpy())
            y_scores.extend(torch.sigmoid(model(xb).squeeze()).numpy())

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)

    print(f"Results:")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.4f}")


   # parser = argparse.ArgumentParser()
   # parser.add_argument("data", default="behavioral_features.csv")
   # parser.add_argument("model", default="large_lstm_classifier.pt")
   # parser.add_argument("batch_size", type=int, default=2048)
   # parser.add_argument("seq_len", type=int, default=7)
   # args = parser.parse_args()
 
