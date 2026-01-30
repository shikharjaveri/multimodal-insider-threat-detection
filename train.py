import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

# Dataset 
class UserAwareSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.sequences = []
        self.labels = []
        
        # Grouped by user to ensure sequences are logically connected
        for user, group in df.groupby("user"):
            features = group[feature_cols].values.astype(np.float32)
            targets = group[target_col].values.astype(np.float32)
            
            # Only created sequences if user had enough days of data
            if len(features) > seq_len:
                # Used sliding window logic
                for i in range(len(features) - seq_len):
                    self.sequences.append(features[i : i + seq_len])
                    self.labels.append(targets[i + seq_len])
                    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

# Large LSTM Model
class LargeLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        # Architecture from experiments.ipynb
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Taking the last time step
        last_step = out[:, -1, :]
        logits = self.fc(self.dropout(last_step))
        return logits 

# Training Logic
def main(args):
    # Load and Sort Data
    df = pd.read_csv(args.data)
    df['day'] = pd.to_datetime(df['day'])
    df = df.sort_values(['user', 'day']) 

    # Feature Selection
    feature_cols = ['logins', 'off_hour_logins', 'file_ops', 'usb_events', 'http_uploads', 'emails_sent']
    target_col = "suspicious"

    # Scaling
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Create User-Aware Sequences
    print(f"Generating sequences (Length: {args.seq_len})...")
    dataset = UserAwareSequenceDataset(df, feature_cols, target_col, args.seq_len)
    
    # Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Initialize Model & Weighted Loss
    model = LargeLSTMClassifier(len(feature_cols), hidden_dim=128, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Using pos_weight=21 because suspicious cases are rare (~4.4%)
    pos_weight = torch.tensor([21.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Starting training")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb).squeeze(), yb).item()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {v_loss/len(val_loader):.4f}")

        if (v_loss/len(val_loader)) < best_loss:
            best_loss = v_loss/len(val_loader)
            torch.save(model.state_dict(), args.model_out)
            np.save("scaler_scale.npy", scaler.scale_)
            np.save("scaler_min.npy", scaler.min_)


   # parser = argparse.ArgumentParser()
   # parser.add_argument("data", default="behavioral_features.csv")
   # parser.add_argument("epochs", type=int, default=5)
   # parser.add_argument("batch_size", type=int, default=2048) 
   # parser.add_argument("lr", type=float, default=0.001)      
   # parser.add_argument("seq_len", type=int, default=7)       
   # parser.add_argument("model_out", default="large_lstm_classifier.pt")
   # args = parser.parse_args()
   
