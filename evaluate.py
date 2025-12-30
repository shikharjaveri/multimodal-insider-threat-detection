import torch
import numpy as np
from sklearn.metrics import classification_report
from models.lstm_model import InsiderLSTM

# load test arrays saved during training (or regenerate)
# you can reuse the train.py code for loading

# load model
model = InsiderLSTM()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# compute predictions and print report
# (same logic as Kaggle)
