import torch
import numpy as np
from sklearn.metrics import classification_report
from models.lstm_model import InsiderLSTM

# load model
model = InsiderLSTM()
model.load_state_dict(torch.load("model.pt"))
model.eval()
