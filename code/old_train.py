
'''
""" this file train CNN 
CNN-based binary classifier for spindle detection using 1D convolution.
Trains on preprocessed window data: X (shape: N, channels, samples) and y (binary labels).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load data
X = np.load('data/X_windows.npy')
y = np.load('data/y_labels.npy')

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), batch_size=32
)

# CNN architecture
class SpindleCNN(nn.Module):
    def __init__(self):
        super(SpindleCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))

# Initialize model
model = SpindleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Evaluate helper
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    print(classification_report(all_labels, all_preds))

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    evaluate(model, val_loader)  
    ''' 

######

'''
# train_2dcnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# --- Load preprocessed EEG data ---
X = np.load('data/X_windows.npy')      # Shape: (N, 16, 512)
y = np.load('data/y_labels.npy')       # Shape: (N,)

# --- Reshape for 2D CNN ---
X = X[:, np.newaxis, :, :]  # Shape: (N, 1, 16, 512)

# --- Convert to torch tensors ---
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# --- Train-validation split ---
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# --- Dataloaders ---
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# --- Define 2D CNN Model ---
class Spindle2DCNN(nn.Module):
    def __init__(self):
        super(Spindle2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))

# --- Model, loss, optimizer ---
model = Spindle2DCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Evaluation helper ---
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    print(classification_report(all_labels, all_preds))

# --- Training loop ---
epochs = 10
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    evaluate(model, val_loader)
'''
# spindle_pipeline.py
"""
EEG Spindle Detection Pipeline using 2D CNN

"""

import mne
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from evaluate import *
# === CONFIG ===
EDF_PATH = Path("data/raw/P002_1_raw.edf")
JSON_PATH = Path("data/labels/sleep_block_spindle_output_P002_1.json")
SAVE_DIR = Path("data/windows/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 2.0
STEP_SEC = 1.0
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD EDF + FILTER ===
print("Loading EDF...")
raw = mne.io.read_raw_edf(EDF_PATH, preload=True)
sfreq = raw.info["sfreq"]

EEG_CHANNELS = ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4', 'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'F7', 'F8']
raw.pick_channels([ch for ch in EEG_CHANNELS if ch in raw.ch_names])
raw.filter(10.0, 15.0, fir_design='firwin', verbose=False)

# === LOAD SPINDLE LABELS ===
print("Loading JSON...")
with open(JSON_PATH) as f:
    spindle_data = json.load(f)
spindles = [(s["start"], s["end"]) for s in spindle_data["detected_spindles"]]
'''
def is_spindle(start, end):
    return any((max(start, s) < min(end, e)) for s, e in spindles)
'''

def is_spindle(start, end, overlap_threshold=0.7):
    window_duration = end - start
    for s_start, s_end in spindles:
        overlap = max(0, min(end, s_end) - max(start, s_start))
        if overlap / window_duration >= overlap_threshold:
            return True
    return False

def create_windows(start_sec, end_sec, prefix):    # creat overlapping windows
    print(f"Processing {prefix}...")
    X, y = [], []
    win = int(WINDOW_SEC * sfreq)  #no of sample in one window
    step = int(STEP_SEC * sfreq)
    start_samp = int(start_sec * sfreq)
    end_samp = int(end_sec * sfreq)

    for s in range(start_samp, end_samp - win, step):
        segment = raw.get_data(start=s, stop=s+win)
        t0 = s / sfreq     # convert back
        t1 = (s + win) / sfreq
        label = 1 if is_spindle(t0, t1,  overlap_threshold=0.5) else 0
        X.append(segment)
        y.append(label)

    X, y = np.array(X), np.array(y)
    np.save(SAVE_DIR / f"X_{prefix}.npy", X)
    np.save(SAVE_DIR / f"y_{prefix}.npy", y)
    print(f"Saved: X_{prefix}.npy ({X.shape})")

# === CREATE DATA SPLITS ===
create_windows(0, 3*3600, "train")
create_windows(3*3600, 6*3600, "test")
create_windows(6*3600, 8*3600, "val")

# === LOAD TRAINING DATA ===
X = np.load(SAVE_DIR / "X_train.npy")  # (N, C, T)
y = np.load(SAVE_DIR / "y_train.npy")  # (N,)
X = X[:, np.newaxis, :, :]             # (N, 1, C, T)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

print("Train label counts:", np.bincount(y_train.numpy().squeeze().astype(int)))
# checking becouse of division error ..... do i need class implance?????? train longer????
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

# === MODEL ===
class Spindle2DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, (3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),

            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
           # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = Spindle2DCNN().to(DEVICE)
# === Compute pos_weight ===
n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)
#try to fix value --- changes
#  BCEWithLogitsLoss (no sigmoid in o/p) ===
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === TRAINING ===
'''
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = torch.sigmoid_(model(x))
            pred = (out > 0.5).float()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    print(classification_report(y_true, y_pred, digits=4 , zero_division=0))  '''

print("\nTraining...")
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss={loss.item():.4f}")
    evaluate(model ,val_loader ,DEVICE)

# === SAVE MODEL ===
torch.save(model.state_dict(), "model/spindle_cnn2d.pth")
print("Model saved to model/spindle_cnn2d.pth")

# === TEST EVALUATION ===
X_test = np.load(SAVE_DIR / "X_test.npy")
y_test = np.load(SAVE_DIR / "y_test.npy")
X_test = X_test[:, np.newaxis, :, :]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

print("\nFinal test performance (3–6hr):")

evaluate(model, test_loader, DEVICE)
