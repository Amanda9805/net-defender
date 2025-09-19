# %%
# %%
import pandas as pd
import numpy as np
import torch
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# %%
# df = pd.read_csv('Datasets/UCL/uci.csv')
df = pd.read_csv('dataset/uci.csv')

df.drop(columns=['FILENAME', 'Domain', 'TLD', 'Title'], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# pd.set_option('display.max_columns', None)
# df.head()


# %%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# URL Preprocessing

class URLPreprocessor:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.char2idx = None
        self.idx2char = None

    def fit(self, urls):
        # Build character vocabulary
        all_text = ''.join(urls)
        vocab = sorted(list(set(all_text)))
        self.char2idx = {ch: idx+1 for idx, ch in enumerate(vocab)}  # 0 reserved for padding
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def url_to_seq(self, url):
        return [self.char2idx[ch] for ch in url if ch in self.char2idx]

    def pad_or_truncate(self, seq):
        seq = torch.tensor(seq, dtype=torch.long)
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        else:
            padding = torch.zeros(self.max_len - len(seq), dtype=torch.long)
            return torch.cat([seq, padding])

    def transform(self, urls):
        sequences = [self.pad_or_truncate(self.url_to_seq(url)) for url in urls]
        return torch.stack(sequences)  # shape: (num_samples, max_len)
    

# Numerical Feature Preprocessing

class NumericPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_numeric):
        self.scaler.fit(X_numeric)

    def transform(self, X_numeric):
        X_scaled = self.scaler.transform(X_numeric)
        return torch.tensor(X_scaled, dtype=torch.float)
    
    

def preprocess_df(df, url_col='URL', label_col='label'):
    # Separate features
    numeric_cols = [col for col in df.columns if col != url_col and col != label_col]
    X_numeric = df[numeric_cols].values
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    # Preprocess URLs
    url_processor = URLPreprocessor(max_len=200)
    url_processor.fit(df[url_col].tolist())
    X_url = url_processor.transform(df[url_col].tolist())

    # Preprocess numeric features
    numeric_processor = NumericPreprocessor()
    numeric_processor.fit(X_numeric)
    X_numeric_scaled = numeric_processor.transform(X_numeric)

    return X_url, X_numeric_scaled, y, url_processor, numeric_processor



# %%
X_url, X_numeric, y, url_proc, num_proc = preprocess_df(df)
print(X_url.shape, X_numeric.shape, y.shape)

# %%
# CNN-LSTM Model

class CNNLSTMPhishing(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_numeric_features,
                 cnn_out_channels=64, cnn_kernel_size=5, lstm_hidden_size=128,
                 lstm_num_layers=1, num_classes=2, dropout=0.5):
        super(CNNLSTMPhishing, self).__init__()

        # -------------------
        # Embedding layer for URL sequences
        # -------------------
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        
        # -------------------
        # 1D CNN to capture local patterns in URLs
        # -------------------
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=cnn_out_channels,
                               kernel_size=cnn_kernel_size,
                               padding=cnn_kernel_size//2)
        
        # -------------------
        # LSTM to capture sequential dependencies
        # -------------------
        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True)

        # -------------------
        # Dropout layer for regularization
        # -------------------
        self.dropout = nn.Dropout(dropout)

        # -------------------
        # Fully connected layer to combine LSTM output + numeric features
        # -------------------
        self.fc = nn.Linear(lstm_hidden_size*2 + num_numeric_features, num_classes)

    def forward(self, x_url, x_numeric):
        # x_url: (batch_size, seq_len)
        # x_numeric: (batch_size, num_numeric_features)

        # -------------------
        # Embed URL characters
        # -------------------
        x = self.embedding(x_url)  # (batch_size, seq_len, embedding_dim)

        # -------------------
        # CNN expects (batch_size, channels, seq_len)
        # -------------------
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = F.relu(self.conv1(x))  # (batch_size, cnn_out_channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, cnn_out_channels)

        # -------------------
        # Pass through LSTM
        # -------------------
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state from both directions
        lstm_out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, lstm_hidden_size*2)

        # -------------------
        # Concatenate numeric features
        # -------------------
        combined = torch.cat([lstm_out, x_numeric], dim=1)  # (batch_size, lstm_hidden_size*2 + num_numeric_features)
        combined = self.dropout(combined)

        # -------------------
        # Final classification
        # -------------------
        out = self.fc(combined)  # (batch_size, num_classes)
        return out

# %%
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f" Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth", device="cpu"):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f" Loaded checkpoint from epoch {checkpoint['epoch']} at {path}")
        return start_epoch
    else:
        print("No checkpoint found, starting fresh.")
        return 0


def save_final_model(model, path="final_model.pth"):
    torch.save(model.state_dict(), path)
    print(f" Final model saved at {path}")


def train_and_validate(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    num_epochs,
    device, 
    checkpoint_path, 
    save_every):
    
    model.to(device)

    # Resume if checkpoint exists
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    for epoch in range(start_epoch, num_epochs):
        # ---- Training ----
        model.train()
        total_loss = 0
        for x_url_batch, x_num_batch, y_batch in train_loader:
            x_url_batch, x_num_batch, y_batch = (
                x_url_batch.to(device), x_num_batch.to(device), y_batch.to(device)
            )

            optimizer.zero_grad()
            outputs = model(x_url_batch, x_num_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Evaluation ----
        y_true, y_pred, avg_val_loss = evaluate_model(model, val_loader, criterion, device)

        # ---- Metrics ----
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ---- Report ----
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss    : {avg_train_loss:.4f}")
        print(f"Validation Loss     : {avg_val_loss:.4f}")
        print(f"Accuracy      : {accuracy:.2f}%")
        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1-Score      : {f1:.4f}")

        # ---- Save checkpoint ----
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # ---- Save final model ----
    save_final_model(model, "final_model.pth")


def evaluate_model(model=None, model_path=None, test_loader=None, criterion=None, device="cpu"):
    """
    Evaluate model performance on given loader.
    - model: a PyTorch model object (if already in memory).
    - model_path: path to a saved model (.pth).
    - test_loader: DataLoader with test/val data.
    - criterion: loss function.
    """
    if model is None and model_path is None:
        raise ValueError(" Provide either a model instance or a model_path.")

    # Load model if only path is provided
    if model_path is not None and model is not None:
        print(" Both model and model_path provided. Using model_path.")
    
    if model_path is not None:
        # Instantiate a fresh model with the same architecture - must match how it was originally created
        model = CNNLSTMPhishing(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_numeric_features=num_numeric_features,
            num_classes=num_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" Loaded model from {model_path}")

    model.to(device)
    
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x_url_batch, x_num_batch, y_batch in test_loader:
            x_url_batch, x_num_batch, y_batch = (
                x_url_batch.to(device), x_num_batch.to(device), y_batch.to(device)
            )

            outputs = model(x_url_batch, x_num_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return np.array(all_labels), np.array(all_preds), avg_loss

def predict_single_url(url, model, url_processor, numeric_features=9, device="cpu"):
    """
    Predict if a single URL is phishing or not.
    - url: string, the URL to check
    - model: trained CNNLSTMPhishing model
    - url_processor: trained URLPreprocessor
    - numeric_features: number of numeric features expected by model
    - device: "cpu" or "cuda"
    """
    model.to(device)
    model.eval()

    # --- Preprocess URL ---
    X_url = url_processor.transform([url])  # shape: (1, seq_len)
    
    # --- Placeholder numeric features ---
    X_num = torch.zeros(1, numeric_features)  # shape: (1, num_numeric_features)

    X_url = X_url.to(device)
    X_num = X_num.to(device)

    with torch.no_grad():
        outputs = model(X_url, X_num)
        pred = torch.argmax(outputs, dim=1).item()

    # Convert to label
    label = "Phishing" if pred == 1 else "Safe"
    return label

# ------------------------
# Metrics
# ------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def display_metrics(y_true, y_pred, test_loss=None):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("\n===== Test Results =====")
    if test_loss is not None:
        print(f"Test Loss     : {test_loss:.4f}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

# %%
# Split 1: train vs (val+test)
X_url_train, X_url_temp, X_num_train, X_num_temp, y_train, y_temp = train_test_split(
    X_url, X_numeric, y, test_size=0.3, random_state=42, stratify=y
)

# Split 2: validation vs test
X_url_val, X_url_test, X_num_val, X_num_test, y_val, y_test = train_test_split(
    X_url_temp, X_num_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Convert numpy -> torch tensors
train_dataset = TensorDataset(X_url_train, X_num_train, y_train)
val_dataset   = TensorDataset(X_url_val, X_num_val, y_val)
test_dataset  = TensorDataset(X_url_test, X_num_test, y_test)


# DataLoader wraps data for batching + shuffling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# Hyperparameters
vocab_size = len(url_proc.char2idx) + 1
embedding_dim = 50
num_numeric_features = X_numeric.shape[1]
num_classes = 2

# Model
model = CNNLSTMPhishing(vocab_size, embedding_dim, num_numeric_features, num_classes=num_classes)

# Loss function - for classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer - best for text
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Train
train_and_validate(
    model, 
    train_loader,
    val_loader,
    criterion, 
    optimizer,
    num_epochs=5, 
    device="cpu", 
    checkpoint_path="checkpoint.pth", 
    save_every=1
)

# Evaluate (unseen data)
y_true, y_pred, test_loss = evaluate_model(
    test_loader,
    criterion, 
    device="cpu",
    model_path="final_model.pth")

# Metrics
display_metrics(y_true, y_pred, test_loss)


# %%



