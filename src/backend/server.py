# server.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
import random
import numpy as np
import pickle
from sklearn.discriminant_analysis import StandardScaler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


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
    

class NumericPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_numeric):
        self.scaler.fit(X_numeric)

    def transform(self, X_numeric):
        X_scaled = self.scaler.transform(X_numeric)
        return torch.tensor(X_scaled, dtype=torch.float)


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
    

def predict_single_url(url, model, url_processor, numeric_features=50, device="cpu"):
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


app = FastAPI()

# Allow cross-origin requests (React frontend → FastAPI backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for React dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

with open("models/model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

vocab_size = metadata["vocab_size"]
embedding_dim = metadata["embedding_dim"]
num_numeric_features = metadata["num_numeric_features"]
char2idx = metadata["char2idx"]
max_len = metadata["max_len"]

# Rebuild URLPreprocessor with same vocab
url_processor = URLPreprocessor(max_len=max_len)
url_processor.char2idx = char2idx
url_processor.idx2char = {idx: ch for ch, idx in char2idx.items()}

# Rebuild model
model = CNNLSTMPhishing(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_numeric_features=num_numeric_features,
    num_classes=2
)
model.load_state_dict(torch.load("models/final_modelA.pth", map_location="cpu"))
model.eval()


@app.post("/predict")
def predict(request: URLRequest):
    url = request.url
    
    result = predict_single_url(url, model, url_processor)
    return {"url": url, "is_phishing": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
