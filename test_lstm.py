import torch
import torch.nn as nn
import numpy as np

# simulate exactly your data shape
X = np.random.randn(116273, 24, 1).astype(np.float32)
y = np.random.randn(116273, 1).astype(np.float32)

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

lstm = nn.LSTM(1, 64, 2, batch_first=True)
dropout = nn.Dropout(0.2)
fc = nn.Linear(64, 1)

batch_X = X_t[:256]
print("running forward pass with real shape...")
out, _ = lstm(batch_X)
out = out[:, -1, :]
out = dropout(out)
out = fc(out)
print(f"done: {out.shape}")