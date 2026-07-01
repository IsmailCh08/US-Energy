import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# LSTM Model Architecture

class LTSMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LTSMModel, self).__init__() # load parent class
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2) # remove 20% of neurons
        self.fc = nn.Linear(hidden_size, output_size) # hidden size --> output size
    
    def forward(self,x):
        # take input and using LSTM create it into 64 hidden values
        out, _ = self.lstm(x)

        # Dont need all 24 hours, just current one to predict
        out = out[:, -1, :]

        # apply dropout to hidden values (so it doesnt overfit)
        out = self.dropout(out)

        # convert to pred
        out = self.fc(out)

        return out
    
def train_lstm(model, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss() # loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimize loss

    # Move data to the SAME device
    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    
    n_samples = X_train_t.shape[0]
    n_batches = n_samples // batch_size
    print(f"Training with {n_batches} batches per epoch")

    for epoch in range(epochs):
        model.train()
        # reset loss
        epoch_loss = 0

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            
            batch_X = X_train_t[start:end]
            batch_y = y_train_t[start:end]

            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {i}/{n_batches} | Loss: {loss.item():.6f}")

        print(f"Epoch {epoch+1} complete | Avg Loss: {epoch_loss/n_batches:.6f}")

    return model
    
def create_sequences(data, seq_length=24):
    X, y = [], []

    for i in range(len(data)-seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def prepare_lstm_data(df, target_col='PJME_MW', seq_length=24, test_ratio=0.2):
    # extract target values
    data = df[target_col].values.reshape(-1, 1)
    
    # scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # create sequences
    X, y = create_sequences(data_scaled.flatten(), seq_length)
    
    # reshape for lstm
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(-1, 1)
    
    # split chronologically
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"training sequences: {X_train.shape[0]}")
    print(f"test sequences: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler


def evaluate_lstm(model, X_test, y_test, scaler):
    device = next(model.parameters()).device  # Get the model's device
    model.eval()
    
    # Move test data to the SAME device
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_t).cpu().numpy()  # Back to CPU for numpy
    
    predictions = scaler.inverse_transform(predictions_scaled)
    y_test_original = scaler.inverse_transform(y_test)
    rmse = np.sqrt(np.mean((predictions.flatten() - y_test_original.flatten()) ** 2))
    
    return predictions, y_test_original, rmse