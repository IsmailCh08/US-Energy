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
    
def train_lstm(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, lr=0.001):
    # loss function
    criterion = nn.MSELoss()
        
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # data loader
    train_dataset = torch.utils.data.TensorDataset(X_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # training loop
    for epoch in range(epochs):
        model.train()  # Set to training mode
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
        # Forward pass: make predictions
            output = model(batch_X)
            
        # Calculate loss: how wrong are we?
            loss = criterion(output, batch_y)
            
        # Backward pass: calculate gradients
            optimizer.zero_grad()
            loss.backward()
            
        # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
    # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.6f}")
    
    return model
    
def create_sequences(data, seq_length=24):
    X, y = [], []

    for i in range(len(data)-seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)