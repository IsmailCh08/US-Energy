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
        # pass the input though the LSTM
        out, _ = self.lstm(x)

        # take only the last output
        out = out[:, -1, :]

        # apply dropout
        out = self.dropout(out)

        # convert to pred
        out = self.fc(out)

        return out