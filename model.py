import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from scipy.special import comb


class CompositionalRecognizer(nn.Module):
    def __init__(self, class_count=10, hidden_size=512, dropout=0):
        super().__init__()
        self.class_count = class_count
        self.hidden_size = hidden_size
        self.comp_count = class_count**2 + int(comb(class_count, 2)) + class_count

        self.h0 = Parameter(torch.zeros((1, 1, hidden_size)))
        self.c0 = Parameter(torch.zeros((1, 1, hidden_size)))
        self.lstm = nn.LSTM(class_count, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, self.comp_count)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        h0 = self.h0.repeat_interleave(x.size(0), dim=1)
        c0 = self.c0.repeat_interleave(x.size(0), dim=1)
        ht = self.lstm(x.view(x.size(0), x.size(1), -1), (h0, c0))[0][:, -1, :]

        return self.fc(ht)

    def forward_loss(self, x, labels):
        result = self(x)

        return self.loss(result, labels)
