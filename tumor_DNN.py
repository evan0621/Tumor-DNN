import torch
import torch.nn as nn


class USDNN(nn.Module):
    def __init__(self):
        super(USDNN, self).__init__()

        self.dnn = nn.Sequential(
            nn.Linear(in_features=17, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.dnn(x)
        return output
