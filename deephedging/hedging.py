"""A module that contains the hedging strategies."""

import torch
import torch.nn as nn


class FeedForwardHedgingStrategy(nn.Module):
    """
    A class that implements the simple hedging strategy. It consists of a
    feedforward neural networks for each timestep.
    """

    def __init__(self, hidden_nodes, num_timesteps):
        """Initialize the hedging strategy."""
        super(FeedForwardHedgingStrategy, self).__init__()

        self.steps = nn.ModuleList()
        for _ in range(num_timesteps):
            self.steps.append(
                nn.Sequential(
                    nn.Linear(1, hidden_nodes),
                    nn.ReLU(),
                    nn.Linear(hidden_nodes, hidden_nodes),
                    nn.ReLU(),
                    nn.Linear(hidden_nodes, 1),
                )
            )

    def forward(self, x):
        """Forward pass of the hedging strategy."""
        dS = torch.diff(x)
        hedge = torch.zeros_like(dS)
        hedgingPnL = torch.zeros_like(dS[:, 0])

        for i in range(len(self.steps)):
            logS = torch.log((x[:, i] / x[:, 0]))  # Normalisation
            H_i = self.steps[i](logS.unsqueeze(-1))
            hedgingPnL += dS[:, i] * H_i.flatten()
            hedge[:, i] = H_i.flatten()
        return hedgingPnL, hedge
