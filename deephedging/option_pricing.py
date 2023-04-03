"""A module for option pricing."""

import torch


def get_european_call_price(x: torch.Tensor, strike: float) -> torch.Tensor:
    """
    Computes the price of a European call option.

    Args:
        x: Price vector of the underlying asset.
        strike: Strike price of the option.

    Returns:
        Price of the option.
    """
    return torch.max(x, torch.ones_like(x) * strike) - strike
