"""A module for option pricing."""

import torch
import numpy as np
from scipy import stats


class BlackScholesCallOption:
    """Class representing a call option in the Black-Scholes model."""

    def __init__(self, price, strike, r, dividend, time_to_T, sigma):
        """
        Initializes a Black-Scholes call option.

        Args:
            price: Price of the underlying asset.
            strike: Strike price of the option.
            r: Risk-free interest rate.
            dividend: Dividend yield.
            time_to_T: Time to maturity.
            sigma: Volatility.
        """
        self.price = price
        self.strike = strike
        self.r = r
        self.dividend = dividend
        self.time_to_T = time_to_T
        self.sigma = sigma

        self.d1 = (
            np.log(self.price / self.strike)
            + (self.r - self.dividend + 0.5 * self.sigma**2) * self.time_to_T
        ) / (self.sigma * np.sqrt(self.time_to_T))

    def get_price(self):
        """
        Computes the price of the option.

        Returns:
            Price of the option.
        """
        call_price = self.price * np.exp(
            -self.dividend * self.time_to_T
        ) * stats.norm.cdf(self.d1) - self.strike * np.exp(
            -self.r * self.time_to_T
        ) * stats.norm.cdf(
            self.d1 - np.sqrt(self.time_to_T) * self.sigma
        )

        return call_price

    def get_delta(self):
        """
        Computes the delta of the option.

        Returns:
            Delta of the option.
        """
        delta = np.exp(-self.dividend * self.time_to_T) * stats.norm.cdf(
            self.d1
        )
        return delta


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
