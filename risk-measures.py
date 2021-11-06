# File that contains risk measures classes
import tensorflow.keras.backend as ktf

def CVaR(x = None, w = None, alpha = None):
    """Conditional Value at Risk"""
    return ktf.mean(w + (ktf.maximum(-x-w, 0)/(1.0 - alpha)))

def Entropy(x = None, _lambda = None):
    """Exponential risk measure (Entropy)"""
    return (1/_lambda) * ktf.log(ktf.mean(ktf.exp(-_lambda * x)))