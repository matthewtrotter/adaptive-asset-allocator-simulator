import pandas as pd

def total_return(prices: pd.DataFrame, returns: pd.DataFrame, lookback: int):
    """Calculate the total return over the time period.

    Returns
    -------
    metric
        The computed metric values for each asset.
    ascending: bool
        True means "higher numbers are better", otherwise False.
    """
    metric = prices.iloc[-1] - prices.iloc[0]
    ascending = True
    return metric, ascending

def sharpe_ratio(prices: pd.DataFrame, returns: pd.DataFrame, lookback: int):
    """Calculate the Sharpe ratio over the time period.

    Returns
    -------
    metric
        The computed metric values for each asset.
    ascending: bool
        True means "higher numbers are better", otherwise False.
    """
    total_return = prices.iloc[-1] - prices.iloc[0]
    stddev = returns.std()
    metric = total_return/stddev
    ascending = True
    return metric, ascending

def z_score(prices: pd.DataFrame, returns: pd.DataFrame, lookback: int):
    """Calculate the Z-score over the time period.

    Returns
    -------
    metric
        The computed metric values for each asset.
    ascending: bool
        True means "higher numbers are better", otherwise False.
    """
    avg = prices.mean()
    stddev = returns.std()
    metric = (prices.iloc[-1] - avg)/stddev
    ascending = True
    return metric, ascending

