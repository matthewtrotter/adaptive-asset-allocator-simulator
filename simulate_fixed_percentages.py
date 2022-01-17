from Backtester import Backtester
import datetime
from multiprocessing import Pool
import pandas as pd
import itertools

from assetuniverse import AssetUniverse, Asset
import metrics
from subportfolio import Subportfolio, FixedAllocationSubportfolio
from portfolio import Portfolio


start = datetime.datetime(1930, 7, 10)
end = datetime.datetime.today()

# Define asset universe
cashasset = Asset(start, end, None)
assets = [
    Asset(start, end, 'FFIDX', display_name='US Stocks'),
    Asset(start, end, 'FIEUX', display_name='European Stocks'),
    Asset(start, end, 'FPBFX', display_name='Pacific Stocks'),
    # Asset(start, end, 'FOSFX', display_name='International Stocks'),
    Asset(start, end, 'VUSTX', display_name='Long-Term Treasury Fund'),
    # Asset(start, end, 'SLSSX', display_name='International Bonds'),
    Asset(start, end, 'GC=F', display_name='Gold'),
    # Asset(start, end, 'GC=F', display_name='Gold'),
    # Asset(start, end, 'GC=F', display_name='Gold'),
    # Asset(start, end, 'GC=F', display_name='Gold'),
]

percentage_allocations = [0.4, 0.1, 0.1, 0.3, 0.1]

leverage = 1.0      # Multiplier for portfolio of subportfolios (1.0 = no leverage)

# Define dates and lookbacks
rebalance_period = 21
au = AssetUniverse(start, end, assets, cashasset)
au.download()
rebalance_dates = au.prices().index[1::rebalance_period]

# Define subportfolio parameters
momentum_metrics = [
    metrics.total_return,
    metrics.sharpe_ratio,
    metrics.z_score
]
subportfolio_thresholds = [0.5, 1.0]            # Keep the top %
subportfolio_min_keep = [2,]               # Keep at least this many assets
max_ind_allocations = [0.95]           # Allocate at most this % to each asset


# Create subportfolio
subportfolios = [FixedAllocationSubportfolio(percentage_allocations, au, assets, rebalance_dates)]

# Combine subportfolios
portfolio = Portfolio(subportfolios, leverage)

# Backtest
backtester = Backtester(portfolio, au, rebalance_dates)
backtester.backtest()

# Display
backtester.getstats()
backtester.plotlyplot()