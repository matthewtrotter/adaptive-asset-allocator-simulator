from Backtester import Backtester
import datetime
from multiprocessing import Pool
import pandas as pd
import itertools

from assetuniverse import AssetUniverse, Asset
import metrics
from subportfolio import Subportfolio
from portfolio import Portfolio


start = datetime.datetime(1930, 7, 10)
end = datetime.datetime.today()

# Define asset universe
cashasset = Asset(start, end, None)
assets = [
    # Asset(start, end, 'FFIDX', display_name='US Stocks'),
    # Asset(start, end, 'FIEUX', display_name='European Stocks'),
    # Asset(start, end, 'FPBFX', display_name='Pacific Stocks'),
    # Asset(start, end, 'FOSFX', display_name='International Stocks'),
    # Asset(start, end, 'VUSTX', display_name='Long-Term Treasury Fund'),
    # Asset(start, end, 'SLSSX', display_name='International Bonds'),
    Asset(start, end, 'GLD', display_name='Gold'),
    Asset(start, end, 'SLV', display_name='Silver'),
    Asset(start, end, 'SPY', display_name='S&P 500'),
    Asset(start, end, 'TLT', display_name='30-year Treasury Bonds'),
    Asset(start, end, 'FXI', display_name='Chinese Stocks'),
    Asset(start, end, 'AAPL', display_name='Apple'),
    Asset(start, end, 'VHT', display_name='Healthcare'),
    Asset(start, end, 'VNQ', display_name='US Real Estate'),
    Asset(start, end, 'QQQ', display_name='Technology'),
    Asset(start, end, 'XLE', display_name='Energy Stocks'),
]

# assets = ["ABT", "ADP", "AXP", "DIS", "JNJ", "KO", "MMM", "TXN", "MSFT", "AAPL", "SBUX", "SPY", "MDY",
#         "TEPLX", "OPPAX", "MAPCX", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWW", "EWS", "EWU",
#         "VUSTX", "US Corp Bonds", "LBNDX",
#         "Gold", "Silver", "Palladium", "Platinum"]
leverage = 1.0      # Multiplier for portfolio of subportfolios (1.0 = no leverage)

# Define dates and lookbacks
lookbacks = [120, 180, 260,]
rebalance_period = 21
au = AssetUniverse(start, end, assets, cashasset)
au.download()
rebalance_dates = au.prices().index[max(lookbacks)::rebalance_period]

# Define subportfolio parameters
momentum_metrics = [
    metrics.total_return,
    metrics.sharpe_ratio,
    metrics.z_score
]
subportfolio_thresholds = [0.3, 0.5,]            # Keep the top %
subportfolio_min_keep = [2,]               # Keep at least this many assets
max_ind_allocations = [0.5, 0.75, 0.95]           # Allocate at most this % to each asset


# Create subportfolios
num_subportfolios = len(lookbacks)* \
                    len(momentum_metrics)* \
                    len(subportfolio_thresholds)* \
                    len(subportfolio_min_keep)* \
                    len(max_ind_allocations)

params = [p for p in itertools.product(
    lookbacks,
    momentum_metrics,
    subportfolio_thresholds,
    subportfolio_min_keep,
    max_ind_allocations
)
]

# with Pool() as p:
#     subportfolios = p.starmap(
subportfolios = [sp for sp in itertools.starmap(
        Subportfolio,
        zip(
            params,
            itertools.repeat(au),
            itertools.repeat(assets),
            range(1, len(params)+1),
            itertools.repeat(num_subportfolios),
            itertools.repeat(rebalance_dates)
        )
    )]

# Combine subportfolios
portfolio = Portfolio(subportfolios, leverage)

# Backtest
backtester = Backtester(portfolio, au, rebalance_dates)
backtester.backtest()

# Display
backtester.getstats()
backtester.plotlyplot()