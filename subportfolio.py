from typing import Dict, List
from assetuniverse import AssetUniverse
import numpy as np
import scipy.optimize as opt
import pandas as pd

class Subportfolio(object):
    def __init__(self, params: Dict, au: AssetUniverse, assets: pd.DataFrame, id: int, total_ids: int, rebalance_dates: pd.DatetimeIndex):
        self.lookback = params[0]
        self.momentum_metric = params[1]
        self.subportfolio_threshold = params[2]
        self.subportfolio_min_keep = params[3]
        self.max_ind_allocation = params[4]

        self.au = au
        self.rebalance_dates = rebalance_dates

        self.target_weights = self.set_target_weights()

        # Print status
        if id % 10 == 0 and id > 0:
            print(f'Finished subportfolio: {id} of {total_ids}')

    def set_target_weights(self):
        """Set the target weights on all rebalance dates
        """
        tickers = self.au.tickers(include_cash=False, include_borrow_rate=False)
        target_weights = self.au.prices(tickers=tickers).copy(deep=True)*0     # Reset to 0
        earliest_rebalance = self.au.prices(tickers=tickers).index[self.lookback + 1]
        for date_window_end in self.rebalance_dates:
            if date_window_end > earliest_rebalance:
                date_window_start = date_window_end - np.timedelta64(int(7.4*self.lookback/5), 'D')
                best_assets = self.sort_by_metric(date_window_start, date_window_end)
                target_weights.loc[date_window_end, :] = self.optimize_target_weights(
                    best_assets,
                    date_window_start, 
                    date_window_end
                )

        return target_weights

    def optimize_target_weights(self, assets: List, date_window_start, date_window_end):
        """Calculate the optimal target weights given the history of returns

        Parameters
        ----------
        returns : pd.DataFrame
            History of returns
        """
        # covmatrix = self.au.returns().loc[date_window_start:date_window_end, assets].cov().values
        covmatrix = self.au.covariance_matrix(assets, date_window_start, date_window_end)
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},           # sum of weights must match 1.0
            ]

        result = opt.minimize(
            fun=self._expected_variance,
            x0=np.ones(len(assets))/len(assets),
            args=covmatrix,
            method='SLSQP',
            constraints=cons,
            bounds=opt.Bounds(0, self.max_ind_allocation),
            tol=1e-13,
            options={'maxiter': 1000, 'disp': False}
        )
        optimized_weights = result.x.transpose()

        tickers = self.au.tickers(include_cash=False, include_borrow_rate=False)
        weights = pd.Series(
            data=np.zeros(len(tickers)),
            index=tickers
            )
        weights.loc[assets] = optimized_weights
        return weights

    def _expected_variance(self, weights, covmatrix):
        return np.matmul(weights, np.matmul(covmatrix, weights.T))

    def sort_by_metric(self, date_window_start, date_window_end) -> List:
        tickers = self.au.tickers(include_cash=False, include_borrow_rate=False)
        metric, ascending = self.momentum_metric(
            self.au.prices(tickers=tickers, start=date_window_start, end=date_window_end), 
            self.au.returns(tickers=tickers, start=date_window_start, end=date_window_end),
            self.lookback
            )
        metric = metric.sort_values(ascending=ascending)
        num_keep = int(np.ceil(max(self.subportfolio_threshold*len(tickers), self.subportfolio_min_keep)))
        return metric[-num_keep:].index.to_list()



class FixedAllocationSubportfolio(Subportfolio):
    def __init__(self, percentage_allocations: List, au: AssetUniverse, assets: pd.DataFrame, rebalance_dates: pd.DatetimeIndex):
        self.percentage_allocations = percentage_allocations
        self.au = au
        self.rebalance_dates = rebalance_dates
        self.target_weights = self.set_target_weights()

    def set_target_weights(self):
        tickers = self.au.tickers(include_cash=False, include_borrow_rate=False)
        target_weights = self.au.prices(tickers=tickers).copy(deep=True)*0     # Reset to 0
        for date_window_end in self.rebalance_dates:
            target_weights.loc[date_window_end, :] = self.percentage_allocations

        return target_weights
