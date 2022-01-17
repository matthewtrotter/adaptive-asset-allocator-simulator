#!/usr/bin/env python

"""
Copyright (C) Matthew Trotter - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Written by Matthew Trotter, 2017.

Backtester Class Definition

"""

from portfolio import Portfolio
import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
from assetuniverse import AssetUniverse

class Backtester:
    # Backtester object runs a full backtest using the assigned returns. This backtester computes
    # the NAV and statistics.

    def __init__(self, portfolio: Portfolio, au: AssetUniverse, rebalance_dates: pd.DatetimeIndex):
        self._assets = au.tickers(include_cash=False, include_borrow_rate=False)
        self._nonassets = ["Cash", "Equity", "Loan Balance", "Daily Borrow Rate", "NAV"]
        state_columns = self._assets + self._nonassets
        self._state = DataFrame(data=np.zeros((au.prices().shape[0], len(state_columns))), columns=state_columns, index=au.prices().index)
        self._state["NAV"].iloc[0] = 1.0        # Starting NAV
        rborrow = au.borrowrate.prices
        self._state.loc[self._state.index[1:], "Daily Borrow Rate"] = (1 + rborrow/100).pow(1/252) - 1

        first_date = self._state.index[0]
        first_asset_returns = pd.DataFrame(data=np.zeros((1, len(self._assets))), index=[first_date], columns=self._assets)
        self.asset_returns = pd.concat([first_asset_returns, au.returns(self._assets)])
        first_cash_returns = pd.DataFrame(data=[0], index=[first_date], columns=[au.cashasset.ticker])
        self.cash_returns = pd.concat([first_cash_returns, au.returns(['Cash'])])

        self.au = au
        self.rebalance_dates = rebalance_dates
        self.portfolio = portfolio


    def backtest(self):
        # After all parameters have been set, this runs a full backtest
        # Calculate the NAV series and returns over all bars
        self.rebalanceEvent(self._state.index[0])      # rebalance before starting backtest
        for n, i in enumerate(self._state.index[1:]):
            self.calculateBar(i)

            if i in self.rebalance_dates:
                # if current bar is a rebalance event
                self.rebalanceEvent(i)

                # update progress
                prog = 100*n/(self.au.cashasset.returns.shape[0]-1)
                if int(divmod(prog, 10)[1]) == 0:
                    progstr = '[' + '='*round(prog/10) + ' '*round(10-prog/10) + '] Backtesting'
                    sys.stdout.write('\r'+progstr)
                    sys.stdout.flush()

        # calculate daily NAV return
        self.NAV = self._state.loc[:, "NAV"].to_numpy()
        temp = np.roll(self.NAV, 1)
        temp[0] = self.NAV[0]
        self.NAVr = self.NAV/temp - 1.0

         # calculate actual weights
        self.aw = self._state[self._assets + ["Cash"]].divide(self._state["NAV"], axis="index")

        progstr = '[' + '='*10 + '] Backtesting'
        sys.stdout.write('\r'+progstr + '\n')

    def rebalanceEvent(self, i):
        # Rebalance to target weights for all assets in portfolio
        self._state.loc[i, self._assets] = self.portfolio.target_weights.loc[i, :]*self._state.loc[i, "NAV"]                        # Calculate new position values based on target weights and current NAV
        self._state.loc[i, "Cash"] = self.portfolio.cash_weight.loc[i]*self._state.loc[i, "NAV"]            # Calculate new cash position
        self._state.loc[i, "Equity"] = self._state.loc[i, self._assets].sum() + self._state.loc[i, "Cash"]      # Calculate new total equity
        self._state.loc[i, "Loan Balance"] = self._state.loc[i, "Equity"] - self._state.loc[i, "NAV"]           # Calculate new margin loan balance

    def calculateBar(self, i):
        # Calculate the new prices of all asset allocations, Equity, NAV, and loan balance
        index_of_prev_i = self._state.index.get_loc(i) - 1
        prev_i = self._state.index[index_of_prev_i]
        self._state.loc[i, self._assets] = self._state.loc[prev_i, self._assets]*(1.0 + self.asset_returns.loc[i, :])        # update equity value of each individual asset
        self._state.loc[i, "Cash"] = self._state.loc[prev_i, "Cash"]*(1.0 + self.cash_returns.loc[i, "Cash"])  # update cash value
        self._state.loc[i, "Equity"] = self._state.loc[i, self._assets+["Cash",]].sum()      # sum position values and cash values to get total equity
        self._state.loc[i, "Loan Balance"] = self._state.loc[prev_i, "Loan Balance"]*(1.0 + self._state.loc[i, "Daily Borrow Rate"])                # update the loan balance
        self._state.loc[i, "NAV"] = self._state.loc[i, "Equity"] - self._state.loc[i, "Loan Balance"]

    def getstats(self):
        # Calculate CAGR, annual volatility, Sharpe Ratio, and maximum drawdown of portfolio
        firstDateIndex = np.argwhere(self._state.index == self.rebalance_dates[0])[0][0]    # first rebalance date
        TR = self.NAV[-1]/self.NAV[firstDateIndex] - 1
        actual_start_date = self.au.cashasset.returns.axes[0][firstDateIndex]
        actual_end_date = self.au.cashasset.returns.axes[0][-1]
        Y = (actual_end_date - actual_start_date).days/365.25

        CAGR = (1+TR)**(1/Y) - 1
        aVol = math.sqrt(252)*self.NAVr[firstDateIndex:-1].std()
        Sharpe = CAGR/aVol

        dim = self.NAV.size
        maxNAV = self.NAV[0]
        DD = np.zeros(dim)
        for i in range(0,dim):
            maxNAV = max(self.NAV[i], maxNAV)
            DD[i] = self.NAV[i]/maxNAV - 1
        maxDD = min(DD)

        self.stats = [CAGR, aVol, Sharpe, maxDD]

        print('Backtest performance stats:')
        print('CAGR         = %3.1f' % (100*self.stats[0]))
        print('Annual Vol.  = %3.1f' % (100*self.stats[1]))
        print('Sharpe Ratio = %3.2f' % self.stats[2])
        print('Max Drawdown = %3.1f' % (100*self.stats[3]))
        print('')


    def plot(self):
        # Plot performance charts of NAV and individual assets
        plt.figure()
        plt.subplot(2,1,1)
        plt.semilogy(self.au.cashasset.prices.axes[0], self.NAV, linewidth=2.0)
        plt.gca().set_prop_cycle(None)
        plt.hold(True)
        plottickers = self.au.tickers(include_borrow_rate=False)
        prices = self.au.prices(plottickers)
        plt.semilogy(prices, alpha=0.15)
        plt.grid(True)
        ymin = 0.9*min(min(self.NAV), prices.min().min())
        ymax = 1.1*max(max(self.NAV), prices.max().max())
        plt.ylim([ymin, ymax])
        plt.xlabel('Date')
        plt.ylabel('NAV')
        plt.title('Backtest Results.\nNAV Growth and Asset Weights')

        plt.subplot(2,1,2)
        plt.stackplot(self.aw.index, self.aw.values.T.tolist())
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.grid(True)
        plt.title('Actual Allocation Weights')
        plt.draw()
        #sym = list(self.r.columns.values)
        #plt.legend(sym, loc='best')


    def plotlyplot(self):
        import plotly.graph_objs as go

        # Plot actual weights over time
        fig2 = go.Figure()
        for col in self.aw.columns:
            fig2.add_trace(go.Scatter(x=self.aw.index, y=100*self.aw[col], name=col, mode="lines", stackgroup="one"))
        fig2.show()

        # Plot asset prices and NAV over time
        plottickers = self.au.tickers(include_cash=False, include_borrow_rate=False)
        prices = self.au.prices(plottickers)
        data1 = [{'x': prices.index, 'y': prices[col], 'name': col, 'line': {'width': 1}
                } for col in prices.columns]
        data1.append({'x': self.au.cashasset.prices.index, 'y': self.au.cashasset.prices_normalized.values, 'name': self.au.cashasset.display_name,
                     'line': {'color': 'green', 'width': 1}})
        data1.append({'x': prices.index, 'y': self.NAV, 'name': 'NAV',
                     'line': {'color': 'purple', 'width': 5}})

        layout1 = go.Layout(title='Adaptive Asset Allocation Simulation', xaxis={'title': 'Date'},
                           yaxis={'title': 'Value', 'type': 'log', 'autorange': True},
                           width=1000, height=700)

        fig1 = go.Figure(data=data1, layout=layout1)
        fig1.show()

        
        #data2 = [go.Bar(name=col, x=self.aw.index, y=100*self.aw[col]) for col in self.aw.columns]
        #data = [{'x': self.aw.index, 'y': 100*self.aw[col], 'name': col, 'type': 'bar'
        #         } for col in self.aw.columns]

        #layout2 = go.Layout(barmode='stack', bargap=0, bargroupgap=0, title='Actual Weights', xaxis={'title': 'Date'},
        #                   yaxis={'title': 'Weight (%)'},
        #                   width=1000, height=700)

        #fig2 = go.Figure(data=data2, layout=layout2)
        #fig2.show()

