#!/usr/bin/env python

"""
Copyright (C) Matthew Trotter - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Written by Matthew Trotter, 2017.

Optimizer Class Definition

"""

import sys
import scipy.optimize as so
import numpy as np
import datetime
import math
from pandas import DataFrame
import pandas as pd
from multiprocessing import Pool


class Optimizer:
    # The Optimizer object calculates target weights for a backtest.
    """
    filtertype: top rank
        The optimizer will keep the top x percent of assets filtered by the metric where x is keeppercent.
    filtertype: bottom rank
        The optimizer will keep the bottom x percent of assets filtered by the metric where x is keeppercent.
    filtertype: top threshold
        The optimizer will keep all assets whose metric is above the threshold.
    filtertype: bottom threshold
        The optimizer will keep all assets whose metric is below the threshold.
    filtertype: top threshold+rank
        The optimizer will first filter out assets whose metrics are below the threshold. Then, the optimizer will keep
        only the top x percent of the remaining assets where x is keeppercent.
    filtertype: bottom threshold+rank
        The optimizer will first filter out assets whose metrics are above the threshold. Then, the optimizer will keep
        only the bottom x percent of the remaining assets where x is keeppercent.
    """

    def __init__(self, name, threshold, keeppercent, filtertype, m, lb, AU, algo, c, rebbars, estimator, mcstat,
                 mcparams):
        self.name = name                    # name of optimizer (e.g. 'Opt 1')
        self.threshold = threshold            # threshold - see above
        self.keeppercent = keeppercent
        self.filtertype = filtertype            # type of performance filter, 'top rank', 'bottom rank', 'top threshold', 'bottom threshold', or 'top threshold+rank', or 'bottom threshold+rank'
        self.m = self.backfillDF(m)             # metric result for each asset
        self.lb = lb            # lookback period
        self.AU = AU            # Asset Universe
        self.numassets = len(AU.r.columns)
        #self.sym = AU.sym       # symbols of asset universe
        # self.r = r.copy()           # returns table of asset universe
        # self.p = p.copy()
        self.tw = None# 0*AU.r.copy()        # target weights of optimizer
        self.cashw = None# 1.0 + 0*AU.r.iloc[:,1].copy()  # cash weight
        self.algo = algo        # optimization algorithm ('Min. Var.', 'Tar. Var.')
        self.c = c              # constraints for optimizer [min ind. weight, max ind. weight, min total weight, max total weight, target vol.]
        self.rebbars = rebbars  # DataFrame of rebalance bars (True/False) indicators
        self.rebbars_indices = np.where(self.rebbars.to_numpy())[0]
        self.estimator = estimator  # type of statistical estimator to use, sample or perfect
        self.mcstat = mcstat    # Statistic to randomize during monte carlo analysis
        self.mcparams = mcparams    # Random values to modify std devs and correlations of covariance during monte carlo

        first_date = self.AU.p.index[0]
        first_asset_returns = pd.DataFrame(data=np.zeros((1, self.numassets)), index=[first_date], columns=self.AU.p.columns)
        self.asset_returns = pd.concat([first_asset_returns, self.AU.r])

    def simulate(self):
        # simulate between start and end dates. Calculate the target weights for all assets.
        print('Optimizing ' + self.name + ' target weights... ', end="", flush=True)

        # with Pool() as p:
        targetweights = list(map(self.get_target_weights, range(self.AU.p.shape[0])))
        self.tw = DataFrame(data=targetweights, index=self.AU.p.index, columns=self.AU.p.columns)

        self.cashw = 1.0 - self.tw.sum(axis=1)
        self.cashw.loc[self.cashw < 0] = 0
        #self.tw = self.backfillDF(self.tw)

        print("Done.", flush=True)

    def get_target_weights(self, i):
        # return the weights array for this bar
        target_weights = np.zeros(self.numassets)
        if i in self.rebbars_indices:
            relevant_assets = self.relevantAssets(i)
            covmatrix = self.asset_returns[relevant_assets].iloc[(i-self.lb):i].cov().to_numpy()
            if relevant_assets.empty or np.any(np.isnan(covmatrix)):
                return target_weights

            relevant_indices = [self.asset_returns.columns.get_loc(asset) for asset in relevant_assets.to_list()]
            lower_bounds = self.c[0]*np.ones(len(relevant_indices))
            upper_bounds = self.c[1]*np.ones(len(relevant_indices))
            min_total_weight = self.c[2]
            max_total_weight = self.c[3]

            min_var_weights =  self.minVarOptimization_new(covmatrix, lower_bounds, upper_bounds)

            expected_daily_variance = self.expVarFun(min_var_weights, covmatrix)
            expected_annual_volatility = np.sqrt(252)*np.sqrt(expected_daily_variance)
            target_annual_volatility = self.c[4]
            temp_weights = (target_annual_volatility/expected_annual_volatility)*min_var_weights
            temp_sum = np.sum(temp_weights)
            if temp_sum > max_total_weight:
                temp_weights = (max_total_weight/temp_sum)*temp_weights
            elif temp_sum < min_total_weight:
                temp_weights = (min_total_weight/temp_sum)*temp_weights
            target_weights[relevant_indices] = temp_weights

        return target_weights


    def runStats(self, i, operation):
        # compute covariance matrix for relevant assets according to metric and threshold
        ri = self.relevantAssets(i)
        if ri.size > 0:
            if self.estimator == 'none':
                return [ri, 0, 0]
            if self.estimator == 'sample':
                x = self.asset_returns[ri].iloc[(i-self.lb):i]
                if operation == 'correlation':
                    o = x.corr()
                else:
                    o = x.cov()

                return [ri, o, x.mean()]

            elif self.estimator == 'perfect':
                x = self.asset_returns[ri].iloc[i:(i+self.lb)]
                if operation == 'correlation':
                    o = x.corr()
                else:
                    o = x.cov()

                return [ri, o, x.mean()]

            elif self.estimator == 'perfect + random':
                x = self.asset_returns[ri].iloc[i:(i + self.lb)]
                c = self.randmodify(x)     # Randomly modify true covariance
                return [ri, c, x.mean()]

        else:
            return [ri, 0, 0]

    def relevantAssets(self, i):
        # returns the symbol indices that correspond to the relevant assets for the specified bar

        sortedindices = self.m.iloc[i, :].sort_values()
        if self.filtertype == 'top rank':
            # keep the top keeppercent of assets.
            n = int(self.keeppercent*len(self.m.columns))
            ri = sortedindices.iloc[-n:].index
        elif self.filtertype == 'bottom rank':
            # keep the bottom keeppercent of assets.
            num_assets = len(sortedindices.index)
            n = int(self.keeppercent*num_assets)
            ind = -(num_assets-n)
            if ind == 0:
                ri = sortedindices.index
            else:
                ri = sortedindices.iloc[:ind].index
        elif self.filtertype == 'top threshold':
            # keep all assets whose metric is above or equal to the threshold
            sortedindices = sortedindices[sortedindices >= self.threshold]
            ri = sortedindices.index
        elif self.filtertype == 'bottom threshold':
            # keep all assets whose metric is below the threshold
            sortedindices = sortedindices[sortedindices < self.threshold]
            ri = sortedindices.index
        elif self.filtertype == 'top threshold+rank':
            # keep the top keeppercent of assets whose metric is above or equal to the threshold
            sortedindices = sortedindices[sortedindices >= self.threshold]
            n = int(self.keeppercent*len(sortedindices.index))
            ri = sortedindices.iloc[-n:].index
        elif self.filtertype == 'bottom threshold+rank':
            # keep the bottom keeppercent of assets whose metric is below the threshold
            sortedindices = sortedindices[sortedindices < self.threshold]
            num_assets_remaining = len(sortedindices.index)
            n = int(self.keeppercent*num_assets_remaining)
            ind = -(num_assets_remaining-n)
            if ind == 0:
                ri = sortedindices.index
            else:
                ri = sortedindices.iloc[:ind].index

        return ri

    def randmodify(self, x):
        # modify the covariance matrix. Multiply std devs by random values.
        # Add correlations to random values.

        c = x.cov()     # Covariance matrix
        n = c.shape[0]      # Size of covariance matrix

        if self.mcstat == 'std devs' or self.mcstat == 'both':
            r = 2*self.mcparams[0]*(np.random.rand(n)-0.5) + 1.0     # random values from U[1-x, 1+x]
            a = np.zeros((n,n))
            di = np.diag_indices(n)
            a[di] = r           # matrix with random diagonal entries from U[1-x, 1+x]
            temp = np.dot(a, c)
            temp = np.dot(temp, a) # random std devs by multiplying matrices a*c*a
            c.iloc[:,:] = temp        # Assign values to DataFrame covariance matrix

        if self.mcstat == 'correlations' or self.mcstat == 'both':
            r = 2*self.mcparams[1]*(np.random.rand(n,n) - 0.5)      # random matrix from U[-x, x]
            uti = np.triu_indices(n)        # upper triangle indices
            r[uti] = 0                      # lower triangle (except main diagonal) are only random values
            r = r + r.T         # Copy lower triangle to upper triangle. Diagonal values still zero.
            p = x.corr()        # Correlation matrix
            padd = p + r        # Add random values to correlations
            pclip = np.clip(padd, -1, 1)    # Clip correlations to between -1 and +1
            cperfect = np.divide(c, p)  # Divide out true correlations from covariance matrix, make all correlations = 1.0
            temp = np.multiply(cperfect, pclip)    # multiply the randomized correlations into the covariance matrix
            c.iloc[:,:] = temp        # Assign values to DataFrame covariance matrix

        return c

    def setWeights(self, i, stats):
        # set target weights for relevant assets
        ri = stats[0]

        if ri.size > 0:
            # if there are relevant assets, then assign weights.
            # Otherwise, leave weights at 0
            self.cashw.iloc[i] = 0.0
            if self.algo == 'Min. Var.':
                w = self.minVarOptimization(stats[1])
            elif self.algo == 'Tar. Var.':
                w = self.tarVarOptimization(stats[1])
            elif self.algo == 'Eq. Weight':
                cov = stats[1]
                n = cov.shape[0]
                w = np.ones(n)/n
            elif self.algo == 'Vol. Weight':
                w = self.volWeightOptimization(stats[1])
            elif self.algo == '130/30':
                w = np.array([1.3, -0.3])
            elif self.algo == '60/40':
                w = np.array([0.6, 0.4])
            else:
                w = -10
            #self.tw[ri].iloc[i] = w    # assign weights to relevant assets
            temp = self.tw.iloc[i]
            temp.loc[ri] = w
            self.tw.iloc[i] = temp

            # assign cash weight if not using 100% of NAV
            if w.sum() < 1.0:
                self.cashw.iloc[i] = 1.0 - w.sum()

    def minVarOptimization_new(self, covmatrix, lower_bounds, upper_bounds):
        # perform minimum variance optimization, and return weights
        # Based on: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        weights = upper_bounds/np.sum(upper_bounds)     # initial guess of weights
        bounds = so.Bounds(lower_bounds, upper_bounds)  # bounds for each weight
        constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0}]

        result = so.minimize(self.expVarFun, weights, args=covmatrix, bounds=bounds, constraints=constraints,
                             tol=1e-11, options={'maxiter': 1000, 'disp': False})
        if not result.success:
            print('Bad optimization!')
            print(result, end='', flush=True)

        return result.x.transpose()

    def minVarOptimization(self, cov):
        # perform minimum variance optimization, and return weights
        # Based on: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        covmatrix = np.array(cov)        # current estimate of covariance matrix
        n = cov.shape[0]           # number of relevant assets

        weights = np.ones(n)/n    # initial guess at optimal weights. Just guessing equal weight.
        bnds = ((self.c[0], self.c[1]),)*n     # bounds for each weight
        cons = ({'type': 'eq', 'fun': lambda w: weights.sum() - self.c[2]})   # sum of weights must match input

        # optimize
        result = so.minimize(self.expVarFun, weights, args=covmatrix,
                             bounds=bnds, constraints=cons, tol=1e-11, options={'maxiter': 1000, 'disp': False})
        if not result.success:
            print('Bad optimization!')
            print(result, end='', flush=True)

        return result.x.transpose()

    def expVarFun(self, weights, covmatrix):
        # This is the objective function for minimum variance optimization. This returns the expected variance of the
        # portfolio.
        expected_daily_variance = np.matmul(weights, np.matmul(covmatrix, weights.T))
        return expected_daily_variance

    def tarVarOptimization(self, cov):
        # perform min variance optimization, scale weights with cash (0% volatility)
        # to achieve target volatility. Then, return weights.
        # Based on: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        # get initial weights from minimum variance optimization
        w = self.minVarOptimization(cov)

        # calculate expected portfolio volatility
        cm = np.array(cov)
        expectedvar = self.expVarFun(w, cm)
        expectedvol = np.sqrt(252)*np.sqrt(expectedvar)

        # scale weights to match target volatility
        if not np.all(w == 0):
            scalefactor = self.c[4]/expectedvol
            w = w*scalefactor

        return w

    def volWeightOptimization(self, cov):
        # assign weights to equalize volatility.

        n = cov.shape[0]
        w = np.ones(n)/n
        for i in range(0, n):
            w[i] = self.c[1]/(n*100*math.sqrt(cov.iloc[i,i]))

        #if w.sum() > 1.0:
        w = w/w.sum()

        return w

    def backfillDF(self, df):
        # This function backfills the dataframe by repeating previous values until a nonzero value appears
        df = DataFrame(df)      # convert to DataFrame if not already one
        for i in range(1,df.shape[0]):
            if df.iloc[i,:].sum() == 0:   # if current row is zero, then use previous value
                df.iloc[i,:] = df.iloc[i-1,:]
        return df


