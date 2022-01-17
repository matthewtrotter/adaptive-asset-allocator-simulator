#!/usr/bin/env python
"""
Copyright (C) Matthew Trotter - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Written by Matthew Trotter, 2017.

Adaptive Asset Allocation Simulator

"""

from pandas import DataFrame
import plotly.offline as pyof
import plotly.graph_objs as go
import plotly.tools as pytools
import numpy as np
import datetime as d
import AssetUniverse as a
import Optimizer as op
import Backtester as bt


def main():
    # Define simulation start/end dates
    start = d.datetime(1980, 1, 1)
    temp = start.today()
    end = d.datetime(temp.year, temp.month, temp.day)

    # Define asset universe
    cashsym = ('VFISX')  # Fund that serves as cash position
    fedfundsspread = 1.0  # Broker's spread over Fed Funds Rate to borrow money
    #sym = ('VUSTX', 'Gold', 'VWUSX')  # Permanent Portfolio - start 1986
    # sym = ('US Corp Bonds', 'EM Corp Bonds', 'Oil', 'Gold', 'BRK-B', 'EWG', 'EWJ', 'FRESX', 'VUSTX')
    sym = ('SPY', 'VUSTX')
    AU = a.AssetUniverse(start, end, sym, cashsym, fedfundsspread)

    # Define look-ahead and look-back lengths
    la = 21
    lb = 21*6

    # Compute dummy metric
    percth1 = 1.0  # percent threshold rank
    thresholds = round(percth1 * len(sym))
    thtype = 'rank'
    print('Computing metrics... ', end='', flush=True)
    rebbars = np.ones(AU.r.shape[0], dtype=bool)
    for i in range(lb):
        rebbars[i] = False
    m = neutralmetric(AU, [lb], rebbars, thtype)
    print('Done.', flush=True)


    # Evaluate sample correlation matrices with the sample estimator
    O = op.Optimizer('Sample Opt.', thresholds, thtype, m, 1, lb,
                     AU, '', [], rebbars, 'sample', [], [])
    corrs = np.zeros((len(sym), len(sym), len(rebbars)))
    for i in np.where(rebbars)[0]:
        stats = O.runStats(i, 'correlation')
        corrs[:,:,i] = stats[1]

    # Evaluate actual correlation matrices with the perfect estimator
    O = op.Optimizer('Sample Opt.', thresholds, thtype, m, 1, la,
                     AU, '', [], rebbars, 'perfect', [], [])
    truecorrs = np.zeros((len(sym), len(sym), len(rebbars)))
    for i in np.where(rebbars)[0]:
        stats = O.runStats(i, 'correlation')
        truecorrs[:, :, i] = stats[1]


    # Plot
    pytools.set_credentials_file(username='dklsj5983d', api_key='OFpicNHdRhEPsAN1fgJ2')
    trace1 = go.Scatter(x=AU.p.index, y=corrs[0, 1, :],     name='Sample')
    trace2 = go.Scatter(x=AU.p.index, y=truecorrs[0, 1, :], name='Truth')
    data = [trace1, trace2]
    layout = go.Layout(title=str(lb) + '-Day Sample Correlation Compared to<br>'
                             + 'True Correlation for Next ' + str(la) + ' Days for Stocks and Bonds',
                       xaxis={'title': 'Date'},
                       yaxis={'title': 'Correlation', 'range': (-1, 1)},
                       width=750, height=650)

    fig = go.Figure(data=data, layout=layout)
    pyof.plot(fig, filename='correlations.html')



def neutralmetric(AU, mlb, rebbars, thtype):
    # Return zeros
    m = 0*AU.p.copy() + 1
    return m


main()
