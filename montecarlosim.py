#!/usr/bin/env python
"""
Copyright (C) Matthew Trotter - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
Written by Matthew Trotter, 2017.

Monte Carlo Simulator

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
    # Define Monte Carlo parameters and input randomness
    n = 25                   # Number of Monte Carlo simulations to run
    stat = 'both'           # Statistic to randomly modify ('std devs', 'correlations', 'both')
    x = (0.1, 0.6)              # Max random deviation from uniform distribution
                                # x[0]: Max percent change of std devs (e.g. 0.1 is 10% max change)
                                # x[1]: Max absolute change of correlation
                                # (e.g. 0.1 means true correlation +/- 0.1 maximum change)

    # Define simulation start/end dates
    start = d.datetime(1980, 1, 1)
    temp = start.today()
    end = d.datetime(temp.year, temp.month, temp.day)

    # Define asset universe
    cashsym = ('VFISX')  # Fund that serves as cash position
    fedfundsspread = 1.5  # Broker's spread over Fed Funds Rate to borrow money
    #sym = ('VUSTX', 'Gold', 'VWUSX')  # Permanent Portfolio - start 1986
    #sym = ('US Corp Bonds', 'EM Corp Bonds', 'Oil', 'Gold', 'BRK-B', 'EWG', 'EWJ', 'FRESX', 'VUSTX')
    sym = ('SPY', 'VUSTX', 'Oil', 'Gold', 'China Stocks')
    AU = a.AssetUniverse(start, end, sym, cashsym, fedfundsspread)

    # Define the optimizer and constraints
    opttype = 'Min. Var.'
    c = np.ones((5, 1))  # weight constraints
    c[0] = 0  # minimum individual weight
    c[1] = 1.0  # maximum individual weight
    c[2] = 1.0  # min total weight
    c[3] = 1.0  # max total weight
    c[4] = 0    # target variance


    # Determine rebalance days
    mlb = [126, 189]  # lookback periods for evaluating metrics. Average metric over each.
    rp = 21                      # rebalance period
    rebbars = np.zeros(AU.r.shape[0], dtype=bool)
    for i in range(max(mlb), AU.r.shape[0], rp):
        rebbars[i] = True

    # Calculate metric
    percth1 = 1.0  # percent threshold rank
    thresholds = round(percth1 * len(sym))
    thtype = 'rank'
    print('Computing metrics... ', end='', flush=True)
    m = SharpeMetric(AU, mlb, rebbars, thtype)
    print('Done.\n', flush=True)

    # Run simulation with perfect covariance estimator
    estimator = 'perfect'   # Estimator looks into the future to get the true covariance matrix
    statslb = rp+1          # Look into the future until the next rebalance bar
    O = op.Optimizer('Perfect Opt.', thresholds, thtype, m, rp, statslb,
                     AU, opttype, c, rebbars, estimator, [], [])
    O.simulate()
    B = bt.Backtester(O)
    B.backtest()
    B.getstats()
    perfectNAV = B.NAV
    perfectstats = B.stats

    # Run simulation with sample covariance estimator
    estimator = 'sample'    # Estimator looks into the past to estimate covariance matrix
                            # until next rebalance bar
    statslb = mlb[0]        # Look into the past
    O = op.Optimizer('Sample Opt.', thresholds, thtype, m, rp, statslb,
                     AU, opttype, c, rebbars, estimator, [], [])
    O.simulate()
    B = bt.Backtester(O)
    B.backtest()
    B.getstats()
    sampleNAV = B.NAV
    samplestats = B.stats


    # Run Monte Carlo simulations and store results
    estimator = 'perfect + random'
    statslb = rp + 1        # Look into the future until the next rebalance bar
    mcNAV = np.zeros((n, sampleNAV.shape[0]))
    mcstats = np.zeros((n, len(samplestats)))
    for i in range(n):
        O = op.Optimizer('MC sim ' + str(i+1), thresholds, thtype, m, rp, statslb,
                         AU, opttype, c, rebbars, estimator, stat, x)
        O.simulate()
        B = bt.Backtester(O)
        B.backtest()
        B.getstats()
        mcNAV[i, :] = B.NAV
        mcstats[i, :] = B.stats



    # Plot NAV cloud chart - perfect NAV, sample NAV, and 10 Monte Carlo simulations
    nplot = min(10, n)
    pytools.set_credentials_file(username='dklsj5983d', api_key='OFpicNHdRhEPsAN1fgJ2')
    NAVs = [{'x': AU.p.index, 'y': mcNAV[i,:], 'name': 'Monte Carlo Sim ' + str(i+1), 'line': {'width': 1}
            } for i in range(nplot)]
    NAVs.append({'x': AU.p.index, 'y': perfectNAV, 'name': 'Perfect Cov. Estimator', 'line': {'color': 'green', 'width': 5}})
    NAVs.append({'x': AU.p.index, 'y': sampleNAV, 'name': 'Sample Cov. Estimator', 'line': {'color': 'red', 'width': 5}})

    titlestring = 'Monte Carlo Simulation, ' + str(n) + ' Iterations (' + str(nplot) + ' Shown)<br>'
    stddevstring = 'Std Devs Randomly Multiplied by U(' + str(1-x[0]) + ', ' + str(1+x[0]) + ')<br>'
    corrstring = 'Correlations Randomly Added With U(' + str(-x[1]) + ', ' + str(x[1]) + ') and Capped Between [-1, 1]<br>'
    if stat == 'std dev':
        titlestring = titlestring + stddevstring
    elif stat == 'correlations':
        titlestring = titlestring + corrstring
    elif stat == 'both':
        titlestring = titlestring + stddevstring + '\n' + corrstring

    layout1 = go.Layout(title=titlestring, xaxis={'title': 'Date'},
                       yaxis={'title': 'NAV ($1 start)', 'type': 'log', 'autorange': True})

    fig1 = go.Figure(data=NAVs, layout=layout1)
    pyof.plot(fig1, filename='montecarlonavplots.html')

    # Plot scatter plots of CAGR, Annual Vol, Sharpe Ratio, and Max Draw Down
    trace1 = go.Scatter(x=100*mcstats[:,1], y=100*mcstats[:,0], mode='markers',
                        name='Monte Carlo Iterations')
    trace2 = go.Scatter(x=[100*perfectstats[1]], y=[100*perfectstats[0]], mode='markers',
                        name='Perfect Cov. Estimator',
                        marker=dict(size=10, color='rgba(0, 255, 0, .9)', line=dict(width=2)))
    trace3 = go.Scatter(x=[100*samplestats[1]], y=[100*samplestats[0]], mode='markers',
                        name = 'Sample Cov. Estimator',
                        marker=dict(size=10, color='rgba(255, 0, 0, .9)', line=dict(width=2)))
    data = [trace1, trace2, trace3]
    minx = 0
    maxx = 1.1*max(100*mcstats[:,1].max(), perfectstats[1], samplestats[1], 10.0)
    miny = 0
    maxy = 1.1*max(100*mcstats[:,0].max(), perfectstats[0], samplestats[0], 10.0)
    layout = go.Layout(title='Monte Carlo Simulation Statistics',
                       xaxis={'title': 'Annual Vol. (%)', 'autorange': False, 'range': [minx, maxx]},
                       yaxis={'title': 'CAGR (%)', 'autorange': False, 'range': [miny, maxy]})

    fig = go.Figure(data=data, layout=layout)
    pyof.plot(fig, filename='montecarloscatter.html')

    # Plot table of Monte Carlo stats
    table_data = [['', 'Perfect Cov. Est. ', 'Sample Cov. Est.', 'MC Mean', 'MC Std. Dev.'],
                  ['CAGR (%)', 100*perfectstats[0], 100*samplestats[0], 100*np.mean(mcstats[:,0]), 100*np.std(mcstats[:,0])],
                  ['Annual Vol. (%)', 100*perfectstats[1], 100*samplestats[1], 100*np.mean(mcstats[:,1]), 100*np.std(mcstats[:,1])],
                  ['Sharpe Ratio (0%)', perfectstats[2], samplestats[2], np.mean(mcstats[:,2]), np.std(mcstats[:,2])],
                  ['Max Dradown (%)', 100*perfectstats[3], 100*samplestats[3], 100*np.mean(mcstats[:,3]), 100*np.std(mcstats[:,3])]]

    table = pytools.FigureFactory.create_table(table_data)  # , height_constant=60)
    pyof.plot(table, filename='montecarlostats.html')



def SharpeMetric(AU, mlb, rebbars, thtype):
    # Sharpe ratio (Risk free rate = 0) over the past n days. Average result over all n.
    m = 0*AU.r
    dim = AU.r.shape
    if thtype == 'rank':
        dfsize = len(mlb)
    else:
        dfsize = 1
    tempm = DataFrame(np.zeros((dfsize, dim[1])), columns=AU.r.axes[1])
    for i in range(0, dim[0]):
        if rebbars[i]:
            for j in range(0, dfsize):
                if thtype == 'rank':
                    n = mlb[j]
                else:
                    n = mlb
                meanr = AU.r.ix[(i-n+1):i+1].mean()
                sdr = AU.r.ix[(i-n+1):i+1].std()
                tempm.ix[j,:] = meanr/sdr

                if thtype == 'rank':
                    mrange = tempm.ix[j,:].max() - tempm.ix[j,:].min()
                    tempm.ix[j,:] = tempm.ix[j,:]/mrange      # normalize to [0.0, 1.0]
                    tempm.ix[j,:] = tempm.ix[j,:] - tempm.ix[j,:].min()
            m.ix[i] = tempm.mean()
    return m


main()
