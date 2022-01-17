# Adaptive Asset Allocator Simulator
Simulates an adaptive asset allocation investment strategy over time

## Installing
I suggest using a python virtual environment. First, install [Asset Universe](https://github.com/matthewtrotter/assetuniverse). Then, install the rest of the python requirements in a virtual environment:

```python
pip install -r requirements.txt
```

## Using
Open `simulate.py` and edit the parameters:

1. Start/end date
2. Assets 
3. Leverage (1.0 = no leverage, 1.5 = borrowing 50% of your portfolio from the broker at the daily interest rate specified in `au.borrowrate.returns`)
4. Momentum metrics
5. Subportfolio thresholds and allocations

Then, run `python simulate.py`. The code will simulate many subportfolios and print progress after every 10th subportfolio simulated. At the end, you should get console output and some pretty graphs. The plots should open automatically. If not, they are saved as html files, and you can open them separately.

Example output:
```
Downloading asset universe data... 
[*********************100%***********************]  10 of 10 completed
Done.
Backend MacOSX is interactive backend. Turning interactive mode on.
Finished subportfolio: 10 of 54
Finished subportfolio: 20 of 54
Finished subportfolio: 30 of 54
Finished subportfolio: 40 of 54
Finished subportfolio: 50 of 54
[==========] Backtesting
Backtest performance stats:
CAGR         = 11.6
Annual Vol.  = 13.3
Sharpe Ratio = 0.87
Max Drawdown = -26.7
```