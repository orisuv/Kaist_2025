# Pair Trading Strategy

This project implements a statistical arbitrage strategy based on pair trading. The workflow includes data preprocessing, cointegration testing, signal generation, and backtesting to evaluate the profitability and robustness of the strategy.

# Overview

Loads and processes time-series price data for two correlated assets.

Tests for cointegration using statistical methods (Engle-Granger, ADF).

Calculates hedge ratios using linear regression.

Computes the spread and normalizes it using z-score.

Generates long/short trading signals based on spread deviations.

Performs backtesting to measure returns, drawdowns, and performance statistics.

# Methodology
1. Data Preprocessing

Aligns asset price series by date and handles missing values.

Computes log prices and returns for analysis.

2. Cointegration Analysis

Uses Engle-Granger two-step method.

Tests whether the spread between two assets is stationary.

Estimates hedge ratio using OLS.

3. Signal Generation

Spread is standardized using a rolling mean and standard deviation (z-score).

Trading rules commonly include:

Enter Long Spread: z-score < -entry_threshold

Enter Short Spread: z-score > entry_threshold

Exit Positions: z-score reverts toward zero

4. Backtesting

Simulates long/short positions using generated signals.

Calculates:

Cumulative returns

Sharpe ratio

Maximum drawdown

Trade statistics (win/loss ratio, number of trades)

Evaluates strategy stability and mean-reversion strength.
