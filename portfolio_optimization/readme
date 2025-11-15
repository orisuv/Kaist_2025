# Portfolio Optimization Homework

This project implements mean-variance portfolio optimization under several investment constraints. Using monthly stock return data (1990–2024), the goal is to construct efficient portfolios, compare their risk–return characteristics, and analyze how constraints affect the efficient frontier and Sharpe ratios.

# Overview

Uses CRSP stock-level monthly data (100 stocks) and S&P 500 index data.

Computes expected returns, covariance matrices, and optimal portfolio weights.

Constructs portfolios under four different cases:

Case 1 – No short-selling constraints, risk-free asset allowed

Case 2 – No risk-free asset, no short-selling

Case 3 – Risk-free asset allowed, no short-selling

Case 4 – Neither risk-free asset nor short-selling allowed

Compares slopes of the Capital Market Line (CML), volatilities, Sharpe ratios, and cumulative returns.

Demonstrates how relaxing constraints expands the efficient frontier and improves performance.

# Dataset Description

msf.csv — Monthly Stock File

100 stocks, 1990–2024

Variables:

permno: CRSP permanent identifier

ticker, comnam, indname

flag_sector: 1 = money sector, 0 = diversified

mdate: monthly date

ret: monthly total return (split/dividend adjusted)

me, me_l1m, prc, shrout

msp500_risk_free.csv

spret: S&P 500 monthly return

rf: risk-free rate (from Kenneth French database)

spindx: index level

# Methodology
1. Data Handling

Read and merge monthly return data.

Convert dates to datetime format.

Pivot stock returns to construct return matrices.

Compute:

Mean return vector

Covariance matrix

2. Optimization Formulation

Objective: minimize portfolio variance

Constraints:

Weight sum = 1

Optional: no short-selling

Optional: include risk-free asset

Optimization solved using scipy.optimize.minimize (SLSQP).

3. Efficient Frontier & CML

Compute minimum-variance portfolio.

Compute tangency portfolio when a risk-free asset is available.

Plot:

Efficient frontier (with/without constraints)

Capital Market Line (CML)

# Key Results (from HW1.pdf analysis) 

Case Comparisons

All portfolios have the same excess return (6.53%), but differ in volatility and Sharpe ratio.

Case 1 (risk-free allowed, short-selling allowed) achieves:

Lowest volatility (3.37%)

Highest Sharpe ratio (≈ 1.94)

Highest cumulative return

Steepest CML slope (0.559278)

Case 2 (no short-selling, no risk-free asset):

Higher volatility (9.46%)

Sharpe ratio ≈ 0.69

Lower cumulative return than Case 1

Case 3 (risk-free allowed, no short-selling):

Sharpe ratio lower than Case 1 (slope ≈ 0.381223)

Still superior to Case 2 due to access to risk-free asset

Case 4 (most restrictive: no RF, no short-selling):

Highest volatility

Lowest Sharpe ratio (0.41)

Worst performance among optimized portfolios

Interpretation

Fewer constraints → larger efficient frontier → higher Sharpe ratio.

Access to the risk-free asset significantly improves performance.

Allowing short-selling provides the steepest CML and best long-term cumulative return.

Optimized portfolios outperform the S&P 500 benchmark in most cases.
