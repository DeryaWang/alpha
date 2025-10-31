# alpha
# Black-Scholes Validation & Alpha Factor Analysis

## Project Overview

This project implements two main functionalities:
1. **Black-Scholes Model Validation**: Validates the Black-Scholes pricing model accuracy using real options market data
2. **Alpha Factor Analysis**: Develops multiple alpha factors and optimizes portfolio weights to maximize Sharpe ratio

## Features

### Black-Scholes Validation
- Fetches real-time options data from Yahoo Finance
- Calculates historical volatility
- Compares theoretical prices with market prices
- Computes validation metrics (MAE, RMSE, Correlation)
- Visualizes validation results

### Alpha Factor Analysis
- **Momentum Factor**: Based on historical returns
- **Mean Reversion Factor**: Price deviation from moving averages
- **Volatility Factor**: Standard deviation of returns
- **Volume Factor**: Volume ratio analysis
- **RSI Factor**: Relative Strength Index

### Performance Optimization
- Factor weight optimization algorithm
- Sharpe ratio maximization
- Backtesting strategy implementation
- Risk-adjusted return analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run complete analysis
python black_scholes_alpha_analysis.py
```

## Output

The program generates the following:

1. **Black-Scholes Validation Results**
   - Model accuracy metrics
   - Price comparison scatter plot
   - Error distribution histogram

2. **Alpha Factor Analysis Results**
   - Optimal factor weights
   - Strategy performance metrics (Annual return, Sharpe ratio, Maximum drawdown)
   - Cumulative return curves
   - Return distribution plots

## Main Components

### BlackScholesValidator Class
```python
validator = BlackScholesValidator(ticker='AAPL')
bs_prices, market_prices, correlation = validator.validate_model()
```

### AlphaFactorAnalysis Class
```python
analyzer = AlphaFactorAnalysis(tickers, start_date='2022-01-01')
analyzer.fetch_data()
optimal_weights = analyzer.optimize_factor_weights()
```

## Key Metrics

- **Sharpe Ratio**: Risk-adjusted return metric, higher is better
- **Maximum Drawdown**: Largest peak-to-trough decline, lower is better
- **Correlation**: Black-Scholes vs market price correlation, closer to 1 is better
- **Annual Return**: Annualized portfolio return

## Data Sources

- **Options Data**: Yahoo Finance real-time option chains
- **Stock Prices**: Yahoo Finance historical data
- **Risk-free Rate**: Default 4.5% (adjustable)

## Important Notes

1. Ensure stable internet connection for real-time data fetching
2. Some options may lack liquidity, causing price deviations
3. Optimization process may take several minutes
4. Recommend using highly liquid stocks for analysis

## Customization

Modifiable parameters:
- Stock ticker list
- Backtesting time range
- Risk-free rate
- Factor parameters (e.g., momentum lookback period, RSI period)
- Holding period

## Sample Output

```
=== Black-Scholes Model Validation Results ===
Mean Absolute Error: $2.34
Root Mean Square Error: $3.12
Correlation: 0.9234

=== Optimal Factor Weights ===
momentum: 0.284
mean_reversion: 0.213
volatility: 0.189
volume: 0.156
rsi: 0.158

=== Strategy Performance (Optimal Weights) ===
Annual Return: 18.45%
Sharpe Ratio: 1.234
Maximum Drawdown: -12.34%
Total Return: 28.67%
```
