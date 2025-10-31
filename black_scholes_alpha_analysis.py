"""
Black-Scholes Model Validation and Alpha Factor Analysis
Author: Market Analysis System
Date: 2025

This script:
1. Validates Black-Scholes model using real market data
2. Develops alpha factors for market analysis
3. Calculates Sharpe ratios for portfolio optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class BlackScholesValidator:
    """
    Validates Black-Scholes model against real market option prices
    """
    
    def __init__(self, ticker='AAPL', risk_free_rate=0.045):
        """
        Initialize the validator
        
        Parameters:
        ticker: Stock ticker symbol
        risk_free_rate: Current risk-free rate (default 4.5% for US Treasury)
        """
        self.ticker = ticker
        self.r = risk_free_rate
        self.stock = yf.Ticker(ticker)
        
    def get_stock_price(self):
        """Get current stock price"""
        hist = self.stock.history(period="1d")
        return hist['Close'].iloc[-1]
    
    def calculate_historical_volatility(self, days=252):
        """
        Calculate historical volatility from past price data
        
        Parameters:
        days: Number of trading days to look back
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days*1.5)  # Extra buffer for trading days
        
        hist = self.stock.history(start=start_date, end=end_date)
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes call option price
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes put option price
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def get_options_data(self):
        """
        Fetch real options data from Yahoo Finance
        """
        expirations = self.stock.options[:3]  # Get first 3 expiration dates
        
        all_options = []
        current_price = self.get_stock_price()
        
        for exp_date in expirations:
            opts = self.stock.option_chain(exp_date)
            
            # Process calls
            calls = opts.calls
            calls['type'] = 'call'
            calls['expiration'] = exp_date
            
            # Process puts
            puts = opts.puts
            puts['type'] = 'put'
            puts['expiration'] = exp_date
            
            all_options.append(calls)
            all_options.append(puts)
        
        options_df = pd.concat(all_options, ignore_index=True)
        options_df['currentPrice'] = current_price
        
        return options_df
    
    def validate_model(self):
        """
        Compare Black-Scholes prices with actual market prices
        """
        print("Fetching options data...")
        options_df = self.get_options_data()
        
        # Calculate time to expiration
        options_df['expiration'] = pd.to_datetime(options_df['expiration'])
        options_df['T'] = (options_df['expiration'] - datetime.now()).dt.days / 365
        
        # Get current stock price and volatility
        S = self.get_stock_price()
        sigma = self.calculate_historical_volatility()
        
        print(f"\nCurrent {self.ticker} Price: ${S:.2f}")
        print(f"Historical Volatility: {sigma:.2%}")
        print(f"Risk-free Rate: {self.r:.2%}")
        
        # Calculate Black-Scholes prices
        bs_prices = []
        market_prices = []
        
        for idx, row in options_df.iterrows():
            K = row['strike']
            T = row['T']
            market_price = row['lastPrice']
            
            if T <= 0 or pd.isna(market_price):
                continue
                
            if row['type'] == 'call':
                bs_price = self.black_scholes_call(S, K, T, self.r, sigma)
            else:
                bs_price = self.black_scholes_put(S, K, T, self.r, sigma)
            
            bs_prices.append(bs_price)
            market_prices.append(market_price)
        
        # Calculate validation metrics
        bs_prices = np.array(bs_prices)
        market_prices = np.array(market_prices)
        
        mae = np.mean(np.abs(bs_prices - market_prices))
        rmse = np.sqrt(np.mean((bs_prices - market_prices) ** 2))
        correlation = np.corrcoef(bs_prices, market_prices)[0, 1]
        
        print(f"\n=== Black-Scholes Model Validation Results ===")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Root Mean Square Error: ${rmse:.2f}")
        print(f"Correlation: {correlation:.4f}")
        
        return bs_prices, market_prices, correlation


class AlphaFactorAnalysis:
    """
    Develops and analyzes alpha factors for stock market prediction
    """
    
    def __init__(self, tickers, start_date='2023-01-01', end_date=None):
        """
        Initialize alpha factor analysis
        
        Parameters:
        tickers: List of stock tickers
        start_date: Start date for historical data
        end_date: End date for historical data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = {}
        self.returns = {}
        
    def fetch_data(self):
        """Fetch historical data for all tickers"""
        print("Fetching historical data...")
        for ticker in self.tickers:
            self.data[ticker] = yf.download(ticker, 
                                           start=self.start_date, 
                                           end=self.end_date,
                                           progress=False)
            self.returns[ticker] = self.data[ticker]['Close'].pct_change()
        
        print(f"Data fetched for {len(self.tickers)} tickers")
    
    def calculate_momentum_factor(self, lookback=20):
        """
        Calculate momentum factor (past returns)
        
        Parameters:
        lookback: Number of days to look back
        """
        momentum = {}
        for ticker in self.tickers:
            momentum[ticker] = self.data[ticker]['Close'].pct_change(lookback)
        
        return pd.DataFrame(momentum)
    
    def calculate_mean_reversion_factor(self, window=50):
        """
        Calculate mean reversion factor
        
        Parameters:
        window: Rolling window size
        """
        mean_reversion = {}
        for ticker in self.tickers:
            ma = self.data[ticker]['Close'].rolling(window=window).mean()
            mean_reversion[ticker] = (self.data[ticker]['Close'] - ma) / ma
        
        return pd.DataFrame(mean_reversion)
    
    def calculate_volatility_factor(self, window=20):
        """
        Calculate volatility factor
        
        Parameters:
        window: Rolling window size
        """
        volatility = {}
        for ticker in self.tickers:
            volatility[ticker] = self.returns[ticker].rolling(window=window).std() * np.sqrt(252)
        
        return pd.DataFrame(volatility)
    
    def calculate_volume_factor(self, window=20):
        """
        Calculate volume-based factor (volume ratio)
        
        Parameters:
        window: Rolling window size
        """
        volume_ratio = {}
        for ticker in self.tickers:
            current_vol = self.data[ticker]['Volume']
            avg_vol = current_vol.rolling(window=window).mean()
            volume_ratio[ticker] = current_vol / avg_vol
        
        return pd.DataFrame(volume_ratio)
    
    def calculate_rsi_factor(self, period=14):
        """
        Calculate RSI (Relative Strength Index) factor
        
        Parameters:
        period: RSI period
        """
        rsi = {}
        for ticker in self.tickers:
            delta = self.data[ticker]['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi[ticker] = 100 - (100 / (1 + rs))
        
        return pd.DataFrame(rsi)
    
    def combine_factors(self, weights=None):
        """
        Combine multiple alpha factors
        
        Parameters:
        weights: Dictionary of factor weights
        """
        # Calculate all factors
        factors = {
            'momentum': self.calculate_momentum_factor(),
            'mean_reversion': self.calculate_mean_reversion_factor(),
            'volatility': self.calculate_volatility_factor(),
            'volume': self.calculate_volume_factor(),
            'rsi': self.calculate_rsi_factor()
        }
        
        # Default equal weights if not specified
        if weights is None:
            weights = {factor: 1/len(factors) for factor in factors}
        
        # Normalize factors (z-score)
        normalized_factors = {}
        for name, factor_df in factors.items():
            normalized_factors[name] = (factor_df - factor_df.mean()) / factor_df.std()
        
        # Combine factors with weights
        combined_signal = pd.DataFrame(index=factors['momentum'].index, 
                                      columns=self.tickers)
        combined_signal.iloc[:, :] = 0
        
        for factor_name, factor_df in normalized_factors.items():
            combined_signal += weights.get(factor_name, 0) * factor_df
        
        return combined_signal, factors
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.045):
        """
        Calculate Sharpe ratio
        
        Parameters:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe
    
    def backtest_strategy(self, signal, holding_period=5):
        """
        Backtest the alpha factor strategy
        
        Parameters:
        signal: Combined alpha signal
        holding_period: Days to hold position
        """
        portfolio_returns = []
        
        # Create long-short portfolio based on signal
        for i in range(holding_period, len(signal) - holding_period):
            # Get signal rankings
            current_signal = signal.iloc[i].dropna()
            if len(current_signal) < len(self.tickers):
                continue
                
            # Long top 30%, short bottom 30%
            n_positions = max(1, int(len(current_signal) * 0.3))
            long_stocks = current_signal.nlargest(n_positions).index
            short_stocks = current_signal.nsmallest(n_positions).index
            
            # Calculate returns for holding period
            period_returns = []
            for ticker in long_stocks:
                ret = self.returns[ticker].iloc[i:i+holding_period].sum()
                period_returns.append(ret)
            
            for ticker in short_stocks:
                ret = -self.returns[ticker].iloc[i:i+holding_period].sum()
                period_returns.append(ret)
            
            if period_returns:
                portfolio_returns.append(np.mean(period_returns))
        
        portfolio_returns = pd.Series(portfolio_returns)
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252/len(portfolio_returns)) - 1
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = (portfolio_returns.cumsum() - 
                       portfolio_returns.cumsum().expanding().max()).min()
        
        return {
            'returns': portfolio_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def optimize_factor_weights(self):
        """
        Optimize factor weights to maximize Sharpe ratio
        """
        def objective(weights):
            # Ensure weights sum to 1
            weights_dict = {
                'momentum': weights[0],
                'mean_reversion': weights[1],
                'volatility': weights[2],
                'volume': weights[3],
                'rsi': weights[4]
            }
            
            signal, _ = self.combine_factors(weights_dict)
            results = self.backtest_strategy(signal)
            
            # Return negative Sharpe (we want to maximize)
            return -results['sharpe_ratio']
        
        # Initial weights (equal)
        x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Constraints: weights sum to 1, all weights between 0 and 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(5)]
        
        print("\nOptimizing factor weights...")
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = {
            'momentum': result.x[0],
            'mean_reversion': result.x[1],
            'volatility': result.x[2],
            'volume': result.x[3],
            'rsi': result.x[4]
        }
        
        return optimal_weights


def plot_validation_results(bs_prices, market_prices):
    """Plot Black-Scholes validation results"""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(market_prices, bs_prices, alpha=0.5)
    plt.plot([min(market_prices), max(market_prices)], 
             [min(market_prices), max(market_prices)], 
             'r--', label='Perfect Fit')
    plt.xlabel('Market Price ($)')
    plt.ylabel('Black-Scholes Price ($)')
    plt.title('Black-Scholes vs Market Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(1, 2, 2)
    errors = bs_prices - market_prices
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Pricing Error ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pricing Errors')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/bs_validation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_factor_performance(results, optimal_weights):
    """Plot alpha factor performance"""
    plt.figure(figsize=(14, 8))
    
    # Cumulative returns
    plt.subplot(2, 2, 1)
    cumulative_returns = (1 + results['returns']).cumprod()
    plt.plot(cumulative_returns, label='Strategy')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.title('Strategy Cumulative Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 2)
    plt.hist(results['returns'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.title('Returns Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    # Factor weights
    plt.subplot(2, 2, 3)
    factors = list(optimal_weights.keys())
    weights = list(optimal_weights.values())
    plt.bar(factors, weights, color='skyblue', edgecolor='black')
    plt.xlabel('Factors')
    plt.ylabel('Weight')
    plt.title('Optimal Factor Weights')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Performance metrics
    plt.subplot(2, 2, 4)
    metrics = {
        'Annual Return': f"{results['annual_return']:.2%}",
        'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}",
        'Max Drawdown': f"{results['max_drawdown']:.2%}",
        'Total Return': f"{results['total_return']:.2%}"
    }
    
    plt.axis('off')
    text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    plt.text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
             fontfamily='monospace')
    plt.title('Performance Metrics', pad=20)
    
    plt.tight_layout()
    plt.savefig('/home/claude/alpha_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Black-Scholes Validation & Alpha Factor Analysis")
    print("="*60)
    
    # Part 1: Black-Scholes Validation
    print("\n--- Part 1: Black-Scholes Model Validation ---")
    validator = BlackScholesValidator(ticker='AAPL')
    bs_prices, market_prices, correlation = validator.validate_model()
    
    if correlation > 0.9:
        print(f"\n✓ Black-Scholes model is HIGHLY ACCURATE (correlation: {correlation:.4f})")
    elif correlation > 0.7:
        print(f"\n✓ Black-Scholes model is REASONABLY ACCURATE (correlation: {correlation:.4f})")
    else:
        print(f"\n⚠ Black-Scholes model shows MODERATE accuracy (correlation: {correlation:.4f})")
    
    # Plot validation results
    plot_validation_results(bs_prices, market_prices)
    
    # Part 2: Alpha Factor Analysis
    print("\n--- Part 2: Alpha Factor Development & Analysis ---")
    
    # Select diversified portfolio of stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']
    
    analyzer = AlphaFactorAnalysis(tickers, start_date='2022-01-01')
    analyzer.fetch_data()
    
    # Test with equal weights first
    print("\nTesting equal-weighted factors...")
    signal, factors = analyzer.combine_factors()
    equal_results = analyzer.backtest_strategy(signal)
    print(f"Equal-weighted Sharpe Ratio: {equal_results['sharpe_ratio']:.3f}")
    
    # Optimize weights
    optimal_weights = analyzer.optimize_factor_weights()
    
    print("\n=== Optimal Factor Weights ===")
    for factor, weight in optimal_weights.items():
        print(f"{factor}: {weight:.3f}")
    
    # Test with optimal weights
    print("\nTesting optimized factor weights...")
    optimal_signal, _ = analyzer.combine_factors(optimal_weights)
    optimal_results = analyzer.backtest_strategy(optimal_signal)
    
    print("\n=== Strategy Performance (Optimal Weights) ===")
    print(f"Annual Return: {optimal_results['annual_return']:.2%}")
    print(f"Sharpe Ratio: {optimal_results['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {optimal_results['max_drawdown']:.2%}")
    print(f"Total Return: {optimal_results['total_return']:.2%}")
    
    # Plot performance
    plot_factor_performance(optimal_results, optimal_weights)
    
    # Compare improvement
    improvement = optimal_results['sharpe_ratio'] - equal_results['sharpe_ratio']
    print(f"\n✓ Sharpe Ratio Improvement: {improvement:.3f}")
    print(f"  ({equal_results['sharpe_ratio']:.3f} → {optimal_results['sharpe_ratio']:.3f})")
    
    print("\n" + "="*60)
    print("Analysis Complete! Check generated plots for visualizations.")
    print("="*60)
    
    return validator, analyzer, optimal_results, optimal_weights


if __name__ == "__main__":
    # Run the complete analysis
    validator, analyzer, results, weights = main()
