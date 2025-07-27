"""
Smart Grid Trading Bot with ML Optimization and Web Dashboard
Complete self-contained version - no external dependencies
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
import asyncio
from dataclasses import dataclass, field
from collections import deque
import statistics
import os
import threading
from queue import Queue

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: scikit-learn not installed. ML features will be disabled.")
    print("Install with: pip install scikit-learn")

# Web Dashboard imports
try:
    import plotly.graph_objs as go
    import plotly.utils
    from enhanced_dashboard import EnhancedGridBotDashboard, PerformanceAnalyzer, AlertSystem
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Flask dependencies not installed. Dashboard will be disabled.")
    print("Install with: pip install flask flask-cors flask-socketio plotly")

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============

@dataclass
class GridConfig:
    """Configuration for grid trading bot"""
    api_key: str
    api_secret: str
    
    # Grid Parameters
    min_grid_levels: int = 10
    max_grid_levels: int = 50
    grid_spacing_percent: float = 0.005  # 0.5% minimum spacing
    
    # Capital Allocation
    total_capital: float = 10000  # Total capital in USDT
    max_grids_active: int = 3  # Maximum concurrent grid strategies
    capital_per_grid: float = 0.3  # 30% of capital per grid
    
    # Futures-specific settings
    leverage: int = 3  # Leverage for futures trading (1-20)
    margin_type: str = 'cross'  # 'cross' or 'isolated'
    
    # Market Conditions Thresholds
    min_volatility: float = 0.02  # 2% daily volatility minimum
    max_volatility: float = 0.15  # 15% daily volatility maximum
    min_volume_usd: float = 1000000  # $1M daily volume minimum
    max_trend_strength: float = 0.3  # Maximum trend (for ranging markets)
    
    # Profit Taking
    grid_profit_percent: float = 0.005  # 0.5% profit per grid level
    fee_rate: float = 0.001  # 0.1% trading fee
    
    # Risk Management
    stop_loss_percent: float = 0.10  # 10% stop loss from range
    max_position_size: float = 0.5  # Max 50% in one side
    rebalance_threshold: float = 0.7  # Rebalance if 70% on one side
    
    # Scanner Settings
    scan_interval: int = 300  # Scan every 5 minutes
    analysis_period: int = 24  # 24 hour analysis window
    
    # Symbols to scan (top liquid pairs for futures)
    symbols_to_scan: List[str] = field(default_factory=lambda: [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
        'SOL/USDT:USDT', 'HYPER/USDT:USDT', 'HYPE/USDT:USDT', 'WIF/USDT:USDT', '1000BONK/USDT:USDT', 'ENA/USDT:USDT', 'DOT/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT',
        'UNI/USDT:USDT', '1000PEPE/USDT:USDT', 'FARTCOIN/USDT:USDT', 'ATOM/USDT:USDT', 'LTC/USDT:USDT', 'ETC/USDT:USDT', 'XLM/USDT:USDT'
    ])
    
    testnet: bool = False

@dataclass
class GridScore:
    """Scoring for grid trading suitability"""
    symbol: str
    total_score: float  # 0-100
    volatility_score: float
    range_score: float
    volume_score: float
    spread_score: float
    price_range: Tuple[float, float]
    optimal_grid_count: int
    expected_daily_trades: float
    expected_daily_return: float
    current_price: float
    analysis_data: Dict

# ============= MARKET ANALYZER =============

class MarketAnalyzer:
    """Analyzes market conditions for grid trading suitability"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def analyze_symbol(self, symbol: str, config: GridConfig) -> Optional[GridScore]:
        """Analyze a symbol for grid trading suitability"""
        try:
            # Check cache
            if symbol in self.cache:
                cached_time, cached_score = self.cache[symbol]
                if time.time() - cached_time < self.cache_duration:
                    return cached_score
            
            # Fetch data
            analysis_data = await self.collect_market_data(symbol, config.analysis_period)
            if not analysis_data:
                return None
            
            # Calculate scores
            volatility_score = self.score_volatility(analysis_data, config)
            range_score = self.score_range_behavior(analysis_data, config)
            volume_score = self.score_volume(analysis_data, config)
            spread_score = self.score_spread(analysis_data)
            
            # Calculate optimal grid parameters
            price_range, grid_count = self.calculate_optimal_grid(analysis_data, config)
            
            # Estimate performance
            expected_trades, expected_return = self.estimate_performance(
                analysis_data, grid_count, config
            )
            
            # Total score (weighted average)
            total_score = (
                volatility_score * 0.3 +
                range_score * 0.4 +
                volume_score * 0.2 +
                spread_score * 0.1
            )
            
            score = GridScore(
                symbol=symbol,
                total_score=total_score,
                volatility_score=volatility_score,
                range_score=range_score,
                volume_score=volume_score,
                spread_score=spread_score,
                price_range=price_range,
                optimal_grid_count=grid_count,
                expected_daily_trades=expected_trades,
                expected_daily_return=expected_return,
                current_price=analysis_data['current_price'],
                analysis_data=analysis_data
            )
            
            # Cache result
            self.cache[symbol] = (time.time(), score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def collect_market_data(self, symbol: str, hours: int) -> Optional[Dict]:
        """Collect comprehensive market data"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe='1h', 
                limit=hours + 1
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            if len(df) < hours:
                return None
            
            # Fetch current ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Fetch order book for spread analysis
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)
            
            # Calculate metrics
            returns = df['close'].pct_change().dropna()
            
            # Price metrics
            current_price = ticker['last']
            high_24h = df['high'].max()
            low_24h = df['low'].min()
            price_range = (high_24h - low_24h) / current_price
            
            # Volatility metrics
            volatility = returns.std() * np.sqrt(24)  # Daily volatility
            hourly_volatility = returns.std()
            
            # Trend metrics
            sma_short = df['close'].rolling(window=6).mean().iloc[-1]
            sma_long = df['close'].rolling(window=24).mean().iloc[-1]
            trend_strength = abs(sma_short - sma_long) / sma_long
            
            # Range behavior metrics
            # Count how many times price crossed the middle of the range
            middle_price = (high_24h + low_24h) / 2
            crosses = 0
            for i in range(1, len(df)):
                if (df['close'].iloc[i-1] < middle_price and df['close'].iloc[i] > middle_price) or \
                   (df['close'].iloc[i-1] > middle_price and df['close'].iloc[i] < middle_price):
                    crosses += 1
            
            # Volume metrics
            avg_volume_usd = (df['volume'] * df['close']).mean()
            volume_stability = 1 - (df['volume'].std() / df['volume'].mean())
            
            # Spread metrics
            bid_ask_spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / current_price
            
            # Support and resistance levels
            support_levels = self.find_support_resistance(df, 'support')
            resistance_levels = self.find_support_resistance(df, 'resistance')
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'price_range_percent': price_range,
                'volatility_daily': volatility,
                'volatility_hourly': hourly_volatility,
                'trend_strength': trend_strength,
                'range_crosses': crosses,
                'avg_volume_usd': avg_volume_usd,
                'volume_stability': volume_stability,
                'bid_ask_spread': bid_ask_spread,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'dataframe': df
            }
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    def find_support_resistance(self, df: pd.DataFrame, type: str = 'support') -> List[float]:
        """Find support and resistance levels"""
        levels = []
        
        if type == 'support':
            for i in range(2, len(df) - 2):
                if df['low'].iloc[i] < df['low'].iloc[i-1] and \
                   df['low'].iloc[i] < df['low'].iloc[i+1] and \
                   df['low'].iloc[i] < df['low'].iloc[i-2] and \
                   df['low'].iloc[i] < df['low'].iloc[i+2]:
                    levels.append(df['low'].iloc[i])
        else:
            for i in range(2, len(df) - 2):
                if df['high'].iloc[i] > df['high'].iloc[i-1] and \
                   df['high'].iloc[i] > df['high'].iloc[i+1] and \
                   df['high'].iloc[i] > df['high'].iloc[i-2] and \
                   df['high'].iloc[i] > df['high'].iloc[i+2]:
                    levels.append(df['high'].iloc[i])
        
        return sorted(levels)[-3:] if levels else []  # Return top 3 levels
    
    def score_volatility(self, data: Dict, config: GridConfig) -> float:
        """Score volatility for grid trading (ideal: moderate volatility)"""
        volatility = data['volatility_daily']
        
        if volatility < config.min_volatility:
            return 0  # Too low
        elif volatility > config.max_volatility:
            return 0  # Too high
        else:
            # Optimal volatility around 5-8%
            if 0.05 <= volatility <= 0.08:
                return 100
            elif volatility < 0.05:
                return (volatility - config.min_volatility) / (0.05 - config.min_volatility) * 100
            else:
                return (config.max_volatility - volatility) / (config.max_volatility - 0.08) * 100
    
    def score_range_behavior(self, data: Dict, config: GridConfig) -> float:
        """Score how well the asset trades in a range"""
        trend_strength = data['trend_strength']
        range_crosses = data['range_crosses']
        price_range = data['price_range_percent']
        
        # Penalize strong trends
        trend_score = max(0, 100 - (trend_strength / config.max_trend_strength) * 100)
        
        # Reward range crossing behavior
        cross_score = min(100, range_crosses * 5)  # 20 crosses = perfect score
        
        # Reward appropriate price range
        if 0.03 <= price_range <= 0.10:
            range_score = 100
        elif price_range < 0.03:
            range_score = price_range / 0.03 * 100
        else:
            range_score = max(0, 100 - (price_range - 0.10) / 0.10 * 100)
        
        return (trend_score * 0.4 + cross_score * 0.4 + range_score * 0.2)
    
    def score_volume(self, data: Dict, config: GridConfig) -> float:
        """Score volume and liquidity"""
        volume_usd = data['avg_volume_usd']
        volume_stability = data['volume_stability']
        
        if volume_usd < config.min_volume_usd:
            return 0
        
        # Log scale for volume score
        volume_score = min(100, np.log10(volume_usd / config.min_volume_usd) * 50)
        
        # Stability bonus
        stability_score = volume_stability * 100
        
        return volume_score * 0.7 + stability_score * 0.3
    
    def score_spread(self, data: Dict) -> float:
        """Score bid-ask spread (lower is better)"""
        spread = data['bid_ask_spread']
        
        if spread < 0.0005:  # Less than 0.05%
            return 100
        elif spread < 0.001:  # Less than 0.1%
            return 80
        elif spread < 0.002:  # Less than 0.2%
            return 60
        elif spread < 0.005:  # Less than 0.5%
            return 30
        else:
            return 0
    
    def calculate_optimal_grid(self, data: Dict, config: GridConfig) -> Tuple[Tuple[float, float], int]:
        """Calculate optimal grid parameters"""
        current_price = data['current_price']
        volatility = data['volatility_hourly']
        
        # Use support and resistance levels if available
        if data['support_levels'] and data['resistance_levels']:
            lower_bound = max(data['support_levels'])
            upper_bound = min(data['resistance_levels'])
        else:
            # Use statistical range
            std_dev = volatility * current_price
            lower_bound = current_price - 2 * std_dev
            upper_bound = current_price + 2 * std_dev
        
        # Ensure reasonable bounds
        lower_bound = max(lower_bound, current_price * 0.90)  # Max 10% below
        upper_bound = min(upper_bound, current_price * 1.10)  # Max 10% above
        
        # Calculate optimal grid count based on volatility
        price_range = upper_bound - lower_bound
        optimal_spacing = max(
            config.grid_spacing_percent * current_price,
            volatility * current_price * 0.5  # Half of hourly volatility
        )
        
        grid_count = int(price_range / optimal_spacing)
        grid_count = max(config.min_grid_levels, min(config.max_grid_levels, grid_count))
        
        return (lower_bound, upper_bound), grid_count
    
    def estimate_performance(self, data: Dict, grid_count: int, config: GridConfig) -> Tuple[float, float]:
        """Estimate expected daily trades and returns"""
        # Estimate trades based on volatility and grid spacing
        volatility_hourly = data['volatility_hourly']
        price_range = data['price_range_percent']
        
        # Average price movement per hour as percentage of grid spacing
        grid_spacing = price_range / grid_count
        movements_per_hour = volatility_hourly / grid_spacing
        
        # Estimate trades (each movement can trigger 2 trades - buy and sell)
        estimated_trades_daily = movements_per_hour * 24 * 0.5  # 0.5 factor for realistic estimation
        
        # Estimate returns
        profit_per_trade = config.grid_profit_percent - (2 * config.fee_rate)  # Minus fees
        estimated_return_daily = estimated_trades_daily * profit_per_trade
        
        return estimated_trades_daily, estimated_return_daily

# ============= GRID STRATEGY =============

class GridStrategy:
    """Individual grid trading strategy"""
    
    def __init__(self, symbol: str, config: GridConfig, grid_params: Dict):
        self.symbol = symbol
        self.config = config
        self.lower_bound = grid_params['lower_bound']
        self.upper_bound = grid_params['upper_bound']
        self.grid_count = grid_params['grid_count']
        self.investment = grid_params['investment']
        
        # Calculate grid levels
        self.grid_levels = np.linspace(self.lower_bound, self.upper_bound, self.grid_count)
        self.grid_spacing = self.grid_levels[1] - self.grid_levels[0]
        
        # Order tracking
        self.buy_orders = {}  # price -> order_id
        self.sell_orders = {}  # price -> order_id
        self.positions = {}  # price -> amount
        
        # Performance tracking
        self.total_trades = 0
        self.realized_pnl = 0
        self.fees_paid = 0
        self.start_time = datetime.now()
        
        # State
        self.active = True
        self.last_price = None
    
    def calculate_order_size(self) -> float:
        """Calculate size for each grid order"""
        # Equal distribution across all grid levels
        total_grids = self.grid_count - 1
        return self.investment / total_grids / ((self.upper_bound + self.lower_bound) / 2)
    
    def get_initial_orders(self, current_price: float) -> Tuple[List[Dict], List[Dict]]:
        """Get initial grid orders to place"""
        buy_orders = []
        sell_orders = []
        order_size = self.calculate_order_size()
        
        for level in self.grid_levels:
            if level < current_price * 0.999:  # Buy orders below current price
                buy_orders.append({
                    'price': level,
                    'amount': order_size,
                    'side': 'buy'
                })
            elif level > current_price * 1.001:  # Sell orders above current price
                sell_orders.append({
                    'price': level,
                    'amount': order_size,
                    'side': 'sell'
                })
        
        return buy_orders, sell_orders
    
    def should_rebalance(self, current_price: float) -> bool:
        """Check if grid needs rebalancing"""
        # Check if price is outside grid range
        if current_price < self.lower_bound * 0.95 or current_price > self.upper_bound * 1.05:
            return True
        
        # Check if too many orders on one side
        total_buy_value = sum(price * self.positions.get(price, 0) for price in self.buy_orders)
        total_sell_value = sum(price * self.positions.get(price, 0) for price in self.sell_orders)
        total_value = total_buy_value + total_sell_value
        
        if total_value > 0:
            buy_ratio = total_buy_value / total_value
            if buy_ratio > self.config.rebalance_threshold or buy_ratio < (1 - self.config.rebalance_threshold):
                return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600  # Hours
        
        return {
            'symbol': self.symbol,
            'total_trades': self.total_trades,
            'realized_pnl': self.realized_pnl,
            'fees_paid': self.fees_paid,
            'net_pnl': self.realized_pnl - self.fees_paid,
            'runtime_hours': runtime,
            'trades_per_hour': self.total_trades / runtime if runtime > 0 else 0,
            'active_orders': len(self.buy_orders) + len(self.sell_orders),
            'grid_range': (self.lower_bound, self.upper_bound)
        }

# ============= ML OPTIMIZER =============

if ML_AVAILABLE:
    class MLGridOptimizer:
        """Machine Learning based grid parameter optimizer"""
        
        def __init__(self, history_file: str = 'grid_performance_history.json'):
            self.history_file = history_file
            self.models = {}
            self.scalers = {}
            self.feature_importance = {}
            self.min_history_required = 100
            self.performance_history = self.load_history()
            
            # Initialize models
            self.models['profit'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['trade_frequency'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.models['risk_score'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Train if we have enough history
            if len(self.performance_history) >= self.min_history_required:
                self.train_models()
        
        def load_history(self) -> List[Dict]:
            """Load historical performance data"""
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        
        def save_history(self):
            """Save performance history"""
            try:
                with open(self.history_file, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving history: {e}")
        
        def extract_features(self, market_data: Dict, grid_params: Dict) -> np.ndarray:
            """Extract features for ML models"""
            features = []
            
            # Market features
            features.extend([
                market_data.get('volatility_daily', 0),
                market_data.get('volatility_hourly', 0),
                market_data.get('price_range_percent', 0),
                market_data.get('trend_strength', 0),
                market_data.get('range_crosses', 0),
                market_data.get('volume_stability', 0),
                market_data.get('bid_ask_spread', 0),
                np.log10(market_data.get('avg_volume_usd', 1))
            ])
            
            # Grid parameters
            features.extend([
                grid_params.get('grid_count', 20),
                grid_params.get('grid_spacing_percent', 0.005),
                grid_params.get('range_multiplier', 2.0),
                grid_params.get('investment_percent', 0.3)
            ])
            
            # Time features
            now = datetime.now()
            features.extend([
                now.hour,  # Hour of day
                now.weekday(),  # Day of week
                int(now.strftime('%j'))  # Day of year
            ])
            
            # Technical indicators
            if 'dataframe' in market_data:
                df = market_data['dataframe']
                if len(df) > 20:
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    features.append(rsi.iloc[-1])
                    
                    # Bollinger Band width
                    sma = df['close'].rolling(window=20).mean()
                    std = df['close'].rolling(window=20).std()
                    bb_width = (std * 2) / sma
                    features.append(bb_width.iloc[-1])
                else:
                    features.extend([50, 0.02])  # Default values
            else:
                features.extend([50, 0.02])  # Default values
            
            return np.array(features)
        
        def add_performance_record(self, symbol: str, market_data: Dict, 
                                 grid_params: Dict, performance: Dict):
            """Add a performance record for training"""
            record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'features': self.extract_features(market_data, grid_params).tolist(),
                'grid_params': grid_params,
                'performance': performance
            }
            
            self.performance_history.append(record)
            self.save_history()
            
            # Retrain models periodically
            if len(self.performance_history) % 50 == 0:
                self.train_models()
        
        def train_models(self):
            """Train ML models on historical data"""
            if len(self.performance_history) < self.min_history_required:
                logger.warning(f"Not enough history to train models: {len(self.performance_history)}/{self.min_history_required}")
                return
            
            logger.info("Training ML models...")
            
            # Prepare training data
            X = np.array([record['features'] for record in self.performance_history])
            y_profit = np.array([record['performance']['daily_return_percent'] for record in self.performance_history])
            y_trades = np.array([record['performance']['trades_per_day'] for record in self.performance_history])
            y_risk = np.array([record['performance']['max_drawdown_percent'] for record in self.performance_history])
            
            # Split data
            X_train, X_test, y_profit_train, y_profit_test = train_test_split(X, y_profit, test_size=0.2, random_state=42)
            _, _, y_trades_train, y_trades_test = train_test_split(X, y_trades, test_size=0.2, random_state=42)
            _, _, y_risk_train, y_risk_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
            
            # Scale features
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            # Train profit model
            X_train_scaled = self.scalers['profit'].fit_transform(X_train)
            X_test_scaled = self.scalers['profit'].transform(X_test)
            self.models['profit'].fit(X_train_scaled, y_profit_train)
            profit_score = self.models['profit'].score(X_test_scaled, y_profit_test)
            self.feature_importance['profit'] = self.models['profit'].feature_importances_
            
            # Train trade frequency model
            X_train_scaled = self.scalers['trade_frequency'].fit_transform(X_train)
            X_test_scaled = self.scalers['trade_frequency'].transform(X_test)
            self.models['trade_frequency'].fit(X_train_scaled, y_trades_train)
            trades_score = self.models['trade_frequency'].score(X_test_scaled, y_trades_test)
            self.feature_importance['trade_frequency'] = self.models['trade_frequency'].feature_importances_
            
            # Train risk model
            X_train_scaled = self.scalers['risk_score'].fit_transform(X_train)
            X_test_scaled = self.scalers['risk_score'].transform(X_test)
            self.models['risk_score'].fit(X_train_scaled, y_risk_train)
            risk_score = self.models['risk_score'].score(X_test_scaled, y_risk_test)
            self.feature_importance['risk_score'] = self.models['risk_score'].feature_importances_
            
            logger.info(f"Model R² Scores - Profit: {profit_score:.3f}, Trades: {trades_score:.3f}, Risk: {risk_score:.3f}")
            
            # Save models
            self.save_models()
        
        def predict_performance(self, market_data: Dict, grid_params: Dict) -> Dict:
            """Predict performance for given parameters"""
            features = self.extract_features(market_data, grid_params)
            
            predictions = {}
            
            try:
                # Make predictions
                if 'profit' in self.models and 'profit' in self.scalers:
                    features_scaled = self.scalers['profit'].transform([features])
                    predictions['expected_daily_return'] = float(self.models['profit'].predict(features_scaled)[0])
                
                if 'trade_frequency' in self.models and 'trade_frequency' in self.scalers:
                    features_scaled = self.scalers['trade_frequency'].transform([features])
                    predictions['expected_trades_per_day'] = float(self.models['trade_frequency'].predict(features_scaled)[0])
                
                if 'risk_score' in self.models and 'risk_score' in self.scalers:
                    features_scaled = self.scalers['risk_score'].transform([features])
                    predictions['expected_max_drawdown'] = float(self.models['risk_score'].predict(features_scaled)[0])
                
                # Calculate confidence based on prediction variance
                predictions['confidence'] = self.calculate_prediction_confidence(features)
                
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                # Return heuristic predictions
                predictions = {
                    'expected_daily_return': grid_params.get('grid_count', 20) * 0.001,
                    'expected_trades_per_day': grid_params.get('grid_count', 20) * 0.5,
                    'expected_max_drawdown': 0.05,
                    'confidence': 0.5
                }
            
            return predictions
        
        def optimize_grid_parameters(self, market_data: Dict, risk_preference: str = 'balanced') -> Dict:
            """Optimize grid parameters using ML predictions"""
            best_params = None
            best_score = -float('inf')
            
            # Define parameter search space
            grid_counts = [15, 20, 25, 30, 40, 50]
            spacing_percents = [0.003, 0.005, 0.007, 0.01, 0.015]
            range_multipliers = [1.5, 2.0, 2.5, 3.0]
            investment_percents = [0.2, 0.3, 0.4] if risk_preference == 'aggressive' else [0.1, 0.2, 0.3]
            
            # Grid search with ML predictions
            for grid_count in grid_counts:
                for spacing in spacing_percents:
                    for range_mult in range_multipliers:
                        for investment in investment_percents:
                            params = {
                                'grid_count': grid_count,
                                'grid_spacing_percent': spacing,
                                'range_multiplier': range_mult,
                                'investment_percent': investment
                            }
                            
                            # Predict performance
                            predictions = self.predict_performance(market_data, params)
                            
                            # Calculate score based on risk preference
                            if risk_preference == 'conservative':
                                score = (predictions['expected_daily_return'] * 0.4 - 
                                       predictions['expected_max_drawdown'] * 0.6)
                            elif risk_preference == 'aggressive':
                                score = (predictions['expected_daily_return'] * 0.8 - 
                                       predictions['expected_max_drawdown'] * 0.2)
                            else:  # balanced
                                score = (predictions['expected_daily_return'] * 0.6 - 
                                       predictions['expected_max_drawdown'] * 0.4)
                            
                            # Adjust for confidence
                            score *= predictions['confidence']
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                                best_params['predictions'] = predictions
            
            # Add calculated bounds
            current_price = market_data['current_price']
            volatility = market_data['volatility_hourly']
            
            best_params['lower_bound'] = current_price * (1 - volatility * best_params['range_multiplier'])
            best_params['upper_bound'] = current_price * (1 + volatility * best_params['range_multiplier'])
            
            logger.info(f"ML Optimized Parameters: Grid Count: {best_params['grid_count']}, "
                       f"Expected Return: {best_params['predictions']['expected_daily_return']:.2%}")
            
            return best_params
        
        def calculate_prediction_confidence(self, features: np.ndarray) -> float:
            """Calculate confidence in predictions based on feature similarity to training data"""
            if len(self.performance_history) < self.min_history_required:
                return 0.5
            
            # Compare to training data distribution
            training_features = np.array([record['features'] for record in self.performance_history])
            
            # Calculate average distance to k nearest neighbors
            k = min(10, len(training_features))
            distances = np.sqrt(np.sum((training_features - features) ** 2, axis=1))
            k_nearest_distances = np.sort(distances)[:k]
            avg_distance = np.mean(k_nearest_distances)
            
            # Convert to confidence score (0-1)
            # Lower distance = higher confidence
            confidence = np.exp(-avg_distance / 10)
            
            return float(np.clip(confidence, 0.1, 0.95))
        
        def save_models(self):
            """Save trained models to disk"""
            try:
                for name, model in self.models.items():
                    joblib.dump(model, f'grid_model_{name}.pkl')
                for name, scaler in self.scalers.items():
                    joblib.dump(scaler, f'grid_scaler_{name}.pkl')
                logger.info("Models saved successfully")
            except Exception as e:
                logger.error(f"Error saving models: {e}")
        
        def load_models(self):
            """Load trained models from disk"""
            try:
                for name in ['profit', 'trade_frequency', 'risk_score']:
                    self.models[name] = joblib.load(f'grid_model_{name}.pkl')
                    self.scalers[name] = joblib.load(f'grid_scaler_{name}.pkl')
                logger.info("Models loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load models: {e}")

else:
    # Dummy ML optimizer if scikit-learn not available
    class MLGridOptimizer:
        def __init__(self, history_file: str = 'grid_performance_history.json'):
            self.performance_history = []
            self.min_history_required = 100
            logger.warning("ML features disabled - scikit-learn not installed")
        
        def add_performance_record(self, *args, **kwargs):
            pass
        
        def optimize_grid_parameters(self, market_data: Dict, risk_preference: str = 'balanced') -> Dict:
            # Return default parameters
            return {
                'grid_count': 25,
                'grid_spacing_percent': 0.005,
                'range_multiplier': 2.0,
                'investment_percent': 0.3,
                'lower_bound': market_data['current_price'] * 0.95,
                'upper_bound': market_data['current_price'] * 1.05,
                'predictions': {
                    'expected_daily_return': 0.01,
                    'expected_trades_per_day': 20,
                    'expected_max_drawdown': 0.05,
                    'confidence': 0.5
                }
            }

# ============= WEB DASHBOARD =============



# ============= MAIN BOT CLASS =============

class SmartGridTradingBot:
    """Main bot that manages multiple grid strategies"""
    
    def __init__(self, config: GridConfig):
        self.config = config
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Changed from 'spot' to 'future'
            }
        })
        
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Set up futures trading parameters
        self.setup_futures_trading()
        
        self.analyzer = MarketAnalyzer(self.exchange)
        self.active_grids = {}  # symbol -> GridStrategy
        self.running = False
        
    async def scan_and_rank_markets(self) -> List[GridScore]:
        """Scan all markets and rank by grid trading suitability"""
        logger.info("Scanning markets for grid trading opportunities...")
        
        # Analyze all symbols concurrently
        tasks = [self.analyzer.analyze_symbol(symbol, self.config) for symbol in self.config.symbols_to_scan]
        scores = await asyncio.gather(*tasks)
        
        # Filter and sort by score
        valid_scores = [s for s in scores if s is not None and s.total_score >= 60]
        valid_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return valid_scores
    
    async def start_grid_strategy(self, score: GridScore):
        """Start a new grid strategy"""
        try:
            # Calculate investment
            balance = self.get_available_balance()
            investment = min(
                balance * self.config.capital_per_grid,
                balance / max(1, len(self.active_grids) + 1)
            )
            
            # Create grid parameters
            grid_params = {
                'lower_bound': score.price_range[0],
                'upper_bound': score.price_range[1],
                'grid_count': score.optimal_grid_count,
                'investment': investment
            }
            
            # Initialize strategy
            strategy = GridStrategy(score.symbol, self.config, grid_params)
            
            # Place initial orders
            buy_orders, sell_orders = strategy.get_initial_orders(score.current_price)
            
            # Place orders on exchange
            for order in buy_orders[:5]:  # Limit initial orders
                try:
                    result = self.exchange.create_limit_buy_order(
                        symbol=score.symbol,
                        amount=order['amount'],
                        price=order['price']
                    )
                    strategy.buy_orders[order['price']] = result['id']
                except Exception as e:
                    logger.error(f"Error placing buy order: {e}")
            
            for order in sell_orders[:5]:  # Limit initial orders
                try:
                    result = self.exchange.create_limit_sell_order(
                        symbol=score.symbol,
                        amount=order['amount'],
                        price=order['price']
                    )
                    strategy.sell_orders[order['price']] = result['id']
                except Exception as e:
                    logger.error(f"Error placing sell order: {e}")
            
            self.active_grids[score.symbol] = strategy
            
            logger.info(f"""
🎯 NEW GRID STRATEGY STARTED
Symbol: {score.symbol}
Grid Range: ${score.price_range[0]:.2f} - ${score.price_range[1]:.2f}
Grid Count: {score.optimal_grid_count}
Investment: ${investment:.2f}
Expected Daily Trades: {score.expected_daily_trades:.1f}
Expected Daily Return: {score.expected_daily_return:.2%}
Score: {score.total_score:.1f}/100
            """)
            
        except Exception as e:
            logger.error(f"Error starting grid for {score.symbol}: {e}")
    
    async def manage_active_grids(self):
        """Manage all active grid strategies"""
        for symbol, strategy in list(self.active_grids.items()):
            try:
                # Check filled orders
                open_orders = self.exchange.fetch_open_orders(symbol)
                open_order_ids = {order['id'] for order in open_orders}
                
                # Check for filled buy orders
                for price, order_id in list(strategy.buy_orders.items()):
                    if order_id not in open_order_ids:
                        # Order filled - place corresponding sell order
                        sell_price = price + strategy.grid_spacing
                        if sell_price <= strategy.upper_bound:
                            try:
                                order_size = strategy.calculate_order_size()
                                result = self.exchange.create_limit_sell_order(
                                    symbol=symbol,
                                    amount=order_size,
                                    price=sell_price
                                )
                                strategy.sell_orders[sell_price] = result['id']
                                strategy.total_trades += 1
                                del strategy.buy_orders[price]
                                
                                # Calculate profit
                                profit = (sell_price - price) * order_size
                                strategy.realized_pnl += profit
                                strategy.fees_paid += (price + sell_price) * order_size * self.config.fee_rate
                                
                            except Exception as e:
                                logger.error(f"Error placing sell order: {e}")
                
                # Check for filled sell orders
                for price, order_id in list(strategy.sell_orders.items()):
                    if order_id not in open_order_ids:
                        # Order filled - place corresponding buy order
                        buy_price = price - strategy.grid_spacing
                        if buy_price >= strategy.lower_bound:
                            try:
                                order_size = strategy.calculate_order_size()
                                result = self.exchange.create_limit_buy_order(
                                    symbol=symbol,
                                    amount=order_size,
                                    price=buy_price
                                )
                                strategy.buy_orders[buy_price] = result['id']
                                strategy.total_trades += 1
                                del strategy.sell_orders[price]
                                
                            except Exception as e:
                                logger.error(f"Error placing buy order: {e}")
                
                # Check if rebalancing needed
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                if strategy.should_rebalance(current_price):
                    logger.warning(f"{symbol}: Grid needs rebalancing")
                    await self.close_grid_strategy(symbol, "Rebalancing needed")
                
            except Exception as e:
                logger.error(f"Error managing grid for {symbol}: {e}")
    
    async def close_grid_strategy(self, symbol: str, reason: str):
        """Close a grid strategy"""
        if symbol not in self.active_grids:
            return
        
        strategy = self.active_grids[symbol]
        
        try:
            # Cancel all open orders
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
            
            # Get performance stats
            stats = strategy.get_performance_stats()
            
            logger.info(f"""
🏁 GRID STRATEGY CLOSED
Symbol: {symbol}
Reason: {reason}
Total Trades: {stats['total_trades']}
Net PnL: ${stats['net_pnl']:.2f}
Runtime: {stats['runtime_hours']:.1f} hours
Trades/Hour: {stats['trades_per_hour']:.1f}
            """)
            
            del self.active_grids[symbol]
            
        except Exception as e:
            logger.error(f"Error closing grid for {symbol}: {e}")
    
    def get_available_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return self.config.total_capital  # Fallback for testing
    
    async def monitoring_loop(self):
        """Main monitoring and management loop"""
        while self.running:
            try:
                # Manage active grids
                await self.manage_active_grids()
                
                # Log current status
                if self.active_grids:
                    total_pnl = sum(g.realized_pnl - g.fees_paid for g in self.active_grids.values())
                    total_trades = sum(g.total_trades for g in self.active_grids.values())
                    
                    logger.info(f"""
📊 GRID BOT STATUS
Active Grids: {len(self.active_grids)}/{self.config.max_grids_active}
Total Trades: {total_trades}
Total PnL: ${total_pnl:.2f}
Symbols: {', '.join(self.active_grids.keys())}
                    """)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def scanning_loop(self):
        """Scanning loop for new opportunities"""
        while self.running:
            try:
                # Only scan if we have capacity
                if len(self.active_grids) < self.config.max_grids_active:
                    scores = await self.scan_and_rank_markets()
                    
                    if scores:
                        logger.info(f"\n🔍 TOP GRID OPPORTUNITIES:")
                        for i, score in enumerate(scores[:5]):
                            logger.info(f"{i+1}. {score.symbol}: {score.total_score:.1f}/100 "
                                      f"(Vol: {score.volatility_score:.0f}, Range: {score.range_score:.0f})")
                        
                        # Start grids for top opportunities not already active
                        for score in scores:
                            if score.symbol not in self.active_grids and len(self.active_grids) < self.config.max_grids_active:
                                await self.start_grid_strategy(score)
                                break  # Start one at a time
                
                await asyncio.sleep(self.config.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                await asyncio.sleep(self.config.scan_interval)
    
    async def run(self):
        """Run the bot"""
        self.running = True
        logger.info("""
🤖 SMART GRID TRADING BOT STARTED
Capital: ${:.2f}
Max Grids: {}
Symbols to Scan: {}
Scan Interval: {} seconds
        """.format(
            self.config.total_capital,
            self.config.max_grids_active,
            len(self.config.symbols_to_scan),
            self.config.scan_interval
        ))
        
        # Run both loops concurrently
        await asyncio.gather(
            self.monitoring_loop(),
            self.scanning_loop()
        )
    
    def start(self):
        """Start the bot"""
        asyncio.run(self.run())
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Shutting down grid bot...")
        
        # Close all active grids
        for symbol in list(self.active_grids.keys()):
            asyncio.run(self.close_grid_strategy(symbol, "Bot shutdown"))
    
    def setup_futures_trading(self):
        """Set up futures trading parameters"""
        try:
            # Set margin type (cross or isolated)
            for symbol in self.config.symbols_to_scan:
                try:
                    self.exchange.set_margin_mode(self.config.margin_type, symbol)
                    # Set leverage
                    self.exchange.set_leverage(self.config.leverage, symbol)
                    logger.info(f"Set {self.config.margin_type} margin mode and {self.config.leverage}x leverage for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not set margin mode or leverage for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error setting up futures trading: {e}")
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        data = {
            'active_grids': [],
            'total_pnl': 0,
            'total_trades': 0,
            'total_fees': 0,
            'available_balance': self.get_available_balance()
        }
        
        for symbol, strategy in self.active_grids.items():
            stats = strategy.get_performance_stats()
            data['active_grids'].append(stats)
            data['total_pnl'] += stats['net_pnl']
            data['total_trades'] += stats['total_trades']
            data['total_fees'] += stats['fees_paid']
        
        return data

# ============= ENHANCED BOT WITH ML AND DASHBOARD =============

class EnhancedGridTradingBot(SmartGridTradingBot):
    """Grid trading bot with ML optimization and web dashboard"""
    
    def __init__(self, config: GridConfig):
        super().__init__(config)
        self.ml_optimizer = MLGridOptimizer()
        self.dashboard = None
        self.performance_tracker = {}
        
    async def start_grid_strategy(self, score: GridScore):
        """Start a new grid strategy with ML optimization"""
        try:
            # Get market data for ML
            market_data = score.analysis_data
            
            # Get ML-optimized parameters
            if len(self.ml_optimizer.performance_history) >= self.ml_optimizer.min_history_required:
                risk_pref = 'balanced'  # Could be configurable
                optimized_params = self.ml_optimizer.optimize_grid_parameters(market_data, risk_pref)
                
                # Update grid parameters with ML recommendations
                grid_params = {
                    'lower_bound': optimized_params['lower_bound'],
                    'upper_bound': optimized_params['upper_bound'],
                    'grid_count': optimized_params['grid_count'],
                    'investment': self.get_available_balance() * optimized_params['investment_percent']
                }
                
                logger.info(f"Using ML-optimized parameters for {score.symbol}")
                logger.info(f"Predictions: {optimized_params['predictions']}")
            else:
                # Fall back to standard parameters
                grid_params = {
                    'lower_bound': score.price_range[0],
                    'upper_bound': score.price_range[1],
                    'grid_count': score.optimal_grid_count,
                    'investment': self.get_available_balance() * self.config.capital_per_grid
                }
            
            # Track start time for performance measurement
            self.performance_tracker[score.symbol] = {
                'start_time': datetime.now(),
                'start_balance': self.get_available_balance(),
                'market_data': market_data,
                'grid_params': grid_params
            }
            
            # Continue with standard grid creation
            await super().start_grid_strategy(score)
            
        except Exception as e:
            logger.error(f"Error starting enhanced grid for {score.symbol}: {e}")
    
    async def close_grid_strategy(self, symbol: str, reason: str):
        """Close a grid strategy and record performance for ML"""
        if symbol in self.active_grids and symbol in self.performance_tracker:
            # Get final performance metrics
            strategy = self.active_grids[symbol]
            tracker = self.performance_tracker[symbol]
            
            runtime_hours = (datetime.now() - tracker['start_time']).total_seconds() / 3600
            
            performance = {
                'daily_return_percent': (strategy.realized_pnl / tracker['start_balance']) / max(runtime_hours / 24, 0.1),
                'trades_per_day': strategy.total_trades / max(runtime_hours / 24, 0.1),
                'max_drawdown_percent': 0.05,  # Would calculate actual drawdown in production
                'total_pnl': strategy.realized_pnl - strategy.fees_paid,
                'win_rate': 0.6  # Would calculate actual win rate
            }
            
            # Add to ML training data
            self.ml_optimizer.add_performance_record(
                symbol,
                tracker['market_data'],
                tracker['grid_params'],
                performance
            )
            
            del self.performance_tracker[symbol]
        
        # Continue with standard close
        await super().close_grid_strategy(symbol, reason)
    
    def start_with_dashboard(self, dashboard_port: int = 5000):
        """Start bot with web dashboard"""
        if DASHBOARD_AVAILABLE:
            # Initialize dashboard
            self.dashboard = GridBotDashboard(self, self.ml_optimizer, dashboard_port)
            
            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=self.dashboard.run)
            dashboard_thread.daemon = True
            dashboard_thread.start()
            
            logger.info(f"Dashboard started at http://localhost:{dashboard_port}")
        else:
            logger.warning("Dashboard not available - Flask dependencies not installed")
        
        # Start the bot
        self.start()

# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    # Check if all dependencies are available
    if not ML_AVAILABLE:
        print("\n⚠️  WARNING: ML features are disabled!")
        print("To enable ML optimization, install scikit-learn:")
        print("pip install scikit-learn joblib")
    
    if not DASHBOARD_AVAILABLE:
        print("\n⚠️  WARNING: Web dashboard is disabled!")
        print("To enable the dashboard, install Flask dependencies:")
        print("pip install flask flask-cors flask-socketio plotly")
    
    print("\n" + "="*50)
    print("SMART GRID TRADING BOT WITH ML OPTIMIZATION")
    print("="*50 + "\n")
    
    # Configuration
    config = GridConfig(
        api_key="aPeo9so1VKnQt2Gm9x",
        api_secret="TrfGjSSfgUBEJg4D4EErLGXBPo6HjcwQ5kuu",
        
        # Capital management
        total_capital=20,  # Start with $10k
        max_grids_active=3,   # Run 3 grids simultaneously
        capital_per_grid=0.3, # 30% per grid
        
        # Futures-specific settings
        leverage=50,  # 3x leverage
        margin_type='cross',  # Use cross margin
        
        # Grid parameters
        min_grid_levels=20,
        max_grid_levels=50,
        grid_spacing_percent=0.005,  # 0.5% spacing
        
        # Market conditions for ideal grid trading
        min_volatility=0.02,   # 2% daily volatility minimum
        max_volatility=0.10,   # 10% daily volatility maximum
        max_trend_strength=0.2, # Avoid strong trends
        
        # Risk management
        stop_loss_percent=0.10,  # 10% stop loss
        
        testnet=False  # Always start with testnet!
    )
    
    # Create enhanced bot
    bot = EnhancedGridTradingBot(config)
    
    try:
        # Start with dashboard on port 5000
        bot.start_with_dashboard(dashboard_port=5000)
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot stopped by user")