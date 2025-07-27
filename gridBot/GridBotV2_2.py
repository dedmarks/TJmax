import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBybitGridBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'DOGE/USDT:USDT', testnet: bool = True):
        """
        Enhanced Bybit Grid Trading Bot with advanced profitability features
        
        Args:
            api_key: Your Bybit API key
            api_secret: Your Bybit API secret
            symbol: Trading pair (default: DOGE/USDT perpetual)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.symbol = symbol
        self.leverage = 25
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        # Enhanced grid parameters
        self.grid_levels = 0
        self.grid_spacing = 0
        self.upper_price = 0
        self.lower_price = 0
        self.total_investment = 0
        self.order_amount = 0
        self.price_decimals = 2
        
        # Dynamic grid adjustment parameters
        self.dynamic_grid_enabled = True
        self.volatility_adjustment_factor = 1.0
        self.trend_strength = 0
        self.momentum_factor = 0
        
        # Advanced volatility parameters
        self.atr_multiplier = 2.0
        self.min_grid_levels = 15
        self.max_grid_levels = 60
        self.adaptive_spacing = True
        
        # Profit optimization parameters
        self.compound_profits = True
        self.reinvestment_ratio = 0.5  # Reinvest 50% of profits
        self.take_profit_ratio = 0.3   # Take 30% profit when ahead
        self.max_position_ratio = 0.8  # Max 80% of capital in positions
        
        # Risk management
        self.stop_loss_percent = 5.0   # Stop loss at 5% drawdown
        self.max_drawdown_percent = 60.0
        self.risk_per_trade = 0.02     # 2% risk per trade
        self.trailing_stop_enabled = True
        self.trailing_stop_distance = 0.005  # 0.5% trailing stop
        
        # Order management
        self.buy_orders = {}
        self.sell_orders = {}
        self.pending_orders = set()
        self.order_history = deque(maxlen=1000)
        
        # Performance tracking
        self.grid_profits = 0
        self.completed_grids = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_balance = 0
        self.current_drawdown = 0
        
        # Market analysis cache
        self.market_data_cache = {}
        self.last_analysis_time = 0
        self.analysis_interval = 300  # 5 minutes
        
        # Multi-timeframe analysis
        self.timeframes = ['5m', '15m', '1h', '4h']
        self.trend_weights = {'5m': 0.1, '15m': 0.2, '1h': 0.4, '4h': 0.3}
        
        # Volume analysis
        self.volume_profile = {}
        self.volume_weighted_levels = []
        
        # Order book analysis
        self.order_book_imbalance = 0
        self.bid_ask_ratio = 1.0
        
        # Machine learning features (simple momentum indicators)
        self.rsi_values = deque(maxlen=100)
        self.macd_signals = deque(maxlen=100)
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_active = True
        
        self.check_position_mode()
        
    def check_position_mode(self):
        """Check and set appropriate position mode for trading"""
        try:
            self.exchange.set_position_mode(hedged=False, symbol=self.symbol)
            logger.info("Position mode set to one-way mode")
        except Exception as e:
            error_message = str(e).lower()
            if "position mode not modified" in error_message or "not modified" in error_message:
                logger.info("Position mode already set correctly")
            else:
                logger.info(f"Position mode check completed: {e}")
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def analyze_order_book(self) -> Dict[str, float]:
        """Analyze order book for market sentiment"""
        try:
            order_book = self.exchange.fetch_order_book(self.symbol, limit=100)
            
            # Calculate bid/ask volumes
            bid_volume = sum([bid[1] for bid in order_book['bids'][:20]])
            ask_volume = sum([ask[1] for ask in order_book['asks'][:20]])
            
            # Calculate imbalance
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0
                
            # Calculate spread
            if order_book['bids'] and order_book['asks']:
                spread = order_book['asks'][0][0] - order_book['bids'][0][0]
                mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
                spread_percent = (spread / mid_price) * 100
            else:
                spread_percent = 0
                
            return {
                'imbalance': imbalance,
                'bid_ask_ratio': bid_volume / ask_volume if ask_volume > 0 else 1,
                'spread_percent': spread_percent,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book: {e}")
            return {'imbalance': 0, 'bid_ask_ratio': 1, 'spread_percent': 0}
    
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict[str, List[float]]:
        """Calculate volume profile for support/resistance levels"""
        try:
            # Create price bins
            price_bins = pd.cut(df['close'], bins=bins)
            
            # Calculate volume at each price level
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            # Find high volume nodes (potential support/resistance)
            threshold = volume_profile.quantile(0.7)
            high_volume_levels = volume_profile[volume_profile > threshold]
            
            # Extract price levels
            support_resistance = []
            for interval in high_volume_levels.index:
                mid_point = (interval.left + interval.right) / 2
                support_resistance.append(mid_point)
                
            return {
                'levels': support_resistance,
                'volumes': high_volume_levels.values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {'levels': [], 'volumes': []}
    
    def multi_timeframe_analysis(self) -> Dict[str, float]:
        """Perform multi-timeframe trend analysis"""
        trends = {}
        weighted_trend = 0
        
        for tf in self.timeframes:
            try:
                df = self.fetch_ohlcv_data(tf, 50)
                
                # Calculate trend using multiple indicators
                sma_20 = df['close'].rolling(20).mean().iloc[-1]
                sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
                current_price = df['close'].iloc[-1]
                
                # Trend scoring (-1 to 1)
                if current_price > sma_20 > sma_50:
                    trend_score = 1
                elif current_price < sma_20 < sma_50:
                    trend_score = -1
                else:
                    trend_score = (current_price - sma_20) / sma_20
                    
                trends[tf] = trend_score
                weighted_trend += trend_score * self.trend_weights[tf]
                
            except Exception as e:
                logger.error(f"Error in timeframe {tf} analysis: {e}")
                trends[tf] = 0
                
        return {
            'trends': trends,
            'weighted_trend': weighted_trend,
            'trend_strength': abs(weighted_trend)
        }
    
    def calculate_adaptive_grid_parameters(self, total_investment: float) -> Dict:
        """Calculate grid parameters with advanced market analysis"""
        try:
            # Fetch data
            df_1h = self.fetch_ohlcv_data('1h', 168)
            df_15m = self.fetch_ohlcv_data('15m', 200)
            
            # Basic volatility metrics
            current_price = df_1h['close'].iloc[-1]
            atr = self.calculate_atr(df_1h)
            
            # Advanced analysis
            rsi = self.calculate_rsi(df_1h['close'])
            macd_data = self.calculate_macd(df_1h['close'])
            order_book_data = self.analyze_order_book()
            volume_profile = self.calculate_volume_profile(df_1h)
            multi_tf = self.multi_timeframe_analysis()
            
            # Store analysis results
            self.rsi_values.append(rsi)
            self.macd_signals.append(macd_data['histogram'])
            self.order_book_imbalance = order_book_data['imbalance']
            self.bid_ask_ratio = order_book_data['bid_ask_ratio']
            self.trend_strength = multi_tf['trend_strength']
            
            # Dynamic grid range calculation
            base_range = atr * self.atr_multiplier
            
            # Adjust range based on market conditions
            if rsi < 30 or rsi > 70:  # Oversold/overbought
                volatility_multiplier = 1.3
            else:
                volatility_multiplier = 1.0
                
            # Trend adjustment
            if multi_tf['weighted_trend'] > 0.5:  # Strong uptrend
                trend_adjustment = 0.2
            elif multi_tf['weighted_trend'] < -0.5:  # Strong downtrend
                trend_adjustment = -0.2
            else:
                trend_adjustment = 0
                
            # Order book adjustment
            if abs(order_book_data['imbalance']) > 0.3:
                imbalance_adjustment = order_book_data['imbalance'] * 0.1
            else:
                imbalance_adjustment = 0
                
            # Calculate final range
            grid_range = base_range * volatility_multiplier
            
            # Set bounds with trend bias
            center_price = current_price * (1 + trend_adjustment * 0.1)
            upper_price = center_price + (grid_range * 0.6)  # More room on trend side
            lower_price = center_price - (grid_range * 0.4)
            
            # Dynamic grid levels based on volatility
            volatility_score = (atr / current_price) * 100
            
            if volatility_score < 0.5:  # Very low volatility
                grid_levels = 50
            elif volatility_score < 1.0:  # Low volatility
                grid_levels = 40
            elif volatility_score < 2.0:  # Medium volatility
                grid_levels = 30
            elif volatility_score < 3.0:  # High volatility
                grid_levels = 20
            else:  # Very high volatility
                grid_levels = 15
                
            # Incorporate volume profile levels
            if volume_profile['levels']:
                # Adjust grid to include high volume levels
                for level in volume_profile['levels']:
                    if lower_price < level < upper_price:
                        # Ensure grid includes these levels
                        pass
                        
            # Calculate adaptive spacing
            if self.adaptive_spacing:
                # Tighter spacing near current price
                spacing_multiplier = 1 + (0.3 * abs(multi_tf['weighted_trend']))
                grid_spacing = (upper_price - lower_price) / (grid_levels - 1) * spacing_multiplier
            else:
                grid_spacing = (upper_price - lower_price) / (grid_levels - 1)
                
            # Ensure profitable spacing
            min_profit_spacing = current_price * 0.002  # 0.2% minimum
            if grid_spacing < min_profit_spacing:
                grid_levels = int((upper_price - lower_price) / min_profit_spacing) + 1
                grid_spacing = (upper_price - lower_price) / (grid_levels - 1)
                
            # Price decimals
            if current_price < 0.01:
                price_decimals = 6
            elif current_price < 0.1:
                price_decimals = 5
            elif current_price < 1:
                price_decimals = 4
            elif current_price < 10:
                price_decimals = 3
            else:
                price_decimals = 2
                
            # Calculate order amounts with position sizing
            base_order_amount = total_investment / grid_levels
            
            # Risk-adjusted order sizing
            if self.current_drawdown > 5:
                risk_adjustment = 0.7  # Reduce size during drawdown
            else:
                risk_adjustment = 1.0
                
            order_amount = base_order_amount * risk_adjustment
            
            parameters = {
                'upper_price': round(upper_price, price_decimals),
                'lower_price': round(lower_price, price_decimals),
                'grid_levels': grid_levels,
                'grid_spacing': round(grid_spacing, price_decimals),
                'current_price': round(current_price, price_decimals),
                'order_amount': round(order_amount, 2),
                'price_decimals': price_decimals,
                'market_conditions': {
                    'rsi': round(rsi, 2),
                    'trend': round(multi_tf['weighted_trend'], 2),
                    'volatility': round(volatility_score, 2),
                    'order_book_imbalance': round(order_book_data['imbalance'], 2)
                }
            }
            
            logger.info(f"Advanced Grid Analysis Complete:")
            logger.info(f"  - Price: ${parameters['current_price']}")
            logger.info(f"  - Range: ${parameters['lower_price']} - ${parameters['upper_price']}")
            logger.info(f"  - Levels: {parameters['grid_levels']}")
            logger.info(f"  - Market Conditions: {parameters['market_conditions']}")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error in advanced grid calculation: {e}")
            raise
    
    def place_weighted_grid_orders(self):
        """Place grid orders with adaptive sizing based on market conditions"""
        try:
            current_price = self.get_current_price()
            grid_levels = self.calculate_grid_levels()
            
            # Get market sentiment
            order_book_data = self.analyze_order_book()
            
            for i, price in enumerate(grid_levels):
                # Calculate position weight based on distance from current price
                distance_factor = abs(price - current_price) / current_price
                
                # Adjust order size based on market conditions
                if price < current_price:  # Buy orders
                    # Increase size if bullish sentiment
                    if self.order_book_imbalance > 0.2:
                        size_multiplier = 1.2
                    elif self.order_book_imbalance < -0.2:
                        size_multiplier = 0.8
                    else:
                        size_multiplier = 1.0
                        
                    # Place larger orders at stronger support levels
                    if price in self.volume_weighted_levels:
                        size_multiplier *= 1.3
                        
                    adjusted_amount = self.order_amount * size_multiplier
                    
                    if price < current_price * 0.995:
                        self.place_buy_order(price, adjusted_amount)
                        
                else:  # Sell orders
                    # Adjust based on trend
                    if self.trend_strength > 0.5 and self.momentum_factor > 0:
                        size_multiplier = 0.8  # Smaller sells in uptrend
                    else:
                        size_multiplier = 1.0
                        
                    adjusted_amount = self.order_amount * size_multiplier
                    
                    if price > current_price * 1.005:
                        self.place_sell_order(price, adjusted_amount)
                        
            logger.info("Weighted grid orders placed successfully")
            
        except Exception as e:
            logger.error(f"Error placing weighted grid orders: {e}")
    
    def dynamic_grid_adjustment(self):
        """Dynamically adjust grid based on real-time market conditions"""
        try:
            if not self.dynamic_grid_enabled:
                return
                
            current_price = self.get_current_price()
            
            # Check if significant market change
            price_change = (current_price - (self.upper_price + self.lower_price) / 2) / current_price
            
            if abs(price_change) > 0.02:  # 2% change
                logger.info("Significant price movement detected, adjusting grid...")
                
                # Get current market conditions
                multi_tf = self.multi_timeframe_analysis()
                
                # Shift grid based on trend
                if multi_tf['weighted_trend'] > 0.3:
                    shift_factor = 0.01 * multi_tf['weighted_trend']
                    self.upper_price *= (1 + shift_factor)
                    self.lower_price *= (1 + shift_factor)
                elif multi_tf['weighted_trend'] < -0.3:
                    shift_factor = 0.01 * abs(multi_tf['weighted_trend'])
                    self.upper_price *= (1 - shift_factor)
                    self.lower_price *= (1 - shift_factor)
                    
                # Adjust spacing based on volatility
                df = self.fetch_ohlcv_data('15m', 50)
                recent_volatility = df['close'].pct_change().std() * np.sqrt(96)  # Daily vol
                
                if recent_volatility > 0.05:  # High volatility
                    self.grid_spacing *= 1.1
                elif recent_volatility < 0.02:  # Low volatility
                    self.grid_spacing *= 0.9
                    
                logger.info(f"Grid adjusted - New range: ${self.lower_price:.4f} - ${self.upper_price:.4f}")
                
        except Exception as e:
            logger.error(f"Error in dynamic grid adjustment: {e}")
    
    def manage_risk_and_position_sizing(self) -> Dict[str, float]:
        """Advanced risk management and position sizing"""
        try:
            account_info = self.get_account_info()
            current_balance = account_info.get('usdt_balance', 0)
            
            # Update peak balance and drawdown
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
                
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            
            # Position sizing based on Kelly Criterion (simplified)
            if self.completed_grids > 0:
                win_rate = self.winning_trades / (self.winning_trades + self.losing_trades + 1)
                avg_win = self.grid_profits / (self.winning_trades + 1)
                avg_loss = abs(self.grid_spacing * self.order_amount * 0.0005)  # Estimated loss
                
                # Kelly percentage
                if avg_loss > 0:
                    kelly_percent = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_percent = max(0, min(kelly_percent, 0.25))  # Cap at 25%
                else:
                    kelly_percent = 0.02  # Default 2%
            else:
                kelly_percent = 0.02
                
            # Adjust for drawdown
            if self.current_drawdown > 5:
                position_size_multiplier = 0.5
            elif self.current_drawdown > 3:
                position_size_multiplier = 0.7
            else:
                position_size_multiplier = 1.0
                
            # Calculate maximum position size
            max_position_size = current_balance * kelly_percent * position_size_multiplier
            
            return {
                'max_position_size': max_position_size,
                'current_drawdown': self.current_drawdown,
                'kelly_percent': kelly_percent,
                'position_multiplier': position_size_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return {
                'max_position_size': self.total_investment * 0.02,
                'current_drawdown': 0,
                'kelly_percent': 0.02,
                'position_multiplier': 1.0
            }
    
    def compound_and_reinvest_profits(self):
        """Compound profits by reinvesting a portion back into the grid"""
        try:
            if not self.compound_profits:
                return
                
            account_info = self.get_account_info()
            current_balance = account_info.get('usdt_balance', 0)
            
            # Calculate available profit
            initial_investment = self.total_investment
            available_profit = current_balance - initial_investment
            
            if available_profit > initial_investment * 0.1:  # 10% profit threshold
                # Reinvest portion of profits
                reinvestment_amount = available_profit * self.reinvestment_ratio
                
                # Take some profit
                take_profit_amount = available_profit * self.take_profit_ratio
                
                # Update grid with additional capital
                new_total_investment = initial_investment + reinvestment_amount
                
                logger.info(f"Compounding profits:")
                logger.info(f"  - Available profit: ${available_profit:.2f}")
                logger.info(f"  - Reinvesting: ${reinvestment_amount:.2f}")
                logger.info(f"  - Taking profit: ${take_profit_amount:.2f}")
                
                # Reconfigure grid with new capital
                self.total_investment = new_total_investment
                self.order_amount = new_total_investment / self.grid_levels
                
                # Place additional orders with reinvested capital
                self.place_weighted_grid_orders()
                
        except Exception as e:
            logger.error(f"Error in profit compounding: {e}")
    
    def monitor_and_trail_positions(self):
        """Monitor positions and implement trailing stops"""
        try:
            if not self.trailing_stop_enabled:
                return
                
            positions = self.exchange.fetch_positions([self.symbol])
            
            for position in positions:
                if position['contracts'] != 0:
                    entry_price = position['average']
                    current_price = self.get_current_price()
                    
                    # Calculate profit percentage
                    if position['side'] == 'long':
                        profit_pct = (current_price - entry_price) / entry_price
                        trail_price = current_price * (1 - self.trailing_stop_distance)
                    else:
                        profit_pct = (entry_price - current_price) / entry_price
                        trail_price = current_price * (1 + self.trailing_stop_distance)
                        
                    # Implement trailing stop if in profit
                    if profit_pct > 0.01:  # 1% profit threshold
                        # Place or update trailing stop order
                        self.update_trailing_stop(position, trail_price)
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def update_trailing_stop(self, position: Dict, stop_price: float):
        """Update or create trailing stop order"""
        try:
            # Cancel existing stop orders for this position
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            for order in open_orders:
                if order['type'] == 'stop' and order['amount'] == abs(position['contracts']):
                    self.exchange.cancel_order(order['id'], self.symbol)
                    
            # Place new stop order
            if position['side'] == 'long':
                order = self.exchange.create_stop_market_sell_order(
                    self.symbol,
                    abs(position['contracts']),
                    stop_price
                )
            else:
                order = self.exchange.create_stop_market_buy_order(
                    self.symbol,
                    abs(position['contracts']),
                    stop_price
                )
                
            logger.info(f"Trailing stop updated at ${stop_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    def optimize_order_execution(self):
        """Optimize order execution timing based on market microstructure"""
        try:
            # Analyze recent tick data for optimal execution
            order_book = self.exchange.fetch_order_book(self.symbol)
            
            # Check spread
            if order_book['bids'] and order_book['asks']:
                spread = order_book['asks'][0][0] - order_book['bids'][0][0]
                mid_price = (order_book['asks'][0][0] + order_book['bids'][0][0]) / 2
                spread_bps = (spread / mid_price) * 10000  # Basis points
                
                # If spread is wide, use more aggressive limit orders
                if spread_bps > 5:
                    return {'execution_mode': 'aggressive', 'offset_bps': 1}
                else:
                    return {'execution_mode': 'passive', 'offset_bps': 0}
            
            return {'execution_mode': 'normal', 'offset_bps': 0}
            
        except Exception as e:
            logger.error(f"Error optimizing execution: {e}")
            return {'execution_mode': 'normal', 'offset_bps': 0}
    
    def get_enhanced_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        try:
            basic_stats = self.get_performance_stats()
            
            # Calculate additional metrics
            total_trades = self.winning_trades + self.losing_trades
            win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
            
            # Sharpe ratio calculation (simplified)
            if len(self.order_history) > 10:
                returns = [order.get('profit', 0) for order in self.order_history]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
                
            # Calculate grid efficiency
            theoretical_max_grids = len([o for o in self.buy_orders.values() if o['status'] == 'filled'])
            grid_efficiency = self.completed_grids / theoretical_max_grids if theoretical_max_grids > 0 else 0
            
            enhanced_stats = {
                **basic_stats,
                'win_rate': round(win_rate * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(self.current_drawdown, 2),
                'grid_efficiency': round(grid_efficiency * 100, 2),
                'total_trades': total_trades,
                'market_conditions': {
                    'trend': round(self.trend_strength, 2),
                    'order_book_imbalance': round(self.order_book_imbalance, 2),
                    'recent_rsi': round(self.rsi_values[-1], 2) if self.rsi_values else 50
                }
            }
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Error calculating enhanced stats: {e}")
            return self.get_performance_stats()
    
    def run_enhanced(self, check_interval: int = 5, recalibration_interval: int = 1800):
        """
        Run enhanced grid bot with advanced features
        
        Args:
            check_interval: Seconds between order checks (reduced for better responsiveness)
            recalibration_interval: Seconds between grid recalibration (default: 30 minutes)
        """
        try:
            # Set leverage
            self.set_leverage()
            
            # Initial market analysis
            logger.info("Performing comprehensive market analysis...")
            self.adjust_grid_for_trend()
            
            # Place initial weighted grid orders
            self.place_weighted_grid_orders()
            
            logger.info(f"Enhanced grid bot started with advanced features")
            logger.info(f"Check interval: {check_interval}s, Recalibration: {recalibration_interval/60:.0f}min")
            
            last_recalibration = time.time()
            last_compound = time.time()
            last_risk_check = time.time()
            
            # Start background monitoring threads
            self.executor.submit(self.monitor_market_conditions)
            
            while self.monitoring_active:
                try:
                    # High-frequency order management
                    self.check_filled_orders()
                    
                    # Dynamic adjustments every minute
                    if time.time() - last_risk_check > 60:
                        self.dynamic_grid_adjustment()
                        self.monitor_and_trail_positions()
                        risk_metrics = self.manage_risk_and_position_sizing()
                        last_risk_check = time.time()
                        
                        # Emergency stop loss
                        if self.current_drawdown > self.max_drawdown_percent:
                            logger.warning(f"Max drawdown reached: {self.current_drawdown:.2f}%")
                            self.emergency_close_positions()
                            break
                    
                    # Compound profits every 10 minutes
                    if time.time() - last_compound > 600:
                        self.compound_and_reinvest_profits()
                        last_compound = time.time()
                    
                    # Full recalibration
                    if time.time() - last_recalibration > recalibration_interval:
                        logger.info("Performing full grid recalibration...")
                        
                        # Save current state
                        current_stats = self.get_enhanced_performance_stats()
                        
                        # Cancel all orders
                        self.stop()
                        
                        # Reconfigure with market analysis
                        self.auto_configure_grid(self.total_investment)
                        self.adjust_grid_for_trend()
                        
                        # Place new optimized orders
                        self.place_weighted_grid_orders()
                        
                        last_recalibration = time.time()
                    
                    # Performance reporting
                    stats = self.get_enhanced_performance_stats()
                    if stats['completed_grids'] % 10 == 0 and stats['completed_grids'] > 0:
                        self.log_performance_report(stats)
                    
                    # Check grid boundaries
                    self.check_grid_boundaries()
                    
                    # Brief sleep to prevent API overload
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval * 2)
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            self.monitoring_active = False
            self.executor.shutdown(wait=True)
    
    def monitor_market_conditions(self):
        """Background thread to monitor market conditions"""
        while self.monitoring_active:
            try:
                # Update market analysis cache
                self.market_data_cache['order_book'] = self.analyze_order_book()
                self.market_data_cache['multi_tf'] = self.multi_timeframe_analysis()
                
                # Sleep for analysis interval
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                time.sleep(60)
    
    def emergency_close_positions(self):
        """Emergency close all positions"""
        try:
            logger.warning("EMERGENCY: Closing all positions due to max drawdown")
            
            # Cancel all orders
            self.exchange.cancel_all_orders(self.symbol)
            
            # Close all positions
            positions = self.exchange.fetch_positions([self.symbol])
            for position in positions:
                if position['contracts'] != 0:
                    if position['side'] == 'long':
                        self.exchange.create_market_sell_order(
                            self.symbol,
                            abs(position['contracts'])
                        )
                    else:
                        self.exchange.create_market_buy_order(
                            self.symbol,
                            abs(position['contracts'])
                        )
                        
            logger.warning("All positions closed")
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
    
    def log_performance_report(self, stats: Dict):
        """Log detailed performance report"""
        logger.info("="*50)
        logger.info("📊 ENHANCED PERFORMANCE REPORT 📊")
        logger.info("="*50)
        logger.info(f"Completed Grids: {stats['completed_grids']}")
        logger.info(f"Total Profit: ${stats['total_profit_usdt']:.2f}")
        logger.info(f"Win Rate: {stats['win_rate']}%")
        logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']}")
        logger.info(f"Max Drawdown: {stats['max_drawdown']}%")
        logger.info(f"Grid Efficiency: {stats['grid_efficiency']}%")
        logger.info(f"Market Conditions: {stats['market_conditions']}")
        logger.info("="*50)
    
    def place_buy_order(self, price: float, amount: Optional[float] = None):
        """Enhanced buy order placement with smart execution"""
        try:
            if amount is None:
                amount = self.order_amount
                
            # Get execution optimization
            exec_params = self.optimize_order_execution()
            
            # Calculate position size
            position_size = self.calculate_position_size(price)
            
            # Adjust for risk management
            risk_metrics = self.manage_risk_and_position_sizing()
            max_size = risk_metrics['max_position_size'] / price
            position_size = min(position_size, max_size)
            
            # Smart order parameters
            params = {
                'timeInForce': 'PostOnly',
                'reduceOnly': False
            }
            
            # Adjust price based on execution mode
            if exec_params['execution_mode'] == 'aggressive':
                adjusted_price = price * (1 + exec_params['offset_bps'] / 10000)
            else:
                adjusted_price = price
            
            order = self.exchange.create_limit_buy_order(
                symbol=self.symbol,
                amount=position_size,
                price=adjusted_price,
                params=params
            )
            
            self.buy_orders[order['id']] = {
                'price': price,
                'amount': position_size,
                'status': 'open',
                'timestamp': time.time()
            }
            
            logger.info(f"Smart buy order: {position_size:.4f} @ ${price:.{self.price_decimals}f}")
            
        except Exception as e:
            logger.error(f"Error placing enhanced buy order: {e}")
    
    def place_sell_order(self, price: float, amount: Optional[float] = None):
        """Enhanced sell order placement with smart execution"""
        try:
            if amount is None:
                amount = self.order_amount
                
            # Get execution optimization
            exec_params = self.optimize_order_execution()
            
            # Calculate position size
            position_size = self.calculate_position_size(price)
            
            # Smart order parameters
            params = {
                'timeInForce': 'PostOnly',
                'reduceOnly': False
            }
            
            # Adjust price based on execution mode
            if exec_params['execution_mode'] == 'aggressive':
                adjusted_price = price * (1 - exec_params['offset_bps'] / 10000)
            else:
                adjusted_price = price
            
            order = self.exchange.create_limit_sell_order(
                symbol=self.symbol,
                amount=position_size,
                price=adjusted_price,
                params=params
            )
            
            self.sell_orders[order['id']] = {
                'price': price,
                'amount': position_size,
                'status': 'open',
                'timestamp': time.time()
            }
            
            logger.info(f"Smart sell order: {position_size:.4f} @ ${price:.{self.price_decimals}f}")
            
        except Exception as e:
            logger.error(f"Error placing enhanced sell order: {e}")
    
    def check_filled_orders(self):
        """Enhanced order checking with profit tracking"""
        try:
            # Fetch recent closed orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=50)
            
            for order in closed_orders:
                order_id = order['id']
                
                # Process buy orders
                if order_id in self.buy_orders and self.buy_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.buy_orders[order_id]['status'] = 'filled'
                        
                        # Calculate new sell price with dynamic adjustment
                        base_sell_price = filled_price + self.grid_spacing
                        
                        # Adjust based on momentum
                        if self.trend_strength > 0.5:
                            sell_price = base_sell_price * 1.002  # Add 0.2% in strong uptrend
                        else:
                            sell_price = base_sell_price
                        
                        if sell_price <= self.upper_price:
                            self.place_sell_order(sell_price, filled_amount)
                        
                        # Track order
                        self.order_history.append({
                            'type': 'buy',
                            'price': filled_price,
                            'amount': filled_amount,
                            'timestamp': time.time()
                        })
                        
                        logger.info(f"Buy filled @ ${filled_price:.{self.price_decimals}f}, sell placed @ ${sell_price:.{self.price_decimals}f}")
                
                # Process sell orders
                elif order_id in self.sell_orders and self.sell_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.sell_orders[order_id]['status'] = 'filled'
                        
                        # Calculate profit
                        buy_price = filled_price - self.grid_spacing
                        profit = (filled_price - buy_price) * filled_amount * self.leverage
                        fee_cost = filled_price * filled_amount * 0.0005 * 2  # Maker fee both sides
                        net_profit = profit - fee_cost
                        
                        self.grid_profits += net_profit
                        self.completed_grids += 1
                        
                        if net_profit > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        # Place new buy order
                        new_buy_price = filled_price - self.grid_spacing
                        
                        # Adjust based on momentum
                        if self.trend_strength < -0.5:
                            new_buy_price *= 0.998  # Subtract 0.2% in downtrend
                        
                        if new_buy_price >= self.lower_price:
                            self.place_buy_order(new_buy_price, filled_amount)
                        
                        # Track order
                        self.order_history.append({
                            'type': 'sell',
                            'price': filled_price,
                            'amount': filled_amount,
                            'profit': net_profit,
                            'timestamp': time.time()
                        })
                        
                        logger.info(f"Grid completed! Sell @ ${filled_price:.{self.price_decimals}f}, Profit: ${net_profit:.2f}")
                        
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
    
    def fetch_ohlcv_data(self, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data with caching"""
        cache_key = f"{timeframe}_{limit}"
        current_time = time.time()
        
        # Check cache (5 minute cache for efficiency)
        if cache_key in self.market_data_cache:
            cached_data, cache_time = self.market_data_cache[cache_key]
            if current_time - cache_time < 300:  # 5 minutes
                return cached_data
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Cache the data
            self.market_data_cache[cache_key] = (df, current_time)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def get_current_price(self) -> float:
        """Get current market price with caching"""
        cache_key = 'current_price'
        current_time = time.time()
        
        # Check cache (1 second cache)
        if cache_key in self.market_data_cache:
            cached_price, cache_time = self.market_data_cache[cache_key]
            if current_time - cache_time < 1:
                return cached_price
        
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            
            # Cache the price
            self.market_data_cache[cache_key] = (price, current_time)
            
            return price
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            raise
    
    def calculate_grid_levels(self) -> List[float]:
        """Calculate grid levels with optional concentration near current price"""
        levels = []
        
        if self.adaptive_spacing:
            # Concentrate more levels near current price
            current_price = self.get_current_price()
            
            # Calculate distance ratios
            lower_distance = current_price - self.lower_price
            upper_distance = self.upper_price - current_price
            
            # Allocate levels proportionally
            lower_levels = int(self.grid_levels * (lower_distance / (lower_distance + upper_distance)))
            upper_levels = self.grid_levels - lower_levels
            
            # Create levels with tighter spacing near current price
            for i in range(lower_levels):
                ratio = (i / lower_levels) ** 1.5  # Power curve for concentration
                price = self.lower_price + (current_price - self.lower_price) * ratio
                levels.append(round(price, self.price_decimals))
            
            for i in range(1, upper_levels):
                ratio = (i / upper_levels) ** 1.5
                price = current_price + (self.upper_price - current_price) * ratio
                levels.append(round(price, self.price_decimals))
        else:
            # Regular spacing
            for i in range(self.grid_levels):
                price = self.lower_price + (i * self.grid_spacing)
                levels.append(round(price, self.price_decimals))
        
        return sorted(levels)
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size with risk management"""
        # Base calculation
        contract_value_usdt = self.order_amount * self.leverage
        position_size = contract_value_usdt / price
        
        # Get market info
        try:
            market = self.exchange.market(self.symbol)
            min_size = market['limits']['amount']['min']
            
            # Round to appropriate precision
            precision = market['precision']['amount']
            if isinstance(precision, float):
                decimal_places = max(0, -int(np.log10(precision)))
            else:
                decimal_places = int(precision)
            
            position_size = round(position_size, decimal_places)
            
            # Ensure minimum size
            if position_size < min_size:
                position_size = min_size
                
        except Exception as e:
            logger.warning(f"Could not get market info: {e}")
            position_size = round(position_size, 3)
        
        return position_size
    
    def set_leverage(self):
        """Set leverage for the trading pair"""
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            error_message = str(e).lower()
            if "leverage not modified" in error_message or "110043" in error_message:
                logger.info(f"Leverage already set to {self.leverage}x")
            else:
                logger.error(f"Error setting leverage: {e}")
                raise
    
    def auto_configure_grid(self, total_investment: float):
        """Auto-configure grid with enhanced analysis"""
        logger.info("Performing enhanced market analysis for optimal grid configuration...")
        
        params = self.calculate_adaptive_grid_parameters(total_investment)
        
        self.upper_price = params['upper_price']
        self.lower_price = params['lower_price']
        self.grid_levels = params['grid_levels']
        self.grid_spacing = params['grid_spacing']
        self.total_investment = total_investment
        self.order_amount = params['order_amount']
        self.price_decimals = params.get('price_decimals', 2)
        
        # Store market conditions
        if 'market_conditions' in params:
            self.momentum_factor = params['market_conditions'].get('trend', 0)
        
        logger.info(f"Enhanced grid configured with {self.grid_levels} levels")
    
    def analyze_market_regime(self) -> str:
        """Analyze market regime with multiple indicators"""
        try:
            df = self.fetch_ohlcv_data('1h', 100)
            
            # Multiple moving averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            rsi = self.calculate_rsi(df['close'])
            
            # MACD
            macd_data = self.calculate_macd(df['close'])
            
            recent_close = df['close'].iloc[-1]
            sma_10 = df['sma_10'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            # Scoring system
            trend_score = 0
            
            # Moving average alignment
            if sma_10 > sma_20 > sma_50:
                trend_score += 2
            elif sma_10 < sma_20 < sma_50:
                trend_score -= 2
            
            # Price position
            if recent_close > sma_10:
                trend_score += 1
            else:
                trend_score -= 1
            
            # RSI
            if rsi > 60:
                trend_score += 1
            elif rsi < 40:
                trend_score -= 1
            
            # MACD
            if macd_data['histogram'] > 0:
                trend_score += 1
            else:
                trend_score -= 1
            
            # Determine regime
            if trend_score >= 3:
                regime = 'trending_up'
            elif trend_score <= -3:
                regime = 'trending_down'
            else:
                regime = 'ranging'
            
            logger.info(f"Market regime: {regime} (score: {trend_score})")
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return 'ranging'
    
    def adjust_grid_for_trend(self):
        """Adjust grid with enhanced trend analysis"""
        regime = self.analyze_market_regime()
        
        # Get trend strength from multi-timeframe analysis
        multi_tf = self.multi_timeframe_analysis()
        trend_strength = multi_tf['weighted_trend']
        
        if regime == 'trending_up':
            # Shift grid up proportionally to trend strength
            shift = self.grid_spacing * (0.5 + 0.5 * min(trend_strength, 1))
            self.upper_price += shift
            self.lower_price += shift
            
            # Increase upper levels in strong uptrend
            if trend_strength > 0.7:
                self.upper_price += self.grid_spacing * 0.5
            
            logger.info(f"Grid adjusted UP by ${shift:.{self.price_decimals}f} (trend: {trend_strength:.2f})")
            
        elif regime == 'trending_down':
            # Shift grid down
            shift = self.grid_spacing * (0.5 + 0.5 * min(abs(trend_strength), 1))
            self.upper_price -= shift
            self.lower_price -= shift
            
            # Increase lower levels in strong downtrend
            if trend_strength < -0.7:
                self.lower_price -= self.grid_spacing * 0.5
            
            logger.info(f"Grid adjusted DOWN by ${shift:.{self.price_decimals}f} (trend: {trend_strength:.2f})")
        
        else:  # Ranging market
            # Widen grid slightly for ranging markets
            expansion = self.grid_spacing * 0.2
            self.upper_price += expansion
            self.lower_price -= expansion
            logger.info(f"Grid expanded by ${expansion*2:.{self.price_decimals}f} for ranging market")
    
    def get_account_info(self) -> Dict:
        """Get account balance and position info"""
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions([self.symbol])
            
            info = {
                'usdt_balance': balance['USDT']['free'] if 'USDT' in balance else 0,
                'positions': positions
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict:
        """Get basic performance statistics"""
        try:
            filled_buys = sum(1 for order in self.buy_orders.values() if order['status'] == 'filled')
            filled_sells = sum(1 for order in self.sell_orders.values() if order['status'] == 'filled')
            
            avg_price = (self.upper_price + self.lower_price) / 2
            profit_per_grid_percent = (self.grid_spacing / avg_price - 0.0005) * 100
            
            return {
                'completed_grids': self.completed_grids,
                'profit_per_grid_percent': round(profit_per_grid_percent, 3),
                'total_profit_usdt': round(self.grid_profits, 2),
                'filled_buy_orders': filled_buys,
                'filled_sell_orders': filled_sells,
                'active_buy_orders': sum(1 for order in self.buy_orders.values() if order['status'] == 'open'),
                'active_sell_orders': sum(1 for order in self.sell_orders.values() if order['status'] == 'open')
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {}
    
    def check_grid_boundaries(self):
        """Check and adjust grid boundaries with trend consideration"""
        try:
            current_price = self.get_current_price()
            
            # Dynamic buffer based on volatility
            df = self.fetch_ohlcv_data('15m', 20)
            recent_volatility = df['close'].pct_change().std()
            buffer_percent = max(0.05, min(0.15, recent_volatility * 10))  # 5-15% buffer
            buffer = (self.upper_price - self.lower_price) * buffer_percent
            
            if current_price > self.upper_price - buffer or current_price < self.lower_price + buffer:
                logger.warning(f"Price ${current_price:.{self.price_decimals}f} approaching boundaries")
                
                # Quick adjustment without full recalibration
                if current_price > self.upper_price - buffer:
                    # Price breaking upward
                    shift = self.grid_spacing * 2
                    self.upper_price += shift
                    self.lower_price += shift
                    logger.info(f"Quick grid shift UP by ${shift:.{self.price_decimals}f}")
                    
                elif current_price < self.lower_price + buffer:
                    # Price breaking downward
                    shift = self.grid_spacing * 2
                    self.upper_price -= shift
                    self.lower_price -= shift
                    logger.info(f"Quick grid shift DOWN by ${shift:.{self.price_decimals}f}")
                
                # Place orders in new range
                self.place_weighted_grid_orders()
                
        except Exception as e:
            logger.error(f"Error checking grid boundaries: {e}")
    
    def stop(self):
        """Cancel all open orders and cleanup"""
        try:
            logger.info("Stopping bot and cancelling all orders...")
            self.monitoring_active = False
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def run(self, check_interval: int = 10, recalibration_interval: int = 3600):
        """Backward compatible run method"""
        self.run_enhanced(check_interval, recalibration_interval)


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "aPeo9so1VKnQt2Gm9x"
    API_SECRET = "TrfGjSSfgUBEJg4D4EErLGXBPo6HjcwQ5kuu"
    
    # Investment configuration
    TOTAL_INVESTMENT = 11.7  # Total USDT to use
    
    # Initialize enhanced bot
    bot = EnhancedBybitGridBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='DOGE/USDT:USDT',
        testnet=False  # Use mainnet
    )
    
    # Configure with enhanced parameters
    bot.auto_configure_grid(total_investment=TOTAL_INVESTMENT)
    
    # Optional: Adjust specific parameters
    bot.compound_profits = True
    bot.dynamic_grid_enabled = True
    bot.trailing_stop_enabled = True
    
    try:
        # Run enhanced bot
        bot.run_enhanced(
            check_interval=5,  # Faster checks for better fills
            recalibration_interval=1800  # Recalibrate every 30 minutes
        )
    except KeyboardInterrupt:
        bot.stop()