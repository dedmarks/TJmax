import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BybitGridBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'DOGE/USDT:USDT', testnet: bool = True):
        """
        Initialize Bybit Grid Trading Bot with automatic volatility-based grid calculation
        
        Args:
            api_key: Your Bybit API key
            api_secret: Your Bybit API secret
            symbol: Trading pair (default: BTC/USDT perpetual)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.symbol = symbol
        self.leverage = 50
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # For perpetual futures
                'adjustForTimeDifference': True,
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        # Grid parameters (will be set automatically)
        self.grid_levels = 0
        self.grid_spacing = 0
        self.upper_price = 0
        self.lower_price = 0
        self.total_investment = 0
        self.order_amount = 0
        self.price_decimals = 2  # Will be set based on asset price
        
        # Volatility parameters
        self.atr_multiplier = 2.0  # ATR multiplier for grid range
        self.min_grid_levels = 10
        self.max_grid_levels = 50
        
        # Active orders tracking
        self.buy_orders = {}
        self.sell_orders = {}
        
        # Trend detection parameters
        self.trend_mode = 'neutral'  # 'neutral', 'strong_up', 'strong_down'
        self.trend_threshold = 0.02  # 2% momentum threshold for strong trend
        self.trend_buy_ratio = 0.5  # In strong uptrend, 70% of funds for trending, 30% for grid
        self.trailing_stop_percent = 0.02  # 2% trailing stop in trend mode
        self.trend_position = None  # Track trend following position
        self.highest_price_since_trend = 0
        self.trend_entry_price = 0
        
        # Performance tracking
        self.grid_profits = 0
        self.trend_profits = 0
        
        # Check and set position mode
        self.check_position_mode()
        
    def check_position_mode(self):
        """Check and set appropriate position mode for trading"""
        try:
            # For Bybit V5 API, we need to ensure we're in one-way mode
            # Try to switch to one-way mode
            self.exchange.set_position_mode(hedged=False, symbol=self.symbol)
            logger.info("Position mode set to one-way mode")
        except Exception as e:
            # If setting fails, it might already be in the correct mode
            error_message = str(e).lower()
            if "position mode not modified" in error_message or "not modified" in error_message:
                logger.info("Position mode already set correctly")
            else:
                # Log the error but don't fail - the mode might be correct already
                logger.info(f"Position mode check completed (current mode may be correct): {e}")
            pass
        
    def fetch_ohlcv_data(self, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for analysis
        
        Args:
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
            
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            Current ATR value
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
        
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate various volatility metrics
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volatility metrics
        """
        returns = df['close'].pct_change().dropna()
        
        metrics = {
            'atr': self.calculate_atr(df),
            'atr_percent': (self.calculate_atr(df) / df['close'].iloc[-1]) * 100,
            'std_dev': returns.std(),
            'daily_volatility': returns.std() * np.sqrt(24),  # For hourly data
            'current_price': df['close'].iloc[-1],
            'sma_20': df['close'].rolling(20).mean().iloc[-1],
            'bollinger_upper': df['close'].rolling(20).mean().iloc[-1] + (2 * df['close'].rolling(20).std().iloc[-1]),
            'bollinger_lower': df['close'].rolling(20).mean().iloc[-1] - (2 * df['close'].rolling(20).std().iloc[-1])
        }
        
        return metrics
        
    def detect_strong_trend(self) -> Dict:
        """
        Detect if market is in a strong trend using multiple indicators
        
        Returns:
            Dictionary with trend information
        """
        try:
            # Fetch data for trend analysis
            df_15m = self.fetch_ohlcv_data('15m', 100)
            df_1h = self.fetch_ohlcv_data('1h', 50)
            
            # Calculate moving averages
            df_15m['sma_10'] = df_15m['close'].rolling(10).mean()
            df_15m['sma_20'] = df_15m['close'].rolling(20).mean()
            df_15m['sma_50'] = df_15m['close'].rolling(50).mean()
            
            # Calculate momentum
            momentum_5 = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-5]) / df_15m['close'].iloc[-5]
            momentum_10 = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-10]) / df_15m['close'].iloc[-10]
            
            # Calculate RSI
            delta = df_1h['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Calculate MACD
            exp1 = df_15m['close'].ewm(span=12, adjust=False).mean()
            exp2 = df_15m['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            
            # Trend detection criteria
            current_price = df_15m['close'].iloc[-1]
            sma_10 = df_15m['sma_10'].iloc[-1]
            sma_20 = df_15m['sma_20'].iloc[-1]
            sma_50 = df_15m['sma_50'].iloc[-1]
            
            # Strong uptrend criteria
            strong_uptrend = (
                current_price > sma_10 > sma_20 > sma_50 and  # Price above all MAs in order
                momentum_5 > self.trend_threshold and  # Strong 5-period momentum
                momentum_10 > self.trend_threshold * 0.8 and  # Sustained momentum
                current_rsi > 50 and current_rsi < 80 and  # RSI in bullish zone but not overbought
                macd_histogram.iloc[-1] > 0 and  # MACD histogram positive
                macd_histogram.iloc[-1] > macd_histogram.iloc[-5]  # MACD histogram increasing
            )
            
            # Strong downtrend criteria
            strong_downtrend = (
                current_price < sma_10 < sma_20 < sma_50 and
                momentum_5 < -self.trend_threshold and
                momentum_10 < -self.trend_threshold * 0.8 and
                current_rsi < 50 and current_rsi > 20 and
                macd_histogram.iloc[-1] < 0 and
                macd_histogram.iloc[-1] < macd_histogram.iloc[-5]
            )
            
            # Calculate trend strength (0-1)
            if strong_uptrend:
                trend_strength = min(1.0, (momentum_5 + momentum_10) / (2 * self.trend_threshold * 2))
            elif strong_downtrend:
                trend_strength = min(1.0, abs(momentum_5 + momentum_10) / (2 * self.trend_threshold * 2))
            else:
                trend_strength = 0
            
            return {
                'is_strong_uptrend': strong_uptrend,
                'is_strong_downtrend': strong_downtrend,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'rsi': current_rsi,
                'trend_strength': trend_strength,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return {
                'is_strong_uptrend': False,
                'is_strong_downtrend': False,
                'momentum_5': 0,
                'momentum_10': 0,
                'rsi': 50,
                'trend_strength': 0,
                'current_price': self.get_current_price()
            }
            
    def enter_trend_position(self, trend_direction: str):
        """
        Enter a larger position when strong trend is detected
        
        Args:
            trend_direction: 'up' or 'down'
        """
        try:
            current_price = self.get_current_price()
            
            # Calculate trend position size (larger than grid orders)
            trend_investment = self.total_investment * self.trend_buy_ratio
            position_size = self.calculate_position_size_for_amount(current_price, trend_investment)
            
            if trend_direction == 'up':
                # Place market buy order for immediate entry
                order = self.exchange.create_market_buy_order(
                    symbol=self.symbol,
                    amount=position_size
                )
                
                self.trend_position = {
                    'direction': 'long',
                    'entry_price': current_price,
                    'size': position_size,
                    'order_id': order['id']
                }
                
                self.trend_entry_price = current_price
                self.highest_price_since_trend = current_price
                
                logger.info(f"🚀 Entered TREND LONG position: {position_size:.4f} @ ${current_price:.2f}")
                logger.info(f"Trend investment: ${trend_investment:.2f}")
                
        except Exception as e:
            logger.error(f"Error entering trend position: {e}")
            
    def manage_trend_position(self):
        """
        Manage the trend following position with trailing stop
        """
        if not self.trend_position:
            return
            
        try:
            current_price = self.get_current_price()
            
            if self.trend_position['direction'] == 'long':
                # Update highest price
                if current_price > self.highest_price_since_trend:
                    self.highest_price_since_trend = current_price
                    logger.info(f"📈 New high in trend: ${current_price:.2f}")
                
                # Calculate trailing stop price
                trailing_stop_price = self.highest_price_since_trend * (1 - self.trailing_stop_percent)
                
                # Check if should exit
                if current_price <= trailing_stop_price:
                    # Exit position
                    order = self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=self.trend_position['size']
                    )
                    
                    # Calculate profit
                    profit_percent = ((current_price - self.trend_entry_price) / self.trend_entry_price) * 100
                    profit_usdt = (current_price - self.trend_entry_price) * self.trend_position['size'] * self.leverage
                    
                    self.trend_profits += profit_usdt
                    
                    logger.info(f"📉 Exited trend position @ ${current_price:.2f}")
                    logger.info(f"Trend profit: {profit_percent:.2f}% (${profit_usdt:.2f})")
                    
                    # Reset trend position
                    self.trend_position = None
                    self.trend_mode = 'neutral'
                    self.highest_price_since_trend = 0
                    
                    # Resume normal grid trading
                    self.resume_grid_trading()
                else:
                    # Log trailing stop info
                    distance_to_stop = ((current_price - trailing_stop_price) / current_price) * 100
                    logger.info(f"Trailing stop: ${trailing_stop_price:.2f} ({distance_to_stop:.2f}% away)")
                    
        except Exception as e:
            logger.error(f"Error managing trend position: {e}")
            
    def calculate_position_size_for_amount(self, price: float, usdt_amount: float) -> float:
        """
        Calculate position size for a specific USDT amount
        
        Args:
            price: Current price
            usdt_amount: USDT amount to use
            
        Returns:
            Position size
        """
        contract_value_usdt = usdt_amount * self.leverage
        position_size = contract_value_usdt / price
        
        # Get market info for minimum order size
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
        
    def pause_grid_trading(self):
        """
        Pause grid trading when entering trend mode
        """
        try:
            # Cancel all grid orders
            logger.info("Pausing grid trading for trend following...")
            self.exchange.cancel_all_orders(self.symbol)
            
            # Clear order tracking
            self.buy_orders.clear()
            self.sell_orders.clear()
            
        except Exception as e:
            logger.error(f"Error pausing grid trading: {e}")
            
    def resume_grid_trading(self):
        """
        Resume grid trading after trend mode
        """
        try:
            logger.info("Resuming grid trading...")
            
            # Recalculate grid based on new price
            self.auto_configure_grid(self.total_investment)
            
            # Place new grid orders
            self.place_grid_orders()
            
        except Exception as e:
            logger.error(f"Error resuming grid trading: {e}")
            
    def calculate_optimal_grid_parameters(self, total_investment: float) -> Dict:
        """
        Automatically calculate optimal grid parameters based on volatility
        
        Args:
            total_investment: Total USDT to use for the grid
            
        Returns:
            Dictionary with calculated grid parameters
        """
        try:
            # Fetch multiple timeframes for comprehensive analysis
            df_1h = self.fetch_ohlcv_data('1h', 168)  # 1 week of hourly data
            df_4h = self.fetch_ohlcv_data('4h', 168)  # 4 weeks of 4h data
            
            # Calculate volatility metrics
            metrics_1h = self.calculate_volatility_metrics(df_1h)
            metrics_4h = self.calculate_volatility_metrics(df_4h)
            
            current_price = metrics_1h['current_price']
            
            # Validate current price
            if current_price <= 0:
                logger.error(f"Invalid current price: ${current_price}. Please check symbol configuration.")
                raise ValueError(f"Invalid price for {self.symbol}")
            
            logger.info(f"Analyzing {self.symbol} - Current price: ${current_price}")
            
            # Use weighted average of ATR from different timeframes
            weighted_atr = (metrics_1h['atr'] * 0.7 + metrics_4h['atr'] * 0.3)
            
            # Calculate grid range based on ATR and Bollinger Bands
            # Use a combination of ATR-based range and Bollinger Bands
            atr_range = weighted_atr * self.atr_multiplier
            
            # Consider Bollinger Bands for additional context
            bb_range = metrics_1h['bollinger_upper'] - metrics_1h['bollinger_lower']
            
            # Combine both approaches
            grid_range = (atr_range * 0.6 + bb_range * 0.4)
            
            # Ensure minimum grid range based on asset price
            # For assets < $1, use 5% minimum range
            # For assets $1-$100, use 3% minimum range  
            # For assets > $100, use 2% minimum range
            if current_price < 1:
                min_range_percent = 0.05
            elif current_price < 100:
                min_range_percent = 0.03
            else:
                min_range_percent = 0.02
                
            min_range = current_price * min_range_percent
            if grid_range < min_range:
                grid_range = min_range
                logger.warning(f"Grid range too small, adjusted to minimum: ${grid_range:.6f}")
            
            # Set upper and lower bounds
            upper_price = current_price + (grid_range / 2)
            lower_price = current_price - (grid_range / 2)
            
            # Ensure lower price is reasonable (not below 50% of current price)
            if lower_price < current_price * 0.5:
                lower_price = current_price * 0.5
                upper_price = current_price + (current_price - lower_price)
                logger.warning(f"Adjusted grid bounds to prevent extreme lower price")
            
            # Calculate optimal number of grid levels based on volatility
            # Higher volatility = fewer levels (wider spacing)
            # Lower volatility = more levels (tighter spacing)
            volatility_factor = metrics_1h['atr_percent']
            
            if volatility_factor < 1:  # Low volatility
                grid_levels = 40
            elif volatility_factor < 2:  # Medium volatility
                grid_levels = 30
            elif volatility_factor < 3:  # High volatility
                grid_levels = 20
            else:  # Very high volatility
                grid_levels = 15
                
            # Ensure within bounds
            grid_levels = max(self.min_grid_levels, min(self.max_grid_levels, grid_levels))
            
            # Calculate grid spacing
            grid_spacing = (upper_price - lower_price) / (grid_levels - 1)
            
            # Calculate minimum profitable spacing (considering fees)
            # Bybit maker fee: 0.01%, taker fee: 0.06%
            # For profit, spacing should be > 2 * maker fee + safety margin
            min_spacing_percent = 0.15  # 0.15% minimum spacing
            min_spacing = current_price * (min_spacing_percent / 100)
            
            if grid_spacing < min_spacing:
                # Adjust grid levels to ensure profitable spacing
                grid_levels = int((upper_price - lower_price) / min_spacing) + 1
                grid_spacing = (upper_price - lower_price) / (grid_levels - 1)
            
            # Calculate appropriate decimal places for price based on asset value
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
            
            # Adjust order amount based on trend mode allocation
            # If we allocate funds for trend following, reduce grid order size
            grid_investment = total_investment * (1 - self.trend_buy_ratio)
            
            parameters = {
                'upper_price': round(upper_price, price_decimals),
                'lower_price': round(lower_price, price_decimals),
                'grid_levels': grid_levels,
                'grid_spacing': round(grid_spacing, price_decimals),
                'current_price': round(current_price, price_decimals),
                'volatility_1h': round(volatility_factor, 2),
                'atr': round(weighted_atr, price_decimals),
                'recommended_investment': round(total_investment, 2),
                'order_amount': round(grid_investment / grid_levels, 2),
                'price_decimals': price_decimals
            }
            
            logger.info(f"Volatility Analysis Complete:")
            logger.info(f"  - Current Price: ${parameters['current_price']}")
            logger.info(f"  - ATR (weighted): ${parameters['atr']}")
            logger.info(f"  - Volatility (1h): {parameters['volatility_1h']}%")
            logger.info(f"  - Calculated Range: ${parameters['lower_price']} - ${parameters['upper_price']}")
            logger.info(f"  - Grid Levels: {parameters['grid_levels']}")
            logger.info(f"  - Grid Spacing: ${parameters['grid_spacing']} ({round(grid_spacing/current_price*100, 3)}%)")
            logger.info(f"  - Grid Investment: ${grid_investment:.2f} ({(1-self.trend_buy_ratio)*100:.0f}% of total)")
            logger.info(f"  - Trend Reserve: ${total_investment * self.trend_buy_ratio:.2f} ({self.trend_buy_ratio*100:.0f}% of total)")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error calculating grid parameters: {e}")
            raise
            
    def auto_configure_grid(self, total_investment: float):
        """
        Automatically configure grid based on market volatility
        
        Args:
            total_investment: Total USDT to use for the grid
        """
        logger.info("Analyzing market conditions and calculating optimal grid parameters...")
        
        params = self.calculate_optimal_grid_parameters(total_investment)
        
        self.upper_price = params['upper_price']
        self.lower_price = params['lower_price']
        self.grid_levels = params['grid_levels']
        self.grid_spacing = params['grid_spacing']
        self.total_investment = total_investment
        self.order_amount = params['order_amount']
        self.price_decimals = params.get('price_decimals', 2)
        
        logger.info(f"Grid automatically configured with {self.grid_levels} levels")
        
    def analyze_market_regime(self) -> str:
        """
        Analyze current market regime (trending/ranging)
        
        Returns:
            Market regime: 'trending_up', 'trending_down', or 'ranging'
        """
        try:
            df = self.fetch_ohlcv_data('1h', 50)
            
            # Calculate moving averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # Calculate ADX for trend strength
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate directional movement
            up_move = high.diff()
            down_move = -low.diff()
            
            up_move[up_move < 0] = 0
            down_move[down_move < 0] = 0
            
            # Check trend
            recent_close = df['close'].iloc[-1]
            sma_10 = df['sma_10'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            
            # Trend detection
            if sma_10 > sma_20 and recent_close > sma_10:
                regime = 'trending_up'
            elif sma_10 < sma_20 and recent_close < sma_10:
                regime = 'trending_down'
            else:
                regime = 'ranging'
                
            logger.info(f"Market regime detected: {regime}")
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return 'ranging'
            
    def adjust_grid_for_trend(self):
        """
        Adjust grid parameters based on market trend
        """
        regime = self.analyze_market_regime()
        
        if regime == 'trending_up':
            # Shift grid slightly higher in uptrend
            shift = self.grid_spacing * 0.5
            self.upper_price += shift
            self.lower_price += shift
            logger.info(f"Grid adjusted up by ${shift:.2f} for uptrend")
            
        elif regime == 'trending_down':
            # Shift grid slightly lower in downtrend
            shift = self.grid_spacing * 0.5
            self.upper_price -= shift
            self.lower_price -= shift
            logger.info(f"Grid adjusted down by ${shift:.2f} for downtrend")
            
    def set_leverage(self):
        """Set leverage for the trading pair"""
        try:
            # Set leverage to 30x
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            # Check if it's just "leverage not modified" error
            error_message = str(e).lower()
            if "leverage not modified" in error_message or "110043" in error_message:
                logger.info(f"Leverage already set to {self.leverage}x for {self.symbol}")
            else:
                logger.error(f"Error setting leverage: {e}")
                raise
            
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            raise
            
    def calculate_grid_levels(self) -> List[float]:
        """Calculate all grid price levels"""
        levels = []
        for i in range(self.grid_levels):
            price = self.lower_price + (i * self.grid_spacing)
            levels.append(round(price, self.price_decimals))
        return levels
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size for a given price level"""
        # Calculate the contract size based on order amount and leverage
        contract_value_usdt = self.order_amount * self.leverage
        position_size = contract_value_usdt / price
        
        # Get market info for minimum order size
        try:
            market = self.exchange.market(self.symbol)
            min_size = market['limits']['amount']['min']
            
            # Round to appropriate precision
            precision = market['precision']['amount']
            if isinstance(precision, float):
                # Convert precision to decimal places
                decimal_places = max(0, -int(np.log10(precision)))
            else:
                decimal_places = int(precision)
            
            position_size = round(position_size, decimal_places)
            
            # Ensure minimum size
            if position_size < min_size:
                position_size = min_size
                
        except Exception as e:
            logger.warning(f"Could not get market info: {e}")
            # Default to 3 decimal places for BTC
            position_size = round(position_size, 3)
            
        return position_size
        
    def place_grid_orders(self):
        """Place initial grid orders"""
        try:
            current_price = self.get_current_price()
            grid_levels = self.calculate_grid_levels()
            
            logger.info(f"Current price: ${current_price:.2f}")
            logger.info(f"Placing grid orders...")
            
            for price in grid_levels:
                if price < current_price * 0.995:  # Buy orders below current price
                    self.place_buy_order(price)
                elif price > current_price * 1.005:  # Sell orders above current price
                    self.place_sell_order(price)
                    
            logger.info("Grid orders placed successfully")
            
        except Exception as e:
            logger.error(f"Error placing grid orders: {e}")
            raise
            
    def place_buy_order(self, price: float):
        """Place a limit buy order"""
        try:
            amount = self.calculate_position_size(price)
            
            # Check position mode and set appropriate parameters
            params = {
                'timeInForce': 'PostOnly'  # Maker only to avoid fees
            }
            
            # For Bybit, we need to check if it's hedge mode or one-way mode
            try:
                # Try to get position mode from account info
                account_info = self.exchange.fetch_positions([self.symbol])
                if account_info and len(account_info) > 0:
                    # If positions exist, check their mode
                    if 'info' in account_info[0] and 'positionIdx' in account_info[0]['info']:
                        params['positionIdx'] = 0  # One-way mode
                else:
                    # Default to not specifying positionIdx for one-way mode
                    pass
            except:
                # If we can't determine, don't specify positionIdx
                pass
            
            order = self.exchange.create_limit_buy_order(
                symbol=self.symbol,
                amount=amount,
                price=price,
                params=params
            )
            
            self.buy_orders[order['id']] = {
                'price': price,
                'amount': amount,
                'status': 'open'
            }
            
            logger.info(f"Buy order placed: {amount:.4f} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error placing buy order at ${price}: {e}")
            
    def place_sell_order(self, price: float):
        """Place a limit sell order"""
        try:
            amount = self.calculate_position_size(price)
            
            # Check position mode and set appropriate parameters
            params = {
                'timeInForce': 'PostOnly'  # Maker only to avoid fees
            }
            
            # For Bybit, we need to check if it's hedge mode or one-way mode
            try:
                # Try to get position mode from account info
                account_info = self.exchange.fetch_positions([self.symbol])
                if account_info and len(account_info) > 0:
                    # If positions exist, check their mode
                    if 'info' in account_info[0] and 'positionIdx' in account_info[0]['info']:
                        params['positionIdx'] = 0  # One-way mode
                else:
                    # Default to not specifying positionIdx for one-way mode
                    pass
            except:
                # If we can't determine, don't specify positionIdx
                pass
            
            order = self.exchange.create_limit_sell_order(
                symbol=self.symbol,
                amount=amount,
                price=price,
                params=params
            )
            
            self.sell_orders[order['id']] = {
                'price': price,
                'amount': amount,
                'status': 'open'
            }
            
            logger.info(f"Sell order placed: {amount:.4f} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"Error placing sell order at ${price}: {e}")
            
    def check_filled_orders(self):
        """Check for filled orders and place new ones"""
        try:
            # Skip if in trend mode
            if self.trend_mode == 'strong_up':
                return
                
            # Fetch recent closed orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=50)
            
            for order in closed_orders:
                order_id = order['id']
                
                # Check if it's a buy order that was filled
                if order_id in self.buy_orders and self.buy_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        self.buy_orders[order_id]['status'] = 'filled'
                        
                        # Place a sell order one grid level up
                        new_sell_price = filled_price + self.grid_spacing
                        if new_sell_price <= self.upper_price:
                            self.place_sell_order(new_sell_price)
                            
                        # Track grid profit
                        self.grid_profits += self.grid_spacing * order['filled'] * 0.9995  # Account for fees
                            
                        logger.info(f"Buy order filled at ${filled_price:.2f}, placed sell at ${new_sell_price:.2f}")
                        
                # Check if it's a sell order that was filled
                elif order_id in self.sell_orders and self.sell_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        self.sell_orders[order_id]['status'] = 'filled'
                        
                        # Place a buy order one grid level down
                        new_buy_price = filled_price - self.grid_spacing
                        if new_buy_price >= self.lower_price:
                            self.place_buy_order(new_buy_price)
                            
                        logger.info(f"Sell order filled at ${filled_price:.2f}, placed buy at ${new_buy_price:.2f}")
                        
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
            
    def check_grid_boundaries(self):
        """Check if price has moved outside grid boundaries and adjust if needed"""
        try:
            current_price = self.get_current_price()
            
            # Calculate buffer zone (10% of grid range)
            buffer = (self.upper_price - self.lower_price) * 0.1
            
            if current_price > self.upper_price - buffer or current_price < self.lower_price + buffer:
                logger.warning(f"Price ${current_price:.2f} approaching grid boundaries")
                logger.info("Recalculating grid parameters...")
                
                # Cancel all orders
                self.stop()
                
                # Reconfigure grid
                self.auto_configure_grid(self.total_investment)
                self.adjust_grid_for_trend()
                
                # Place new orders
                self.place_grid_orders()
                
        except Exception as e:
            logger.error(f"Error checking grid boundaries: {e}")
            
    def calculate_grid_profit(self) -> Dict:
        """Calculate realized and unrealized profit from grid trading"""
        try:
            # Count filled orders
            filled_buys = sum(1 for order in self.buy_orders.values() if order['status'] == 'filled')
            filled_sells = sum(1 for order in self.sell_orders.values() if order['status'] == 'filled')
            
            # Estimate profit per grid completion
            # Profit = (grid_spacing / average_price) - (2 * maker_fee)
            avg_price = (self.upper_price + self.lower_price) / 2
            profit_per_grid = (self.grid_spacing / avg_price - 0.0002) * 100  # As percentage
            
            # Calculate number of completed grids (a buy followed by a sell)
            completed_grids = min(filled_buys, filled_sells)
            
            estimated_profit = completed_grids * profit_per_grid * self.order_amount
            
            return {
                'completed_grids': completed_grids,
                'profit_per_grid_percent': round(profit_per_grid, 3),
                'estimated_profit_usdt': round(estimated_profit, 2),
                'filled_buy_orders': filled_buys,
                'filled_sell_orders': filled_sells,
                'grid_profits': round(self.grid_profits, 2),
                'trend_profits': round(self.trend_profits, 2),
                'total_profits': round(self.grid_profits + self.trend_profits, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit: {e}")
            return {}
            
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
            
    def check_and_manage_trend(self):
        """
        Check for strong trends and manage positions accordingly
        """
        trend_info = self.detect_strong_trend()
        
        # If already in trend mode, manage the position
        if self.trend_mode == 'strong_up' and self.trend_position:
            self.manage_trend_position()
            return
            
        # Check if we should enter trend mode
        if trend_info['is_strong_uptrend'] and self.trend_mode == 'neutral':
            logger.info("🚀 STRONG UPTREND DETECTED! Switching to trend following mode...")
            logger.info(f"  - 5-period momentum: {trend_info['momentum_5']*100:.2f}%")
            logger.info(f"  - 10-period momentum: {trend_info['momentum_10']*100:.2f}%")
            logger.info(f"  - RSI: {trend_info['rsi']:.2f}")
            logger.info(f"  - Trend strength: {trend_info['trend_strength']*100:.0f}%")
            
            # Pause grid trading
            self.pause_grid_trading()
            
            # Enter trend mode
            self.trend_mode = 'strong_up'
            self.enter_trend_position('up')
            
    def run(self, check_interval: int = 10, recalibration_interval: int = 3600):
        """
        Run the grid trading bot with automatic volatility-based adjustments and trend detection
        
        Args:
            check_interval: Seconds between checking for filled orders
            recalibration_interval: Seconds between grid recalibration (default: 1 hour)
        """
        try:
            # Set leverage
            self.set_leverage()
            
            # Analyze market and adjust grid for trend
            self.adjust_grid_for_trend()
            
            # Place initial grid orders
            self.place_grid_orders()
            
            logger.info(f"Grid bot started with trend detection. Checking orders every {check_interval} seconds...")
            logger.info(f"Grid will be recalibrated every {recalibration_interval/3600:.1f} hours")
            logger.info(f"Trend detection active - {self.trend_buy_ratio*100:.0f}% allocated for trend following")
            
            last_recalibration = time.time()
            last_trend_check = time.time()
            trend_check_interval = 60  # Check for trends every minute
            
            while True:
                try:
                    # Check for strong trends
                    if time.time() - last_trend_check > trend_check_interval:
                        self.check_and_manage_trend()
                        last_trend_check = time.time()
                    
                    # Check for filled orders (only if not in trend mode)
                    if self.trend_mode == 'neutral':
                        self.check_filled_orders()
                        
                        # Check if price is near boundaries
                        self.check_grid_boundaries()
                    
                    # Get account info and profit
                    info = self.get_account_info()
                    profit_info = self.calculate_grid_profit()
                    
                    if info:
                        logger.info(f"USDT Balance: ${info['usdt_balance']:.2f}")
                        if info['positions']:
                            for pos in info['positions']:
                                if pos['contracts'] != 0:
                                    logger.info(f"Position: {pos['contracts']} contracts, PnL: ${pos['unrealizedPnl']:.2f}")
                    
                    if profit_info:
                        logger.info(f"📊 Performance Summary:")
                        logger.info(f"  - Grid profits: ${profit_info['grid_profits']}")
                        logger.info(f"  - Trend profits: ${profit_info['trend_profits']}")
                        logger.info(f"  - Total profits: ${profit_info['total_profits']}")
                        logger.info(f"  - Completed grids: {profit_info['completed_grids']}")
                    
                    # Display current mode
                    if self.trend_mode == 'strong_up':
                        logger.info(f"⚡ MODE: TREND FOLLOWING (Long)")
                    else:
                        logger.info(f"📊 MODE: GRID TRADING")
                    
                    # Check if it's time to recalibrate (only if not in trend mode)
                    if self.trend_mode == 'neutral' and time.time() - last_recalibration > recalibration_interval:
                        logger.info("Performing periodic grid recalibration...")
                        self.stop()
                        self.auto_configure_grid(self.total_investment)
                        self.adjust_grid_for_trend()
                        self.place_grid_orders()
                        last_recalibration = time.time()
                    
                    # Wait before next check
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval)
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
            
    def stop(self):
        """Cancel all open orders"""
        try:
            logger.info("Cancelling all open orders...")
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "aPeo9so1VKnQt2Gm9x"
    API_SECRET = "TrfGjSSfgUBEJg4D4EErLGXBPo6HjcwQ5kuu"
    
    # Only need to specify investment amount - grid parameters are calculated automatically
    TOTAL_INVESTMENT = 2  # Total USDT to use
    
    # Initialize bot
    bot = BybitGridBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='1000BONK/USDT:USDT',  # DOGE perpetual futures
        testnet=False  # Use mainnet
    )
    
    # Automatically configure grid based on volatility
    bot.auto_configure_grid(total_investment=TOTAL_INVESTMENT)
    
    try:
        # Run the bot with automatic adjustments and trend detection
        bot.run(
            check_interval=10,  # Check orders every 10 seconds
            recalibration_interval=3600  # Recalibrate grid every hour
        )
    except KeyboardInterrupt:
        # Stop the bot
        bot.stop()