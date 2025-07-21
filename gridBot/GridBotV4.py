import ccxt
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedGridBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = '1000PEPE/USDT:USDT', testnet: bool = True):
        """
        Simplified but powerful grid trading bot with strategic level placement
        """
        self.symbol = symbol
        self.leverage = 20
        
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
            
        # Core grid parameters
        self.max_grid_levels = 30  # Maximum levels to place
        self.grid_range_percent = 4.0  # 4% range (2% up, 2% down)
        self.total_investment = 0
        self.order_amount = 0
        
        # Strategic level parameters
        self.use_strategic_levels = True  # Use S/R levels instead of equal spacing
        self.level_strength_threshold = 2  # Minimum touches to consider a level
        self.level_merge_threshold = 0.0015  # Merge levels within 0.15%
        self.volume_profile_enabled = True  # Use volume profile for levels
        
        # Simple but effective features
        self.enable_trailing_grid = True  # Grid follows price
        self.enable_dynamic_spacing = True  # Adjust spacing based on volatility
        self.enable_profit_taking = True  # Take profits on big moves
        
        # Risk management (simplified)
        self.max_drawdown_percent = 25  # Stop if down 25%
        self.take_profit_percent = 50   # Take profits if up 50%
        self.position_limit_multiplier = 3  # Max 3x account size in positions
        
        # Performance tracking
        self.initial_balance = 0
        self.highest_balance = 0
        self.total_profit = 0
        self.trades_won = 0
        self.trades_total = 0
        
        # Order tracking
        self.active_orders = {}  # order_id: {price, amount, side, pair_price}
        self.strategic_levels = []  # List of strategic price levels
        self.last_grid_update = 0
        self.grid_center = 0
        
        # Market data
        self.current_volatility = 0.02  # Default 2%
        self.price_history = deque(maxlen=100)
        
        # Market info (will be set in _setup)
        self.min_order_size = 0.001
        self.price_precision = 0.01
        self.price_decimals = 2  # Default, will be updated
        
        # Initialize
        self._setup()
        
    def _setup(self):
        """Initial setup"""
        try:
            # Set leverage
            try:
                self.exchange.set_leverage(self.leverage, self.symbol)
                logger.info(f"Leverage set to {self.leverage}x")
            except Exception as e:
                if "leverage not modified" in str(e).lower():
                    logger.info(f"Leverage already set to {self.leverage}x")
                else:
                    logger.warning(f"Could not set leverage: {e}")
            
            # Set position mode
            try:
                self.exchange.set_position_mode(hedged=False, symbol=self.symbol)
                logger.info("Position mode set to one-way")
            except Exception as e:
                if "position mode not modified" in str(e).lower():
                    logger.info("Position mode already set correctly")
                else:
                    logger.info(f"Position mode: {e}")
                
            # Get market info
            try:
                market = self.exchange.market(self.symbol)
                self.min_order_size = market['limits']['amount']['min']
                self.price_precision = market['precision']['price']
            except Exception as e:
                logger.warning(f"Could not get market info, using defaults: {e}")
                self.min_order_size = 0.001
                self.price_precision = 0.01
            
            # Determine price decimals based on current price
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker['last']
                
                if price < 0.001:
                    self.price_decimals = 7
                elif price < 0.01:
                    self.price_decimals = 6
                elif price < 0.1:
                    self.price_decimals = 5
                elif price < 1:
                    self.price_decimals = 4
                elif price < 10:
                    self.price_decimals = 3
                elif price < 100:
                    self.price_decimals = 2
                else:
                    self.price_decimals = 1
                    
                logger.info(f"Price decimals set to {self.price_decimals} for price ${price}")
                
            except Exception as e:
                logger.warning(f"Could not determine price decimals, using default: {e}")
                self.price_decimals = 4  # Safe default
                
        except Exception as e:
            logger.error(f"Setup error: {e}")
            # Set safe defaults
            self.min_order_size = 0.001
            self.price_precision = 0.01
            self.price_decimals = 4
            
    def calculate_grid_parameters(self):
        """Calculate optimal grid parameters with strategic levels"""
        try:
            current_price = self.get_current_price()
            
            # Calculate volatility if dynamic spacing enabled
            if self.enable_dynamic_spacing:
                volatility = self.calculate_volatility()
                # Adjust range based on volatility (1-6% range)
                self.grid_range_percent = max(1.0, min(6.0, volatility * 200))
                
            # Get strategic levels if enabled
            if self.use_strategic_levels:
                self.strategic_levels = self.calculate_strategic_levels(current_price)
                actual_levels = len(self.strategic_levels)
                
                # Calculate order amount based on actual levels
                self.order_amount = self.total_investment / actual_levels
                
                # Update grid boundaries based on actual levels
                if self.strategic_levels:
                    self.lower_price = min(self.strategic_levels)
                    self.upper_price = max(self.strategic_levels)
                    self.grid_center = current_price
                    
                logger.info(f"Strategic Grid configured:")
                logger.info(f"  Range: ${self.lower_price:.{self.price_decimals}f} - ${self.upper_price:.{self.price_decimals}f}")
                logger.info(f"  Strategic Levels: {actual_levels}")
                logger.info(f"  Key levels: {[f'${l:.{self.price_decimals}f}' for l in self.strategic_levels[:10]]}")
                logger.info(f"  Order size: ${self.order_amount:.2f}")
                
            else:
                # Fallback to equal spacing
                range_size = current_price * (self.grid_range_percent / 100)
                self.upper_price = current_price + range_size / 2
                self.lower_price = current_price - range_size / 2
                self.grid_center = current_price
                
                # Calculate spacing
                self.grid_spacing = (self.upper_price - self.lower_price) / (self.max_grid_levels - 1)
                
                # Calculate order amount
                self.order_amount = self.total_investment / self.max_grid_levels
                
                logger.info(f"Equal-spacing Grid configured:")
                logger.info(f"  Range: ${self.lower_price:.{self.price_decimals}f} - ${self.upper_price:.{self.price_decimals}f}")
                logger.info(f"  Levels: {self.max_grid_levels}")
                logger.info(f"  Spacing: ${self.grid_spacing:.{self.price_decimals}f}")
                logger.info(f"  Order size: ${self.order_amount:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating grid parameters: {e}")
            
    def calculate_volatility(self) -> float:
        """Simple volatility calculation"""
        try:
            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '5m', limit=20)
            closes = [x[4] for x in ohlcv]
            
            # Calculate returns
            returns = pd.Series(closes).pct_change().dropna()
            
            # Simple volatility (standard deviation of returns)
            volatility = returns.std()
            self.current_volatility = volatility
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02  # Default 2%
            
    def calculate_strategic_levels(self, current_price: float) -> List[float]:
        """Calculate strategic support and resistance levels"""
        try:
            levels = []
            
            # 1. Get historical data for analysis
            ohlcv_1h = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=200)
            ohlcv_15m = self.exchange.fetch_ohlcv(self.symbol, '15m', limit=200)
            
            # Convert to pandas for easier analysis
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 2. Find swing highs and lows (basic S/R)
            swing_levels = self._find_swing_levels(df_1h)
            levels.extend(swing_levels)
            
            # 3. Find high volume levels (where lots of trading occurred)
            if self.volume_profile_enabled:
                volume_levels = self._find_volume_levels(df_15m)
                levels.extend(volume_levels)
                
            # 4. Add psychological levels (round numbers)
            psychological_levels = self._find_psychological_levels(current_price)
            levels.extend(psychological_levels)
            
            # 5. Add pivot points
            pivot_levels = self._calculate_pivot_levels(df_1h)
            levels.extend(pivot_levels)
            
            # 6. Merge nearby levels
            levels = self._merge_nearby_levels(levels)
            
            # 7. Filter levels within our range
            range_size = current_price * (self.grid_range_percent / 100)
            upper_bound = current_price + range_size / 2
            lower_bound = current_price - range_size / 2
            
            levels = [l for l in levels if lower_bound <= l <= upper_bound]
            
            # 8. Ensure we have enough levels
            if len(levels) < self.max_grid_levels:
                # Add some equally spaced levels to fill gaps
                additional_levels = self._fill_level_gaps(levels, lower_bound, upper_bound)
                levels.extend(additional_levels)
                
            # Sort and limit to max grid levels
            levels = sorted(levels)[:self.max_grid_levels]
            
            logger.info(f"Found {len(levels)} strategic levels")
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating strategic levels: {e}")
            # Fallback to equal spacing
            return self._calculate_equal_spacing_levels(current_price)
            
    def _find_swing_levels(self, df: pd.DataFrame) -> List[float]:
        """Find swing highs and lows that act as S/R"""
        levels = []
        
        # Look for local minima and maxima
        for i in range(2, len(df) - 2):
            # Swing high
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                levels.append(df['high'].iloc[i])
                
            # Swing low
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                levels.append(df['low'].iloc[i])
                
        return levels
        
    def _find_volume_levels(self, df: pd.DataFrame) -> List[float]:
        """Find price levels with high volume (volume profile)"""
        levels = []
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, 50)
        
        # Calculate volume at each price level
        volume_profile = {}
        
        for i in range(len(df)):
            # Approximate volume distribution across the candle
            candle_low = df['low'].iloc[i]
            candle_high = df['high'].iloc[i]
            candle_volume = df['volume'].iloc[i]
            
            # Find which bins this candle covers
            for j in range(len(bins) - 1):
                if bins[j] <= candle_high and bins[j+1] >= candle_low:
                    # Add volume to this bin
                    bin_price = (bins[j] + bins[j+1]) / 2
                    if bin_price not in volume_profile:
                        volume_profile[bin_price] = 0
                    volume_profile[bin_price] += candle_volume
                    
        # Find high volume levels
        if volume_profile:
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            # Take top 20% as significant levels
            significant_count = max(1, len(sorted_levels) // 5)
            for price, volume in sorted_levels[:significant_count]:
                levels.append(price)
                
        return levels
        
    def _find_psychological_levels(self, current_price: float) -> List[float]:
        """Find psychological levels (round numbers)"""
        levels = []
        
        # Determine the round number interval based on price
        if current_price < 1:
            interval = 0.01
        elif current_price < 10:
            interval = 0.1
        elif current_price < 100:
            interval = 1
        elif current_price < 1000:
            interval = 10
        elif current_price < 10000:
            interval = 100
        else:
            interval = 1000
            
        # Find round numbers within range
        range_size = current_price * (self.grid_range_percent / 100)
        lower = current_price - range_size
        upper = current_price + range_size
        
        start = int(lower / interval) * interval
        level = start
        
        while level <= upper:
            if level >= lower:
                levels.append(level)
            level += interval
            
        return levels
        
    def _calculate_pivot_levels(self, df: pd.DataFrame) -> List[float]:
        """Calculate pivot point levels"""
        levels = []
        
        # Use yesterday's data for pivot calculation
        if len(df) > 0:
            high = df['high'].iloc[-24:].max()  # Last 24 hours
            low = df['low'].iloc[-24:].min()
            close = df['close'].iloc[-1]
            
            # Classic pivot points
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            levels.extend([s2, s1, pivot, r1, r2])
            
        return levels
        
    def _merge_nearby_levels(self, levels: List[float]) -> List[float]:
        """Merge levels that are too close together"""
        if not levels:
            return []
            
        levels = sorted(set(levels))  # Remove duplicates and sort
        merged = [levels[0]]
        
        for level in levels[1:]:
            # Check if this level is too close to the last merged level
            if abs(level - merged[-1]) / merged[-1] > self.level_merge_threshold:
                merged.append(level)
            else:
                # Merge by averaging
                merged[-1] = (merged[-1] + level) / 2
                
        return merged
        
    def _fill_level_gaps(self, levels: List[float], lower_bound: float, upper_bound: float) -> List[float]:
        """Fill large gaps between levels"""
        additional = []
        
        # Add boundary levels if missing
        if not levels or levels[0] > lower_bound * 1.001:
            additional.append(lower_bound)
        if not levels or levels[-1] < upper_bound * 0.999:
            additional.append(upper_bound)
            
        # Fill large gaps
        levels_with_bounds = sorted(levels + [lower_bound, upper_bound])
        
        for i in range(len(levels_with_bounds) - 1):
            gap = levels_with_bounds[i+1] - levels_with_bounds[i]
            gap_percent = gap / levels_with_bounds[i]
            
            # If gap is larger than 0.5%, add intermediate levels
            if gap_percent > 0.005:
                num_fills = int(gap_percent / 0.003)  # Add level every 0.3%
                for j in range(1, num_fills):
                    fill_level = levels_with_bounds[i] + (gap * j / num_fills)
                    additional.append(fill_level)
                    
        return additional
        
    def _calculate_equal_spacing_levels(self, current_price: float) -> List[float]:
        """Fallback to equal spacing if strategic levels fail"""
        levels = []
        range_size = current_price * (self.grid_range_percent / 100)
        upper = current_price + range_size / 2
        lower = current_price - range_size / 2
        
        spacing = (upper - lower) / (self.max_grid_levels - 1)
        
        for i in range(self.max_grid_levels):
            level = lower + i * spacing
            levels.append(level)
            
        return levels
            
    def place_grid_orders(self):
        """Place grid orders at strategic levels"""
        try:
            current_price = self.get_current_price()
            placed_orders = {'buy': 0, 'sell': 0}
            
            # Use strategic levels or equal spacing
            if self.use_strategic_levels and self.strategic_levels:
                levels = self.strategic_levels
            else:
                # Equal spacing fallback
                levels = []
                for i in range(self.max_grid_levels):
                    level = self.lower_price + i * self.grid_spacing
                    levels.append(level)
                    
            # Place orders at each level
            for level in levels:
                # Skip orders too close to current price
                if abs(level - current_price) / current_price < 0.001:
                    continue
                    
                # Calculate position size
                position_size = (self.order_amount * self.leverage) / level
                position_size = self.round_size(position_size)
                
                # Place order
                if level < current_price:
                    order = self.place_order('buy', level, position_size)
                    if order:
                        placed_orders['buy'] += 1
                else:
                    order = self.place_order('sell', level, position_size)
                    if order:
                        placed_orders['sell'] += 1
                        
            logger.info(f"Grid orders placed: {placed_orders['buy']} buys, {placed_orders['sell']} sells")
            self.last_grid_update = time.time()
            
        except Exception as e:
            logger.error(f"Error placing grid orders: {e}")
            
    def place_order(self, side: str, price: float, amount: float) -> Optional[Dict]:
        """Place a single order"""
        try:
            price = round(price, self.price_decimals)
            
            params = {
                'timeInForce': 'PostOnly',  # Try to be maker for lower fees
                'reduceOnly': False
            }
            
            if side == 'buy':
                order = self.exchange.create_limit_buy_order(
                    self.symbol, amount, price, params
                )
            else:
                order = self.exchange.create_limit_sell_order(
                    self.symbol, amount, price, params
                )
                
            # Track order
            self.active_orders[order['id']] = {
                'price': price,
                'amount': amount,
                'side': side,
                'pair_price': None,
                'placed_at': time.time()
            }
            
            return order
            
        except Exception as e:
            # If PostOnly fails, try without it
            if "post only" in str(e).lower():
                try:
                    params['timeInForce'] = 'GTC'
                    if side == 'buy':
                        order = self.exchange.create_limit_buy_order(
                            self.symbol, amount, price, params
                        )
                    else:
                        order = self.exchange.create_limit_sell_order(
                            self.symbol, amount, price, params
                        )
                    
                    self.active_orders[order['id']] = {
                        'price': price,
                        'amount': amount,
                        'side': side,
                        'pair_price': None,
                        'placed_at': time.time()
                    }
                    
                    return order
                except:
                    pass
                    
            logger.error(f"Error placing order: {e}")
            return None
            
    def check_filled_orders(self):
        """Check for filled orders and place new ones at next strategic level"""
        try:
            # Get recent closed orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=50)
            
            for order in closed_orders:
                order_id = order['id']
                
                if order_id in self.active_orders and order['status'] == 'closed':
                    order_info = self.active_orders[order_id]
                    filled_price = order['price']
                    filled_amount = order['filled']
                    
                    # Remove from active orders
                    del self.active_orders[order_id]
                    
                    # Find next strategic level for counter order
                    if self.use_strategic_levels and self.strategic_levels:
                        if order_info['side'] == 'buy':
                            # Find next resistance level above
                            next_levels = [l for l in self.strategic_levels if l > filled_price * 1.001]
                            if next_levels:
                                new_price = min(next_levels)  # Use closest level above
                            else:
                                new_price = filled_price * 1.003  # Default 0.3% above
                        else:
                            # Find next support level below
                            next_levels = [l for l in self.strategic_levels if l < filled_price * 0.999]
                            if next_levels:
                                new_price = max(next_levels)  # Use closest level below
                            else:
                                new_price = filled_price * 0.997  # Default 0.3% below
                    else:
                        # Equal spacing mode
                        if order_info['side'] == 'buy':
                            new_price = filled_price + self.grid_spacing
                        else:
                            new_price = filled_price - self.grid_spacing
                            
                    # Place counter order
                    if order_info['side'] == 'buy':
                        new_order = self.place_order('sell', new_price, filled_amount)
                        if new_order:
                            self.active_orders[new_order['id']]['pair_price'] = filled_price
                            logger.info(f"Buy filled at ${filled_price:.{self.price_decimals}f}, sell placed at ${new_price:.{self.price_decimals}f}")
                            
                    else:  # sell order filled
                        new_order = self.place_order('buy', new_price, filled_amount)
                        if new_order:
                            self.active_orders[new_order['id']]['pair_price'] = filled_price
                            logger.info(f"Sell filled at ${filled_price:.{self.price_decimals}f}, buy placed at ${new_price:.{self.price_decimals}f}")
                            
                        # Calculate profit if this was a paired trade
                        if order_info.get('pair_price'):
                            profit = (filled_price - order_info['pair_price']) * filled_amount * self.leverage
                            fee_cost = (filled_price + order_info['pair_price']) * filled_amount * 0.0001 * 2
                            net_profit = profit - fee_cost
                            
                            self.total_profit += net_profit
                            self.trades_total += 1
                            if net_profit > 0:
                                self.trades_won += 1
                                
                            logger.info(f"Trade completed: ${net_profit:.2f} profit (bought ${order_info['pair_price']:.{self.price_decimals}f}, sold ${filled_price:.{self.price_decimals}f})")
                            
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
            
    def check_grid_shift(self):
        """Check if grid needs to be shifted (trailing grid)"""
        if not self.enable_trailing_grid:
            return
            
        try:
            current_price = self.get_current_price()
            distance_from_center = abs(current_price - self.grid_center) / self.grid_center
            
            # Shift grid if price moved more than 1.5% from center
            if distance_from_center > 0.015:
                logger.info(f"Price moved {distance_from_center*100:.1f}% from center, shifting grid...")
                
                # Cancel all orders
                self.cancel_all_orders()
                
                # Recalculate grid with new strategic levels
                self.calculate_grid_parameters()
                
                # Place new orders
                self.place_grid_orders()
                
        except Exception as e:
            logger.error(f"Error checking grid shift: {e}")
            logger.error(f"Error checking filled orders: {e}")
            
    def check_grid_shift(self):
        """Check if grid needs to be shifted (trailing grid)"""
        if not self.enable_trailing_grid:
            return
            
        try:
            current_price = self.get_current_price()
            distance_from_center = abs(current_price - self.grid_center) / self.grid_center
            
            # Shift grid if price moved more than 1.5% from center
            if distance_from_center > 0.015:
                logger.info(f"Price moved {distance_from_center*100:.1f}% from center, shifting grid...")
                
                # Cancel all orders
                self.cancel_all_orders()
                
                # Recalculate grid
                self.calculate_grid_parameters()
                
                # Place new orders
                self.place_grid_orders()
                
        except Exception as e:
            logger.error(f"Error checking grid shift: {e}")
            
    def manage_risk(self) -> bool:
        """Simple risk management"""
        try:
            # Get account info
            balance = self.exchange.fetch_balance()
            current_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            if self.initial_balance == 0:
                self.initial_balance = current_balance
                self.highest_balance = current_balance
                
            # Update highest balance
            if current_balance > self.highest_balance:
                self.highest_balance = current_balance
                
            # Check drawdown
            drawdown = (self.highest_balance - current_balance) / self.highest_balance
            if drawdown > self.max_drawdown_percent / 100:
                logger.error(f"Max drawdown reached: {drawdown*100:.1f}%")
                self.stop_bot()
                return False
                
            # Check profit target
            total_return = (current_balance - self.initial_balance) / self.initial_balance
            if total_return > self.take_profit_percent / 100:
                logger.info(f"Profit target reached: {total_return*100:.1f}%")
                self.stop_bot()
                return False
                
            # Check position size
            positions = self.exchange.fetch_positions([self.symbol])
            total_position_value = 0
            for pos in positions:
                if pos['contracts'] != 0:
                    total_position_value += abs(pos['contracts'] * pos['markPrice'])
                    
            if total_position_value > current_balance * self.position_limit_multiplier:
                logger.warning(f"Position limit reached: ${total_position_value:.2f}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return True
            
    def take_profit_on_trends(self):
        """Take profits during strong price movements"""
        if not self.enable_profit_taking:
            return
            
        try:
            current_price = self.get_current_price()
            
            # Store price history
            self.price_history.append(current_price)
            
            if len(self.price_history) < 20:
                return
                
            # Calculate recent price movement
            prices = list(self.price_history)
            price_20_ago = prices[-20]
            price_change = (current_price - price_20_ago) / price_20_ago
            
            # If price moved more than 2% in last 20 checks
            if abs(price_change) > 0.02:
                positions = self.exchange.fetch_positions([self.symbol])
                
                for pos in positions:
                    if pos['contracts'] != 0:
                        # Take partial profits on favorable positions
                        if (pos['side'] == 'long' and price_change > 0.02) or \
                           (pos['side'] == 'short' and price_change < -0.02):
                            
                            # Close 30% of position
                            close_amount = abs(pos['contracts']) * 0.3
                            
                            if pos['side'] == 'long':
                                self.exchange.create_market_sell_order(
                                    self.symbol, close_amount,
                                    params={'reduceOnly': True}
                                )
                            else:
                                self.exchange.create_market_buy_order(
                                    self.symbol, close_amount,
                                    params={'reduceOnly': True}
                                )
                                
                            logger.info(f"Took partial profits on {pos['side']} position")
                            
        except Exception as e:
            logger.error(f"Error taking profits: {e}")
            
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        try:
            balance = self.exchange.fetch_balance()
            current_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            if self.initial_balance > 0:
                total_return = (current_balance - self.initial_balance) / self.initial_balance * 100
                drawdown = (self.highest_balance - current_balance) / self.highest_balance * 100
            else:
                total_return = 0
                drawdown = 0
                
            win_rate = (self.trades_won / self.trades_total * 100) if self.trades_total > 0 else 0
            
            return {
                'current_balance': current_balance,
                'total_profit': self.total_profit,
                'total_return': total_return,
                'drawdown': drawdown,
                'total_trades': self.trades_total,
                'win_rate': win_rate,
                'active_orders': len(self.active_orders)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
            
    def cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            self.exchange.cancel_all_orders(self.symbol)
            self.active_orders.clear()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            
    def stop_bot(self):
        """Stop the bot safely"""
        try:
            logger.info("Stopping bot...")
            
            # Cancel all orders
            self.cancel_all_orders()
            
            # Close all positions
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['contracts'] != 0:
                    if pos['side'] == 'long':
                        self.exchange.create_market_sell_order(
                            self.symbol, abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                    else:
                        self.exchange.create_market_buy_order(
                            self.symbol, abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                        
            logger.info("Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            # Return a default or raise the error
            raise e
        
    def round_size(self, size: float) -> float:
        """Round size to exchange requirements"""
        try:
            market = self.exchange.market(self.symbol)
            precision = market['precision']['amount']
            if isinstance(precision, float):
                decimal_places = max(0, -int(np.log10(precision)))
            else:
                decimal_places = int(precision)
            
            size = round(size, decimal_places)
            return max(size, self.min_order_size)
        except:
            return round(size, 3)
            
    def run(self, total_investment: float, check_interval: int = 10):
        """Main bot loop"""
        try:
            self.total_investment = total_investment
            
            # Get initial balance
            balance = self.exchange.fetch_balance()
            self.initial_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            logger.info(f"Starting Simplified Grid Bot")
            logger.info(f"Initial balance: ${self.initial_balance:.2f}")
            logger.info(f"Investment amount: ${total_investment:.2f}")
            
            # Calculate and place initial grid
            self.calculate_grid_parameters()
            self.place_grid_orders()
            
            # Main loop
            last_stats_time = time.time()
            stats_interval = 300  # 5 minutes
            
            while True:
                try:
                    # Risk management check
                    if not self.manage_risk():
                        break
                        
                    # Check filled orders
                    self.check_filled_orders()
                    
                    # Check if grid needs shifting
                    self.check_grid_shift()
                    
                    # Take profits on trends
                    self.take_profit_on_trends()
                    
                    # Log stats periodically
                    if time.time() - last_stats_time > stats_interval:
                        stats = self.get_performance_stats()
                        logger.info("="*50)
                        logger.info("Performance Update:")
                        logger.info(f"  Balance: ${stats['current_balance']:.2f}")
                        logger.info(f"  P&L: ${stats['total_profit']:.2f}")
                        logger.info(f"  Return: {stats['total_return']:.2f}%")
                        logger.info(f"  Drawdown: {stats['drawdown']:.1f}%")
                        logger.info(f"  Win Rate: {stats['win_rate']:.1f}%")
                        logger.info(f"  Active Orders: {stats['active_orders']}")
                        logger.info("="*50)
                        last_stats_time = time.time()
                        
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval * 2)
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            # Final report
            stats = self.get_performance_stats()
            logger.info("\n" + "="*50)
            logger.info("FINAL REPORT")
            logger.info("="*50)
            logger.info(f"Final Balance: ${stats['current_balance']:.2f}")
            logger.info(f"Total Profit: ${stats['total_profit']:.2f}")
            logger.info(f"Total Return: {stats['total_return']:.2f}%")
            logger.info(f"Total Trades: {stats['total_trades']}")
            logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
            logger.info("="*50)


# Example usage
if __name__ == "__main__":
    # Your API credentials
    API_KEY = "VDpt0WQXIjXul4OBrS"
    API_SECRET = "z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9"
    
    # Create bot instance
    bot = SimplifiedGridBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='1000PEPE/USDT:USDT',  # Trading pair
        testnet=False  # Use testnet for testing
    )
    
    # Configuration (all optional - has good defaults)
    bot.max_grid_levels = 30  # Maximum number of levels
    bot.leverage = 20  # Leverage to use
    
    # Strategic level configuration
    bot.use_strategic_levels = True  # Use S/R levels instead of equal spacing
    bot.volume_profile_enabled = True  # Include volume-based levels
    bot.level_merge_threshold = 0.0015  # Merge levels within 0.15%
    
    # Features (all enabled by default)
    bot.enable_trailing_grid = True  # Grid follows price
    bot.enable_dynamic_spacing = True  # Adjust based on volatility  
    bot.enable_profit_taking = True  # Take profits on big moves
    
    # Risk limits
    bot.max_drawdown_percent = 25  # Stop if down 25%
    bot.take_profit_percent = 50  # Stop if up 50%
    bot.position_limit_multiplier = 3  # Max 3x account in positions
    
    # Run the bot
    try:
        bot.run(
            total_investment=47,  # Amount to invest
            check_interval=10  # Check every 10 seconds
        )
    except KeyboardInterrupt:
        bot.stop_bot()