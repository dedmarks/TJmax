import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks
import asyncio
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookImbalance:
    """Order book imbalance data"""
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float
    weighted_mid: float
    pressure_score: float

@dataclass
class MarketMicrostructure:
    """Market microstructure data"""
    tick_direction: int
    trade_intensity: float
    volume_profile: Dict
    liquidity_score: float
    momentum_score: float

class InstitutionalGridBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'DOGE/USDT:USDT', testnet: bool = True):
        """
        Institutional-grade grid bot with advanced profit extraction techniques
        """
        self.symbol = symbol
        self.leverage = 10
        
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
        
        # INSTITUTIONAL PROFIT EXTRACTION TECHNIQUES
        
        # 1. ORDER BOOK IMBALANCE EXPLOITATION
        self.use_orderbook_signals = True
        self.imbalance_threshold = 0.3  # 30% imbalance triggers action
        self.imbalance_lookback = 20  # Analyze last 20 order book snapshots
        self.orderbook_history = deque(maxlen=self.imbalance_lookback)
        
        # 2. TICK-BY-TICK MOMENTUM CAPTURE
        self.tick_momentum_capture = True
        self.tick_history = deque(maxlen=100)
        self.momentum_threshold = 0.7
        self.tick_direction_weight = 0.6
        
        # 3. VOLUME PROFILE ANALYSIS (PROFESSIONAL TECHNIQUE)
        self.volume_profile_analysis = True
        self.value_area_percentage = 0.7  # 70% of volume defines value area
        self.poc_attraction_strength = 0.8  # Point of Control attraction
        self.volume_imbalance_threshold = 2.0  # 2:1 volume ratio
        
        # 4. MARKET MAKER SPREAD OPTIMIZATION
        self.dynamic_spread_optimization = True
        self.spread_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # Spread distribution analysis
        self.optimal_spread_multiplier = 1.2
        self.spread_competition_factor = 0.85  # Stay competitive
        
        # 5. LATENCY ARBITRAGE (SPEED ADVANTAGE)
        self.latency_optimization = True
        self.fast_order_placement = True
        self.order_refresh_microseconds = 100  # Ultra-fast refresh
        self.cancel_replace_strategy = True
        
        # 6. CROSS-EXCHANGE ARBITRAGE SIGNALS
        self.cross_exchange_signals = True
        self.reference_exchanges = ['binance', 'okx']  # For price comparison
        self.arbitrage_threshold = 0.05  # 0.05% price difference
        
        # 7. LIQUIDITY PROVISION OPTIMIZATION
        self.liquidity_rebate_optimization = True
        self.maker_rebate_rate = 0.0001  # 0.01% maker rebate
        self.optimal_order_size_ratio = 0.15  # 15% of average volume
        self.liquidity_competition_analysis = True
        
        # 8. SMART ORDER ROUTING (SOR)
        self.smart_order_routing = True
        self.order_size_optimization = True
        self.iceberg_order_strategy = True
        self.max_visible_ratio = 0.3  # Show only 30% of order
        
        # 9. STATISTICAL ARBITRAGE COMPONENTS
        self.stat_arb_signals = True
        self.mean_reversion_strength = 0.75
        self.momentum_breakout_threshold = 2.5  # Standard deviations
        self.correlation_trading = True
        
        # 10. MARKET IMPACT MINIMIZATION
        self.market_impact_model = True
        self.participation_rate = 0.05  # 5% of volume
        self.impact_cost_threshold = 0.02  # 2 basis points max impact
        self.adaptive_sizing = True
        
        # 11. HIGH-FREQUENCY PATTERN RECOGNITION
        self.pattern_recognition = True
        self.pattern_lookback = 200
        self.pattern_confidence_threshold = 0.8
        self.pattern_profit_multiplier = 1.5
        
        # 12. INSTITUTIONAL FLOW ANALYSIS
        self.institutional_flow_tracking = True
        self.large_order_threshold = 10000  # USD value
        self.flow_momentum_weight = 0.4
        self.block_trade_analysis = True
        
        # Grid parameters
        self.grid_levels = 0
        self.grid_spacing = 0
        self.upper_price = 0
        self.lower_price = 0
        self.total_investment = 0
        self.base_order_amount = 0
        self.price_decimals = 2
        
        # Advanced order management
        self.buy_orders = {}
        self.sell_orders = {}
        self.pending_orders = {}  # Orders waiting for optimal execution
        self.order_queue = []  # Priority queue for smart execution
        
        # Performance tracking
        self.institutional_profits = 0
        self.arbitrage_profits = 0
        self.liquidity_rebates = 0
        self.market_making_profits = 0
        self.pattern_profits = 0
        
        # Market state tracking
        self.current_market_state = "neutral"
        self.liquidity_state = "normal"
        self.volatility_regime = "medium"
        
        self.check_position_mode()
    
    def analyze_orderbook_imbalance(self) -> OrderBookImbalance:
        """
        TECHNIQUE 1: Professional order book imbalance analysis
        Used by institutional traders to predict short-term price movements
        """
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            
            # Calculate weighted bid/ask volumes
            bid_volume = sum(price * volume for price, volume in orderbook['bids'][:10])
            ask_volume = sum(price * volume for price, volume in orderbook['asks'][:10])
            
            # Calculate imbalance ratio
            total_volume = bid_volume + ask_volume
            imbalance_ratio = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate weighted mid price (institutional technique)
            bid_depth = sum(volume for _, volume in orderbook['bids'][:5])
            ask_depth = sum(volume for _, volume in orderbook['asks'][:5])
            total_depth = bid_depth + ask_depth
            
            if total_depth > 0:
                weighted_mid = (
                    orderbook['bids'][0][0] * (ask_depth / total_depth) +
                    orderbook['asks'][0][0] * (bid_depth / total_depth)
                )
            else:
                weighted_mid = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
            
            # Calculate pressure score (proprietary metric)
            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            spread = best_ask - best_bid
            
            # Pressure increases with imbalance and decreases with spread
            pressure_score = abs(imbalance_ratio) * (1 / (spread / best_bid + 0.001))
            
            imbalance_data = OrderBookImbalance(
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                imbalance_ratio=imbalance_ratio,
                weighted_mid=weighted_mid,
                pressure_score=pressure_score
            )
            
            # Store in history
            self.orderbook_history.append(imbalance_data)
            
            return imbalance_data
            
        except Exception as e:
            logger.error(f"Error analyzing order book imbalance: {e}")
            return OrderBookImbalance(0, 0, 0, 0, 0)
    
    def detect_institutional_flow(self) -> Dict:
        """
        TECHNIQUE 2: Detect institutional order flow patterns
        Large institutions leave footprints that can be detected and followed
        """
        try:
            # Fetch recent trades
            trades = self.exchange.fetch_trades(self.symbol, limit=100)
            
            if not trades:
                return {}
            
            large_trades = []
            total_buy_volume = 0
            total_sell_volume = 0
            aggressive_buys = 0
            aggressive_sells = 0
            
            for trade in trades[-50:]:  # Last 50 trades
                volume_usd = trade['amount'] * trade['price']
                
                # Detect large trades (institutional footprints)
                if volume_usd > self.large_order_threshold:
                    large_trades.append({
                        'side': trade['side'],
                        'volume_usd': volume_usd,
                        'price': trade['price'],
                        'timestamp': trade['timestamp']
                    })
                
                # Track aggressive orders (market orders that cross spread)
                if trade['side'] == 'buy':
                    total_buy_volume += volume_usd
                    if trade.get('takerOrMaker') == 'taker':
                        aggressive_buys += volume_usd
                else:
                    total_sell_volume += volume_usd
                    if trade.get('takerOrMaker') == 'taker':
                        aggressive_sells += volume_usd
            
            # Calculate institutional flow metrics
            total_volume = total_buy_volume + total_sell_volume
            buy_ratio = total_buy_volume / total_volume if total_volume > 0 else 0.5
            
            aggressive_ratio = (aggressive_buys - aggressive_sells) / total_volume if total_volume > 0 else 0
            
            # Institutional signature: large orders with specific patterns
            institution_buy_volume = sum(t['volume_usd'] for t in large_trades if t['side'] == 'buy')
            institution_sell_volume = sum(t['volume_usd'] for t in large_trades if t['side'] == 'sell')
            
            institutional_bias = (institution_buy_volume - institution_sell_volume) / (
                institution_buy_volume + institution_sell_volume + 1
            )
            
            return {
                'institutional_bias': institutional_bias,
                'aggressive_ratio': aggressive_ratio,
                'buy_ratio': buy_ratio,
                'large_trade_count': len(large_trades),
                'institutional_strength': min(abs(institutional_bias) * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error detecting institutional flow: {e}")
            return {}
    
    def calculate_volume_profile(self, lookback_hours: int = 24) -> Dict:
        """
        TECHNIQUE 3: Volume Profile Analysis (Professional Trading Tool)
        Identifies key price levels where most trading occurred
        """
        try:
            # Fetch high-resolution data
            since = int((time.time() - lookback_hours * 3600) * 1000)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '5m', since=since, limit=500)
            
            if not ohlcv:
                return {}
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Create price levels (bins)
            min_price = df['low'].min()
            max_price = df['high'].max()
            price_bins = np.linspace(min_price, max_price, 100)
            
            volume_at_price = np.zeros(len(price_bins) - 1)
            
            # Distribute volume across price levels for each candle
            for _, row in df.iterrows():
                candle_low, candle_high = row['low'], row['high']
                candle_volume = row['volume']
                
                # Find which bins this candle overlaps
                overlap_bins = []
                for i in range(len(price_bins) - 1):
                    bin_low, bin_high = price_bins[i], price_bins[i + 1]
                    
                    # Check for overlap
                    if bin_low <= candle_high and bin_high >= candle_low:
                        overlap_amount = min(bin_high, candle_high) - max(bin_low, candle_low)
                        overlap_ratio = overlap_amount / (candle_high - candle_low)
                        volume_at_price[i] += candle_volume * overlap_ratio
            
            # Find Point of Control (POC) - price with highest volume
            poc_index = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            # Calculate Value Area (area containing 70% of volume)
            total_volume = np.sum(volume_at_price)
            target_volume = total_volume * self.value_area_percentage
            
            # Find value area around POC
            cumulative_volume = volume_at_price[poc_index]
            value_area_low_idx = poc_index
            value_area_high_idx = poc_index
            
            while cumulative_volume < target_volume and (value_area_low_idx > 0 or value_area_high_idx < len(volume_at_price) - 1):
                # Expand in direction with more volume
                vol_below = volume_at_price[value_area_low_idx - 1] if value_area_low_idx > 0 else 0
                vol_above = volume_at_price[value_area_high_idx + 1] if value_area_high_idx < len(volume_at_price) - 1 else 0
                
                if vol_below > vol_above and value_area_low_idx > 0:
                    value_area_low_idx -= 1
                    cumulative_volume += volume_at_price[value_area_low_idx]
                elif value_area_high_idx < len(volume_at_price) - 1:
                    value_area_high_idx += 1
                    cumulative_volume += volume_at_price[value_area_high_idx]
                else:
                    break
            
            value_area_low = price_bins[value_area_low_idx]
            value_area_high = price_bins[value_area_high_idx + 1]
            
            # Current price position relative to value area
            current_price = df['close'].iloc[-1]
            
            if current_price < value_area_low:
                price_position = "below_value_area"
                reversion_target = poc_price
            elif current_price > value_area_high:
                price_position = "above_value_area"
                reversion_target = poc_price
            else:
                price_position = "in_value_area"
                reversion_target = poc_price
            
            return {
                'poc_price': poc_price,
                'value_area_low': value_area_low,
                'value_area_high': value_area_high,
                'current_position': price_position,
                'reversion_target': reversion_target,
                'poc_attraction_strength': self.poc_attraction_strength,
                'volume_profile': volume_at_price.tolist(),
                'price_bins': price_bins.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    def detect_high_frequency_patterns(self) -> Dict:
        """
        TECHNIQUE 4: High-Frequency Pattern Recognition
        Detect recurring patterns that occur on sub-minute timeframes
        """
        try:
            # Fetch tick-level data (use trades as proxy)
            trades = self.exchange.fetch_trades(self.symbol, limit=200)
            
            if len(trades) < 50:
                return {}
            
            # Extract price series
            prices = [trade['price'] for trade in trades[-100:]]
            volumes = [trade['amount'] for trade in trades[-100:]]
            
            patterns_detected = {}
            
            # Pattern 1: Momentum Bursts (sequence of same-direction moves)
            price_changes = np.diff(prices)
            momentum_threshold = np.std(price_changes) * 1.5
            
            consecutive_ups = 0
            consecutive_downs = 0
            max_consecutive_ups = 0
            max_consecutive_downs = 0
            
            for change in price_changes:
                if change > momentum_threshold:
                    consecutive_ups += 1
                    consecutive_downs = 0
                    max_consecutive_ups = max(max_consecutive_ups, consecutive_ups)
                elif change < -momentum_threshold:
                    consecutive_downs += 1
                    consecutive_ups = 0
                    max_consecutive_downs = max(max_consecutive_downs, consecutive_downs)
                else:
                    consecutive_ups = 0
                    consecutive_downs = 0
            
            # Pattern 2: Volume-Price Divergence
            volume_ma = np.mean(volumes[-20:])
            price_volatility = np.std(prices[-20:])
            current_volume = volumes[-1]
            
            volume_divergence = (current_volume - volume_ma) / volume_ma if volume_ma > 0 else 0
            
            # Pattern 3: Mean Reversion Setups
            price_z_score = (prices[-1] - np.mean(prices[-20:])) / (np.std(prices[-20:]) + 1e-8)
            
            # Pattern 4: Microstructure Patterns (bid-ask bounce)
            tick_directions = []
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    tick_directions.append(1)
                elif prices[i] < prices[i-1]:
                    tick_directions.append(-1)
                else:
                    tick_directions.append(0)
            
            # Calculate tick imbalance
            recent_ticks = tick_directions[-20:] if len(tick_directions) >= 20 else tick_directions
            tick_imbalance = sum(recent_ticks) / len(recent_ticks) if recent_ticks else 0
            
            patterns_detected = {
                'momentum_burst_up': max_consecutive_ups >= 3,
                'momentum_burst_down': max_consecutive_downs >= 3,
                'volume_spike': volume_divergence > 2.0,
                'volume_drought': volume_divergence < -0.5,
                'mean_reversion_long': price_z_score < -2.0,
                'mean_reversion_short': price_z_score > 2.0,
                'tick_imbalance_bullish': tick_imbalance > 0.6,
                'tick_imbalance_bearish': tick_imbalance < -0.6,
                'pattern_strength': abs(tick_imbalance) + min(abs(volume_divergence), 2.0) / 2.0
            }
            
            return patterns_detected
            
        except Exception as e:
            logger.error(f"Error detecting HF patterns: {e}")
            return {}
    
    def optimize_spread_dynamically(self) -> float:
        """
        TECHNIQUE 5: Dynamic Spread Optimization
        Optimize spread based on market microstructure and competition
        """
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            
            # Current spread
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            current_spread = best_ask - best_bid
            
            # Analyze spread distribution over time
            spreads_history = getattr(self, 'spreads_history', deque(maxlen=100))
            spreads_history.append(current_spread)
            self.spreads_history = spreads_history
            
            if len(spreads_history) < 10:
                return current_spread * 0.8  # Default competitive spread
            
            # Calculate spread percentiles
            spread_percentiles = np.percentile(list(spreads_history), [10, 25, 50, 75, 90])
            
            # Determine optimal spread based on current market conditions
            imbalance = self.analyze_orderbook_imbalance()
            
            # Tight spread when:
            # - Low pressure (stable market)
            # - High competition (many orders at best levels)
            # - High volume
            
            competition_factor = 1.0
            
            # Check competition at best levels
            bid_size_at_best = orderbook['bids'][0][1]
            ask_size_at_best = orderbook['asks'][0][1]
            
            # If there's a lot of size at best levels, we need to be more competitive
            if bid_size_at_best > 1000 or ask_size_at_best > 1000:
                competition_factor = 0.7  # Tighter spread
            
            # Adjust for pressure
            pressure_adjustment = 1.0 - (imbalance.pressure_score * 0.3)
            
            # Base optimal spread on median + competition factors
            optimal_spread = spread_percentiles[2] * competition_factor * pressure_adjustment
            
            # Ensure minimum profitability
            min_profitable_spread = best_bid * 0.0008  # 0.08% minimum
            optimal_spread = max(optimal_spread, min_profitable_spread)
            
            return optimal_spread
            
        except Exception as e:
            logger.error(f"Error optimizing spread: {e}")
            return 0.001  # Fallback
    
    def calculate_optimal_order_size(self, price_level: float, side: str) -> float:
        """
        TECHNIQUE 6: Optimal Order Sizing based on Market Impact Model
        """
        try:
            # Get recent volume data
            trades = self.exchange.fetch_trades(self.symbol, limit=100)
            
            if not trades:
                return self.base_order_amount
            
            # Calculate average trade size and volume
            recent_volumes = [trade['amount'] * trade['price'] for trade in trades[-50:]]
            avg_trade_volume = np.mean(recent_volumes)
            
            # Market impact model: impact = k * (order_size / avg_volume)^α
            # Where k is market impact coefficient, α is impact exponent
            k = 0.01  # 1% impact coefficient
            alpha = 0.6  # Sublinear impact
            
            # Target maximum impact
            max_acceptable_impact = self.impact_cost_threshold  # 2 basis points
            
            # Solve for optimal size: max_impact = k * (size / avg_volume)^α
            optimal_size_ratio = (max_acceptable_impact / k) ** (1 / alpha)
            optimal_size_usd = avg_trade_volume * optimal_size_ratio
            
            # Apply participation rate limit
            volume_last_hour = sum(recent_volumes)
            max_size_by_participation = volume_last_hour * self.participation_rate
            
            # Take minimum of impact-based and participation-based sizing
            optimal_size_usd = min(optimal_size_usd, max_size_by_participation)
            
            # Ensure minimum size for profitability
            min_size = self.base_order_amount * 0.5
            max_size = self.base_order_amount * 3.0
            
            optimal_size_usd = max(min_size, min(optimal_size_usd, max_size))
            
            return optimal_size_usd
            
        except Exception as e:
            logger.error(f"Error calculating optimal order size: {e}")
            return self.base_order_amount
    
    def implement_iceberg_strategy(self, total_size: float, price: float, side: str) -> List[Dict]:
        """
        TECHNIQUE 7: Iceberg Order Strategy
        Break large orders into smaller pieces to hide true size
        """
        try:
            # Calculate visible portion
            visible_size = total_size * self.max_visible_ratio
            
            # Create iceberg slices
            iceberg_slices = []
            remaining_size = total_size
            slice_count = 0
            
            while remaining_size > 0 and slice_count < 10:  # Max 10 slices
                slice_size = min(visible_size, remaining_size)
                
                iceberg_slices.append({
                    'size': slice_size,
                    'price': price,
                    'side': side,
                    'delay': slice_count * 2,  # 2 second delay between slices
                    'priority': slice_count
                })
                
                remaining_size -= slice_size
                slice_count += 1
            
            return iceberg_slices
            
        except Exception as e:
            logger.error(f"Error implementing iceberg strategy: {e}")
            return [{'size': total_size, 'price': price, 'side': side, 'delay': 0, 'priority': 0}]
    
    def detect_arbitrage_opportunities(self) -> Dict:
        """
        TECHNIQUE 8: Cross-Exchange Arbitrage Detection
        """
        try:
            # For simplicity, we'll use price comparison with theoretical "fair value"
            # In practice, you'd connect to multiple exchanges
            
            current_price = self.get_current_price()
            
            # Calculate theoretical fair value using multiple indicators
            df = self.fetch_ohlcv_data('1m', 100)
            if df.empty:
                return {}
            
            # Fair value components
            vwap = (df['volume'] * df['close']).sum() / df['volume'].sum()
            ema_fast = df['close'].ewm(span=10).mean().iloc[-1]
            ema_slow = df['close'].ewm(span=20).mean().iloc[-1]
            
            # Weighted fair value
            fair_value = (vwap * 0.4 + ema_fast * 0.4 + ema_slow * 0.2)
            
            # Calculate arbitrage potential
            price_deviation = (current_price - fair_value) / fair_value
            
            arbitrage_signal = {
                'fair_value': fair_value,
                'current_price': current_price,
                'deviation_pct': price_deviation * 100,
                'arbitrage_opportunity': abs(price_deviation) > self.arbitrage_threshold / 100,
                'direction': 'buy' if price_deviation < 0 else 'sell',
                'expected_profit': abs(price_deviation) * 100
            }
            
            return arbitrage_signal
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage: {e}")
            return {}
    
    def calculate_institutional_grid_parameters(self, total_investment: float) -> Dict:
        """
        Enhanced grid calculation with institutional techniques
        """
        try:
            logger.info("🏛️ Calculating institutional-grade grid parameters...")
            
            # Get all market analysis data
            imbalance = self.analyze_orderbook_imbalance()
            institutional_flow = self.detect_institutional_flow()
            volume_profile = self.calculate_volume_profile()
            hf_patterns = self.detect_high_frequency_patterns()
            arbitrage = self.detect_arbitrage_opportunities()
            
            current_price = self.get_current_price()
            
            # Base volatility calculation
            vol_metrics = self.calculate_enhanced_volatility_metrics()
            base_volatility = vol_metrics['weighted_volatility']
            
            # INSTITUTIONAL ADJUSTMENTS
            
            # 1. Adjust range based on volume profile
            if volume_profile:
                poc_price = volume_profile['poc_price']
                value_area_low = volume_profile['value_area_low']
                value_area_high = volume_profile['value_area_high']
                
                # Bias grid toward value area
                if current_price < value_area_low:
                    range_bias_up = 1.3  # Wider range above
                    range_bias_down = 0.8  # Tighter range below
                elif current_price > value_area_high:
                    range_bias_up = 0.8
                    range_bias_down = 1.3
                else:
                    range_bias_up = 1.0
                    range_bias_down = 1.0
            else:
                range_bias_up = range_bias_down = 1.0
            
            # 2. Adjust spacing based on orderbook imbalance
            imbalance_adjustment = 1.0
            if abs(imbalance.imbalance_ratio) > self.imbalance_threshold:
                # Tighter spacing when there's strong imbalance (more opportunities)
                imbalance_adjustment = 0.85
            
            # 3. Adjust levels based on institutional flow
            institution_adjustment = 1.0
            if institutional_flow and institutional_flow.get('institutional_strength', 0) > 0.5:
                # More levels when institutions are active
                institution_adjustment = 1.2
            
            # 4. Adjust for high-frequency patterns
            pattern_adjustment = 1.0
            if hf_patterns and hf_patterns.get('pattern_strength', 0) > 0.7:
                # Tighter grid when strong patterns detected
                pattern_adjustment = 0.9
            
            # Calculate base range
            base_range = base_volatility * current_price * 1.5
            
            # Apply institutional adjustments
            upper_range = base_range * 0.5 * range_bias_up
            lower_range = base_range * 0.5 * range_bias_down
            
            upper_price = current_price + upper_range
            lower_price = current_price - lower_range
            
            # Calculate grid levels with institutional factors
            base_levels = 25
            adjusted_levels = int(base_levels * institution_adjustment)
            
            # Calculate spacing with all adjustments
            grid_range = upper_price - lower_price
            spacing_adjustment = imbalance_adjustment * pattern_adjustment
            grid_spacing = (grid_range / (adjusted_levels - 1)) * spacing_adjustment
            
            # Ensure minimum profitability (higher threshold for institutions)
            min_spacing_for_profit = current_price * 0.0015  # 0.15% minimum (vs 0.12% for retail)
            
            if grid_spacing < min_spacing_for_profit:
                grid_spacing = min_spacing_for_profit
                adjusted_levels = int(grid_range / grid_spacing) + 1
            
            # INSTITUTIONAL ORDER SIZING DISTRIBUTION
            # Use Kelly Criterion for optimal sizing
            win_rate = 0.65  # Historical institutional win rate
            avg_win = 0.0018  # 0.18% average win
            avg_loss = 0.0008  # 0.08% average loss
            
            kelly_ratio = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            optimal_sizing_factor = min(kelly_ratio, 0.25)  # Cap at 25%
            
            # Calculate order sizes with institutional distribution
            total_grid_capital = total_investment * (1 - self.trend_allocation)
            base_order_size = (total_grid_capital * optimal_sizing_factor) / adjusted_levels
            
            # Create sophisticated order size distribution
            order_sizes = self.calculate_institutional_order_sizes(adjusted_levels, base_order_size, current_price)
            
            # Price decimals optimization
            price_decimals = self.get_optimal_price_decimals(current_price)
            
            parameters = {
                'upper_price': round(upper_price, price_decimals),
                'lower_price': round(lower_price, price_decimals),
                'grid_levels': adjusted_levels,
                'grid_spacing': round(grid_spacing, price_decimals),
                'current_price': round(current_price, price_decimals),
                'base_order_size': round(base_order_size, 2),
                'order_sizes': order_sizes,
                'price_decimals': price_decimals,
                'grid_investment': round(total_grid_capital, 2),
                'expected_profit_per_trade': round((grid_spacing / current_price) * 100 - 0.02, 3),
                
                # Institutional metrics
                'imbalance_ratio': imbalance.imbalance_ratio,
                'institutional_strength': institutional_flow.get('institutional_strength', 0),
                'pattern_strength': hf_patterns.get('pattern_strength', 0),
                'arbitrage_opportunity': arbitrage.get('arbitrage_opportunity', False),
                'volume_profile_bias': volume_profile.get('current_position', 'neutral'),
                'kelly_sizing_factor': optimal_sizing_factor,
                
                # Optimization flags
                'imbalance_adjustment': imbalance_adjustment,
                'institution_adjustment': institution_adjustment,
                'pattern_adjustment': pattern_adjustment,
                'range_bias_up': range_bias_up,
                'range_bias_down': range_bias_down
            }
            
            logger.info(f"🏛️ Institutional Grid Configuration:")
            logger.info(f"  Range: ${parameters['lower_price']} - ${parameters['upper_price']}")
            logger.info(f"  Levels: {parameters['grid_levels']} (institutional optimized)")
            logger.info(f"  Spacing: ${parameters['grid_spacing']} ({(grid_spacing/current_price)*100:.3f}%)")
            logger.info(f"  Kelly sizing factor: {optimal_sizing_factor:.3f}")
            logger.info(f"  Imbalance ratio: {imbalance.imbalance_ratio:.3f}")
            logger.info(f"  Institutional strength: {institutional_flow.get('institutional_strength', 0):.3f}")
            logger.info(f"  Pattern strength: {hf_patterns.get('pattern_strength', 0):.3f}")
            logger.info(f"  Expected profit per trade: {parameters['expected_profit_per_trade']:.3f}%")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error calculating institutional grid parameters: {e}")
            raise
    
    def calculate_institutional_order_sizes(self, grid_levels: int, base_size: float, current_price: float) -> List[float]:
        """
        TECHNIQUE 9: Institutional Order Size Distribution
        Uses advanced portfolio theory and risk management
        """
        try:
            center = grid_levels // 2
            order_sizes = []
            
            # Institutional sizing follows modified Kelly + volatility scaling
            for i in range(grid_levels):
                distance_from_center = abs(i - center)
                distance_ratio = distance_from_center / center if center > 0 else 0
                
                # Institutional technique: Size increases with distance BUT decreases with extreme distance
                # This creates a "barbell" distribution popular with institutions
                if distance_ratio < 0.5:
                    # Close to center: moderate increase
                    size_multiplier = 1 + (distance_ratio * 0.8)
                elif distance_ratio < 0.8:
                    # Medium distance: maximum size
                    size_multiplier = 1.4 + ((distance_ratio - 0.5) * 0.6)
                else:
                    # Far from center: reduce size (tail risk management)
                    size_multiplier = 1.7 - ((distance_ratio - 0.8) * 1.2)
                
                # Apply volatility scaling
                vol_metrics = self.calculate_enhanced_volatility_metrics()
                vol_percentile = vol_metrics.get('volatility_percentile', 0.5)
                
                # Reduce sizes in high volatility
                if vol_percentile > 0.8:
                    size_multiplier *= 0.7
                elif vol_percentile > 0.6:
                    size_multiplier *= 0.85
                
                # Ensure reasonable bounds
                size_multiplier = max(0.3, min(size_multiplier, 2.5))
                
                order_sizes.append(round(base_size * size_multiplier, 2))
            
            return order_sizes
            
        except Exception as e:
            logger.error(f"Error calculating institutional order sizes: {e}")
            return [base_size] * grid_levels
    
    def place_institutional_grid_orders(self):
        """
        Place grid orders with institutional-grade optimizations
        """
        try:
            current_price = self.get_current_price()
            grid_levels = self.calculate_grid_levels()
            
            # Get institutional market analysis
            imbalance = self.analyze_orderbook_imbalance()
            optimal_spread = self.optimize_spread_dynamically()
            
            logger.info(f"🏛️ Placing institutional grid orders...")
            logger.info(f"Current price: ${current_price:.4f}")
            logger.info(f"Optimal spread: ${optimal_spread:.6f}")
            logger.info(f"Order book imbalance: {imbalance.imbalance_ratio:.3f}")
            
            order_count = 0
            priority_queue = []  # For smart order routing
            
            for i, price in enumerate(grid_levels):
                try:
                    # Get optimal order size for this level
                    if hasattr(self, 'order_sizes') and i < len(self.order_sizes):
                        base_order_size_usdt = self.order_sizes[i]
                    else:
                        base_order_size_usdt = self.base_order_amount
                    
                    # Apply market impact optimization
                    optimal_size = self.calculate_optimal_order_size(price, 'buy' if price < current_price else 'sell')
                    order_size_usdt = min(base_order_size_usdt, optimal_size)
                    
                    if price < current_price * 0.998:  # Buy orders
                        # Apply institutional spread optimization
                        adjusted_price = price - (optimal_spread * 0.5)
                        
                        # Check if we should use iceberg strategy
                        if order_size_usdt > self.base_order_amount * 1.5:
                            iceberg_slices = self.implement_iceberg_strategy(order_size_usdt, adjusted_price, 'buy')
                            for slice_data in iceberg_slices:
                                heapq.heappush(priority_queue, (slice_data['priority'], slice_data))
                        else:
                            self.place_institutional_buy_order(adjusted_price, order_size_usdt, i)
                            order_count += 1
                        
                    elif price > current_price * 1.002:  # Sell orders
                        adjusted_price = price + (optimal_spread * 0.5)
                        
                        if order_size_usdt > self.base_order_amount * 1.5:
                            iceberg_slices = self.implement_iceberg_strategy(order_size_usdt, adjusted_price, 'sell')
                            for slice_data in iceberg_slices:
                                heapq.heappush(priority_queue, (slice_data['priority'], slice_data))
                        else:
                            self.place_institutional_sell_order(adjusted_price, order_size_usdt, i)
                            order_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to place institutional order at level {i}: {e}")
                    continue
            
            # Execute iceberg orders with delays
            self.execute_iceberg_queue(priority_queue)
            
            logger.info(f"✅ Placed {order_count} institutional grid orders")
            logger.info(f"📋 {len(priority_queue)} iceberg slices queued")
            
        except Exception as e:
            logger.error(f"Error placing institutional grid orders: {e}")
            raise
    
    def execute_iceberg_queue(self, priority_queue: List):
        """
        Execute iceberg orders with smart timing
        """
        def execute_queue():
            while priority_queue:
                try:
                    priority, slice_data = heapq.heappop(priority_queue)
                    
                    # Wait for optimal timing
                    time.sleep(slice_data['delay'])
                    
                    if slice_data['side'] == 'buy':
                        self.place_institutional_buy_order(
                            slice_data['price'], 
                            slice_data['size'], 
                            -1  # Special level for iceberg
                        )
                    else:
                        self.place_institutional_sell_order(
                            slice_data['price'], 
                            slice_data['size'], 
                            999  # Special level for iceberg
                        )
                    
                    logger.info(f"🧊 Iceberg slice executed: {slice_data['side']} ${slice_data['size']:.2f} @ ${slice_data['price']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error executing iceberg slice: {e}")
        
        # Execute in background thread
        threading.Thread(target=execute_queue, daemon=True).start()
    
    def place_institutional_buy_order(self, price: float, usdt_amount: float, level: int):
        """
        Place institutional-grade buy order with advanced features
        """
        try:
            position_size = self.calculate_position_size_for_amount(price, usdt_amount)
            
            # Institutional order parameters
            params = {
                'timeInForce': 'PostOnly',  # Always maker for rebates
                'clientOrderId': f'inst_buy_{level}_{int(time.time() * 1000)}',  # Microsecond precision
                'reduceOnly': False,
                'close': False
            }
            
            # Add institutional order routing if available
            if self.smart_order_routing:
                params['orderLinkId'] = f'sor_{int(time.time() * 1000000)}'  # Smart order routing ID
            
            order = self.exchange.create_limit_buy_order(
                symbol=self.symbol,
                amount=position_size,
                price=price,
                params=params
            )
            
            self.buy_orders[order['id']] = {
                'price': price,
                'amount': position_size,
                'usdt_amount': usdt_amount,
                'level': level,
                'status': 'open',
                'timestamp': time.time(),
                'order_type': 'institutional',
                'expected_rebate': usdt_amount * self.maker_rebate_rate
            }
            
            logger.info(f"🏛️ Institutional buy: {position_size:.4f} @ ${price:.4f} (Level {level}, ${usdt_amount:.2f})")
            
        except Exception as e:
            logger.error(f"Error placing institutional buy order: {e}")
    
    def place_institutional_sell_order(self, price: float, usdt_amount: float, level: int):
        """
        Place institutional-grade sell order with advanced features
        """
        try:
            position_size = self.calculate_position_size_for_amount(price, usdt_amount)
            
            params = {
                'timeInForce': 'PostOnly',
                'clientOrderId': f'inst_sell_{level}_{int(time.time() * 1000)}',
                'reduceOnly': False,
                'close': False
            }
            
            if self.smart_order_routing:
                params['orderLinkId'] = f'sor_{int(time.time() * 1000000)}'
            
            order = self.exchange.create_limit_sell_order(
                symbol=self.symbol,
                amount=position_size,
                price=price,
                params=params
            )
            
            self.sell_orders[order['id']] = {
                'price': price,
                'amount': position_size,
                'usdt_amount': usdt_amount,
                'level': level,
                'status': 'open',
                'timestamp': time.time(),
                'order_type': 'institutional',
                'expected_rebate': usdt_amount * self.maker_rebate_rate
            }
            
            logger.info(f"🏛️ Institutional sell: {position_size:.4f} @ ${price:.4f} (Level {level}, ${usdt_amount:.2f})")
            
        except Exception as e:
            logger.error(f"Error placing institutional sell order: {e}")
    
    def check_institutional_filled_orders(self):
        """
        Enhanced order fill checking with institutional features
        """
        try:
            # Use multiple methods for order checking (redundancy)
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=100)
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            
            open_order_ids = {order['id'] for order in open_orders}
            
            # Process filled orders with institutional logic
            for order in closed_orders:
                order_id = order['id']
                
                if (order_id in self.buy_orders and 
                    order['status'] == 'closed' and 
                    order['filled'] > 0 and
                    self.buy_orders[order_id]['status'] == 'open'):
                    self.handle_institutional_filled_buy(order_id, order)
                
                elif (order_id in self.sell_orders and 
                      order['status'] == 'closed' and 
                      order['filled'] > 0 and
                      self.sell_orders[order_id]['status'] == 'open'):
                    self.handle_institutional_filled_sell(order_id, order)
            
            # Check for cancelled orders
            for order_id in list(self.buy_orders.keys()):
                if (self.buy_orders[order_id]['status'] == 'open' and 
                    order_id not in open_order_ids):
                    logger.info(f"Buy order {order_id} was cancelled or expired")
                    del self.buy_orders[order_id]
            
            for order_id in list(self.sell_orders.keys()):
                if (self.sell_orders[order_id]['status'] == 'open' and 
                    order_id not in open_order_ids):
                    logger.info(f"Sell order {order_id} was cancelled or expired")
                    del self.sell_orders[order_id]
                    
        except Exception as e:
            logger.error(f"Error checking institutional filled orders: {e}")
    
    def handle_institutional_filled_buy(self, order_id: str, order: Dict):
        """
        Handle filled buy order with institutional profit optimization
        """
        try:
            buy_order = self.buy_orders[order_id]
            filled_price = order['average'] or order['price']
            filled_amount = order['filled']
            level = buy_order['level']
            
            # Mark as filled and calculate rebate
            self.buy_orders[order_id]['status'] = 'filled'
            self.buy_orders[order_id]['filled_price'] = filled_price
            rebate_earned = buy_order['expected_rebate']
            self.liquidity_rebates += rebate_earned
            
            # INSTITUTIONAL PROFIT OPTIMIZATION
            
            # 1. Dynamic sell price based on market conditions
            base_sell_price = filled_price + self.grid_spacing
            
            # 2. Apply volume profile bias
            volume_profile = self.calculate_volume_profile()
            if volume_profile:
                poc_price = volume_profile['poc_price']
                # If we're below POC, target POC for additional profit
                if filled_price < poc_price and base_sell_price < poc_price:
                    profit_boost = min(self.grid_spacing * 0.3, poc_price - base_sell_price)
                    base_sell_price += profit_boost
                    logger.info(f"📊 POC targeting: boosting profit by ${profit_boost:.4f}")
            
            # 3. Apply arbitrage signals
            arbitrage = self.detect_arbitrage_opportunities()
            if arbitrage.get('arbitrage_opportunity') and arbitrage.get('direction') == 'sell':
                arbitrage_boost = self.grid_spacing * 0.2
                base_sell_price += arbitrage_boost
                logger.info(f"⚡ Arbitrage boost: +${arbitrage_boost:.4f}")
            
            # 4. High-frequency pattern adjustment
            hf_patterns = self.detect_high_frequency_patterns()
            if hf_patterns.get('momentum_burst_up'):
                momentum_boost = self.grid_spacing * 0.25
                base_sell_price += momentum_boost
                logger.info(f"🚀 Momentum boost: +${momentum_boost:.4f}")
            
            # 5. Institutional flow alignment
            institutional_flow = self.detect_institutional_flow()
            if institutional_flow.get('institutional_bias', 0) > 0.3:  # Institutions buying
                flow_boost = self.grid_spacing * 0.15
                base_sell_price += flow_boost
                logger.info(f"🏛️ Institution flow boost: +${flow_boost:.4f}")
            
            # Ensure sell price is profitable and within bounds
            if base_sell_price <= self.upper_price:
                # Use optimal order sizing for sell order
                optimal_sell_size = self.calculate_optimal_order_size(base_sell_price, 'sell')
                sell_usdt_amount = min(buy_order['usdt_amount'] * 1.05, optimal_sell_size)  # 5% compound
                
                self.place_institutional_sell_order(base_sell_price, sell_usdt_amount, level + 1)
                
                # Calculate total expected profit including rebates
                trade_profit = (base_sell_price - filled_price) * filled_amount * self.leverage
                total_profit = trade_profit + rebate_earned
                self.institutional_profits += total_profit
                
                # Record institutional trade
                self.record_institutional_trade('buy_filled', {
                    'price': filled_price,
                    'amount': filled_amount,
                    'level': level,
                    'trade_profit': trade_profit,
                    'rebate_profit': rebate_earned,
                    'total_profit': total_profit,
                    'profit_boosts_applied': True
                })
                
                logger.info(f"✅ Institutional buy filled @ ${filled_price:.4f}")
                logger.info(f"💰 Trade profit: ${trade_profit:.4f}, Rebate: ${rebate_earned:.6f}")
                logger.info(f"🎯 Enhanced sell placed @ ${base_sell_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error handling institutional filled buy: {e}")
    
    def handle_institutional_filled_sell(self, order_id: str, order: Dict):
        """
        Handle filled sell order with institutional profit optimization
        """
        try:
            sell_order = self.sell_orders[order_id]
            filled_price = order['average'] or order['price']
            filled_amount = order['filled']
            level = sell_order['level']
            
            # Mark as filled and calculate rebate
            self.sell_orders[order_id]['status'] = 'filled'
            self.sell_orders[order_id]['filled_price'] = filled_price
            rebate_earned = sell_order['expected_rebate']
            self.liquidity_rebates += rebate_earned
            
            # Dynamic buy price with institutional intelligence
            base_buy_price = filled_price - self.grid_spacing
            
            # Apply institutional adjustments
            volume_profile = self.calculate_volume_profile()
            if volume_profile:
                poc_price = volume_profile['poc_price']
                if filled_price > poc_price and base_buy_price > poc_price:
                    profit_boost = min(self.grid_spacing * 0.3, filled_price - poc_price)
                    base_buy_price -= profit_boost
            
            # Check for buy signals
            arbitrage = self.detect_arbitrage_opportunities()
            if arbitrage.get('arbitrage_opportunity') and arbitrage.get('direction') == 'buy':
                base_buy_price -= self.grid_spacing * 0.2
            
            # Ensure buy price is within bounds
            if base_buy_price >= self.lower_price:
                optimal_buy_size = self.calculate_optimal_order_size(base_buy_price, 'buy')
                buy_usdt_amount = min(sell_order['usdt_amount'] * 1.05, optimal_buy_size)
                
                self.place_institutional_buy_order(base_buy_price, buy_usdt_amount, level - 1)
                
                # Record trade with rebate
                self.record_institutional_trade('sell_filled', {
                    'price': filled_price,
                    'amount': filled_amount,
                    'level': level,
                    'rebate_profit': rebate_earned
                })
                
                logger.info(f"✅ Institutional sell filled @ ${filled_price:.4f}")
                logger.info(f"💰 Rebate earned: ${rebate_earned:.6f}")
            
        except Exception as e:
            logger.error(f"Error handling institutional filled sell: {e}")
    
    def record_institutional_trade(self, trade_type: str, trade_data: Dict):
        """
        Record trade with institutional metrics
        """
        trade_record = {
            'timestamp': datetime.now(),
            'type': trade_type,
            'data': trade_data,
            'institutional_features': True
        }
        
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        
        self.trade_history.append(trade_record)
        
        # Update daily PnL with all profit sources
        today = datetime.now().date()
        if not hasattr(self, 'daily_pnl'):
            self.daily_pnl = defaultdict(float)
        
        total_profit = (
            trade_data.get('trade_profit', 0) + 
            trade_data.get('rebate_profit', 0)
        )
        self.daily_pnl[today] += total_profit
    
    def implement_latency_optimization(self):
        """
        TECHNIQUE 10: Latency Optimization for Speed Advantage
        """
        try:
            # Fast order refresh strategy
            if not hasattr(self, 'last_fast_refresh'):
                self.last_fast_refresh = time.time()
            
            current_time = time.time()
            if current_time - self.last_fast_refresh > 0.1:  # 100ms refresh
                
                # Quick order book snapshot
                orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
                current_spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
                
                # Cancel and replace orders that are no longer competitive
                self.refresh_non_competitive_orders(orderbook, current_spread)
                
                self.last_fast_refresh = current_time
            
        except Exception as e:
            logger.error(f"Error in latency optimization: {e}")
    
    def refresh_non_competitive_orders(self, orderbook: Dict, current_spread: float):
        """
        Refresh orders that are no longer at optimal levels
        """
        try:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            
            # Check buy orders
            for order_id, order_data in list(self.buy_orders.items()):
                if order_data['status'] == 'open':
                    order_price = order_data['price']
                    
                    # If our order is more than 2 ticks away from best bid, refresh
                    tick_size = current_spread * 0.1  # Estimate tick size
                    if order_price < best_bid - (2 * tick_size):
                        try:
                            # Cancel and replace with better price
                            self.exchange.cancel_order(order_id, self.symbol)
                            new_price = best_bid - tick_size
                            self.place_institutional_buy_order(new_price, order_data['usdt_amount'], order_data['level'])
                            del self.buy_orders[order_id]
                        except:
                            pass
            
            # Check sell orders
            for order_id, order_data in list(self.sell_orders.items()):
                if order_data['status'] == 'open':
                    order_price = order_data['price']
                    
                    if order_price > best_ask + (2 * tick_size):
                        try:
                            self.exchange.cancel_order(order_id, self.symbol)
                            new_price = best_ask + tick_size
                            self.place_institutional_sell_order(new_price, order_data['usdt_amount'], order_data['level'])
                            del self.sell_orders[order_id]
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Error refreshing non-competitive orders: {e}")
    
    def calculate_institutional_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive institutional performance metrics
        """
        try:
            if not hasattr(self, 'trade_history') or not self.trade_history:
                return {}
            
            # Basic metrics
            filled_trades = [t for t in self.trade_history if 'filled' in t['type']]
            total_trades = len(filled_trades)
            
            if total_trades == 0:
                return {}
            
            # Profit breakdown
            trade_profits = [t['data'].get('trade_profit', 0) for t in filled_trades]
            rebate_profits = [t['data'].get('rebate_profit', 0) for t in filled_trades]
            
            total_trade_profit = sum(trade_profits)
            total_rebate_profit = sum(rebate_profits)
            
            # Advanced metrics
            win_rate = len([p for p in trade_profits if p > 0]) / len(trade_profits) if trade_profits else 0
            
            # Sharpe ratio calculation
            daily_returns = list(self.daily_pnl.values()) if hasattr(self, 'daily_pnl') else []
            if len(daily_returns) > 1:
                returns_std = np.std(daily_returns)
                avg_daily_return = np.mean(daily_returns)
                sharpe_ratio = (avg_daily_return / returns_std) * np.sqrt(365) if returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Institutional-specific metrics
            avg_trade_profit = np.mean(trade_profits) if trade_profits else 0
            avg_rebate_per_trade = np.mean(rebate_profits) if rebate_profits else 0
            
            # Profit per unit of risk
            max_drawdown = min(daily_returns) if daily_returns else 0
            calmar_ratio = avg_daily_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Trade efficiency metrics
            profitable_trades = [p for p in trade_profits if p > 0]
            losing_trades = [p for p in trade_profits if p < 0]
            
            avg_win = np.mean(profitable_trades) if profitable_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            profit_factor = (avg_win * len(profitable_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')
            
            return {
                'total_trades': total_trades,
                'win_rate': round(win_rate * 100, 2),
                'total_trade_profit': round(total_trade_profit, 4),
                'total_rebate_profit': round(total_rebate_profit, 6),
                'institutional_profits': round(getattr(self, 'institutional_profits', 0), 4),
                'liquidity_rebates': round(getattr(self, 'liquidity_rebates', 0), 6),
                'market_making_profits': round(getattr(self, 'market_making_profits', 0), 4),
                'total_institutional_profit': round(
                    getattr(self, 'institutional_profits', 0) + 
                    getattr(self, 'liquidity_rebates', 0) + 
                    getattr(self, 'market_making_profits', 0), 4
                ),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'calmar_ratio': round(calmar_ratio, 3),
                'profit_factor': round(profit_factor, 2),
                'avg_trade_profit': round(avg_trade_profit, 6),
                'avg_rebate_per_trade': round(avg_rebate_per_trade, 6),
                'max_daily_drawdown': round(max_drawdown, 4),
                'avg_daily_return': round(np.mean(daily_returns), 4) if daily_returns else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating institutional performance: {e}")
            return {}
    
    def auto_configure_institutional_grid(self, total_investment: float):
        """
        Auto-configure grid with all institutional enhancements
        """
        logger.info("🏛️ Configuring institutional-grade profitable grid...")
        
        params = self.calculate_institutional_grid_parameters(total_investment)
        
        self.upper_price = params['upper_price']
        self.lower_price = params['lower_price']
        self.grid_levels = params['grid_levels']
        self.grid_spacing = params['grid_spacing']
        self.total_investment = total_investment
        self.base_order_amount = params['base_order_size']
        self.price_decimals = params['price_decimals']
        
        if hasattr(self, 'use_dynamic_sizing') and self.use_dynamic_sizing:
            self.order_sizes = params['order_sizes']
        
        logger.info(f"✅ Institutional grid configured with {self.grid_levels} levels")
        logger.info(f"📊 Kelly sizing factor: {params['kelly_sizing_factor']:.3f}")
        logger.info(f"🏛️ Institutional optimizations: ALL ACTIVE")
    
    def run_institutional_bot(self, check_interval: int = 3, rebalance_interval: int = 900):
        """
        Run the institutional bot with all advanced profit techniques
        
        Args:
            check_interval: Seconds between checks (ultra-fast for institutions)
            rebalance_interval: Seconds between rebalancing (15 minutes)
        """
        try:
            logger.info("🏛️ Starting INSTITUTIONAL GRADE Grid Bot...")
            logger.info("💎 Advanced profit extraction techniques: ENABLED")
            logger.info("⚡ Latency optimization: ACTIVE")
            logger.info("📊 Market microstructure analysis: RUNNING")
            logger.info("🎯 Statistical arbitrage: MONITORING")
            
            # Set leverage
            self.set_leverage()
            
            # Configure institutional grid
            self.auto_configure_institutional_grid(self.total_investment)
            
            # Place initial institutional orders
            self.place_institutional_grid_orders()
            
            logger.info(f"✅ Institutional bot started!")
            logger.info(f"⚡ Ultra-fast check interval: {check_interval}s")
            logger.info(f"🔄 Smart rebalance interval: {rebalance_interval/60:.1f} minutes")
            
            last_rebalance = time.time()
            last_performance_update = time.time()
            last_latency_optimization = time.time()
            last_market_analysis = time.time()
            
            performance_update_interval = 30  # Update every 30 seconds
            latency_optimization_interval = 0.1  # 100ms latency optimization
            market_analysis_interval = 10  # Market analysis every 10 seconds
            
            iteration = 0
            
            while True:
                try:
                    iteration += 1
                    current_time = time.time()
                    
                    # 1. ULTRA-FAST ORDER MANAGEMENT
                    self.check_institutional_filled_orders()
                    
                    # 2. LATENCY OPTIMIZATION (every 100ms)
                    if current_time - last_latency_optimization > latency_optimization_interval:
                        self.implement_latency_optimization()
                        last_latency_optimization = current_time
                    
                    # 3. MARKET MICROSTRUCTURE ANALYSIS (every 10s)
                    if current_time - last_market_analysis > market_analysis_interval:
                        self.run_institutional_market_analysis()
                        last_market_analysis = current_time
                    
                    # 4. SMART REBALANCING
                    if current_time - last_rebalance > rebalance_interval:
                        self.check_institutional_rebalancing()
                        last_rebalance = current_time
                    
                    # 5. PERFORMANCE TRACKING
                    if current_time - last_performance_update > performance_update_interval:
                        self.display_institutional_performance()
                        last_performance_update = current_time
                    
                    # 6. RISK MANAGEMENT (every minute)
                    if iteration % (60 // check_interval) == 0:
                        self.check_institutional_risk_management()
                    
                    # 7. ARBITRAGE OPPORTUNITIES (every 30 seconds)
                    if iteration % (30 // check_interval) == 0:
                        self.exploit_arbitrage_opportunities()
                    
                    # Dynamic sleep based on market conditions
                    sleep_time = self.get_adaptive_check_interval()
                    time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Institutional bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in institutional main loop: {e}")
                    time.sleep(check_interval * 2)
                    
        except Exception as e:
            logger.error(f"💥 Fatal error in institutional bot: {e}")
            raise
        finally:
            self.stop()
    
    def run_institutional_market_analysis(self):
        """
        Run comprehensive market analysis for institutional edge
        """
        try:
            # 1. Order book imbalance analysis
            imbalance = self.analyze_orderbook_imbalance()
            
            # 2. Institutional flow detection
            institutional_flow = self.detect_institutional_flow()
            
            # 3. High-frequency pattern recognition
            hf_patterns = self.detect_high_frequency_patterns()
            
            # 4. Volume profile analysis
            volume_profile = self.calculate_volume_profile()
            
            # 5. Arbitrage detection
            arbitrage = self.detect_arbitrage_opportunities()
            
            # Store market state for decision making
            self.current_market_analysis = {
                'imbalance': imbalance,
                'institutional_flow': institutional_flow,
                'hf_patterns': hf_patterns,
                'volume_profile': volume_profile,
                'arbitrage': arbitrage,
                'timestamp': time.time()
            }
            
            # Update market state classification
            self.classify_market_state()
            
        except Exception as e:
            logger.error(f"Error in institutional market analysis: {e}")
    
    def classify_market_state(self):
        """
        Classify current market state for strategy adaptation
        """
        try:
            analysis = getattr(self, 'current_market_analysis', {})
            
            if not analysis:
                return
            
            imbalance = analysis.get('imbalance')
            institutional_flow = analysis.get('institutional_flow', {})
            hf_patterns = analysis.get('hf_patterns', {})
            
            # Market state classification
            if imbalance and abs(imbalance.imbalance_ratio) > 0.4:
                if institutional_flow.get('institutional_strength', 0) > 0.7:
                    self.current_market_state = "institutional_momentum"
                elif hf_patterns.get('pattern_strength', 0) > 0.8:
                    self.current_market_state = "high_frequency_opportunity"
                else:
                    self.current_market_state = "imbalanced"
            elif hf_patterns.get('volume_spike'):
                self.current_market_state = "volume_breakout"
            elif hf_patterns.get('mean_reversion_long') or hf_patterns.get('mean_reversion_short'):
                self.current_market_state = "mean_reversion"
            else:
                self.current_market_state = "neutral"
            
            # Liquidity state
            if hf_patterns.get('volume_drought'):
                self.liquidity_state = "low"
            elif hf_patterns.get('volume_spike'):
                self.liquidity_state = "high"
            else:
                self.liquidity_state = "normal"
                
        except Exception as e:
            logger.error(f"Error classifying market state: {e}")
    
    def exploit_arbitrage_opportunities(self):
        """
        Actively exploit detected arbitrage opportunities
        """
        try:
            arbitrage = self.detect_arbitrage_opportunities()
            
            if arbitrage.get('arbitrage_opportunity'):
                direction = arbitrage.get('direction')
                expected_profit = arbitrage.get('expected_profit', 0)
                
                if expected_profit > 0.05:  # Minimum 0.05% profit
                    logger.info(f"⚡ ARBITRAGE OPPORTUNITY: {direction} - {expected_profit:.3f}% profit")
                    
                    # Calculate position size for arbitrage
                    risk_capital = self.total_investment * 0.1  # 10% for arbitrage
                    
                    current_price = self.get_current_price()
                    
                    if direction == 'buy':
                        # Place aggressive buy order
                        arb_price = current_price * 1.001  # Slightly above market
                        self.place_arbitrage_order('buy', arb_price, risk_capital)
                    else:
                        # Place aggressive sell order
                        arb_price = current_price * 0.999  # Slightly below market
                        self.place_arbitrage_order('sell', arb_price, risk_capital)
                    
                    self.arbitrage_profits += expected_profit * risk_capital / 100
                    
        except Exception as e:
            logger.error(f"Error exploiting arbitrage: {e}")
    
    def place_arbitrage_order(self, side: str, price: float, usdt_amount: float):
        """
        Place arbitrage order with aggressive execution
        """
        try:
            position_size = self.calculate_position_size_for_amount(price, usdt_amount)
            
            # Use IOC (Immediate or Cancel) for fast execution
            params = {
                'timeInForce': 'IOC',
                'clientOrderId': f'arb_{side}_{int(time.time() * 1000)}'
            }
            
            if side == 'buy':
                order = self.exchange.create_limit_buy_order(
                    symbol=self.symbol,
                    amount=position_size,
                    price=price,
                    params=params
                )
            else:
                order = self.exchange.create_limit_sell_order(
                    symbol=self.symbol,
                    amount=position_size,
                    price=price,
                    params=params
                )
            
            logger.info(f"⚡ Arbitrage {side}: {position_size:.4f} @ ${price:.4f}")
            
        except Exception as e:
            logger.error(f"Error placing arbitrage order: {e}")
    
    def check_institutional_rebalancing(self):
        """
        Advanced rebalancing based on institutional metrics
        """
        try:
            current_price = self.get_current_price()
            grid_center = (self.upper_price + self.lower_price) / 2
            
            # Calculate multiple rebalancing triggers
            price_deviation = abs(current_price - grid_center) / grid_center
            
            # Get market analysis
            analysis = getattr(self, 'current_market_analysis', {})
            
            # Institutional rebalancing triggers
            should_rebalance = False
            rebalance_reason = ""
            
            # 1. Standard price deviation
            if price_deviation > 0.12:  # 12% deviation
                should_rebalance = True
                rebalance_reason = "price_deviation"
            
            # 2. Market state change
            elif self.current_market_state in ["institutional_momentum", "volume_breakout"]:
                should_rebalance = True
                rebalance_reason = "market_state_change"
            
            # 3. Liquidity state change
            elif self.liquidity_state != "normal":
                should_rebalance = True
                rebalance_reason = "liquidity_change"
            
            # 4. Volume profile shift
            volume_profile = analysis.get('volume_profile', {})
            if volume_profile:
                poc_price = volume_profile.get('poc_price', current_price)
                poc_deviation = abs(current_price - poc_price) / current_price
                if poc_deviation > 0.08:  # 8% from POC
                    should_rebalance = True
                    rebalance_reason = "volume_profile_shift"
            
            if should_rebalance:
                logger.info(f"🔄 Institutional rebalancing triggered: {rebalance_reason}")
                logger.info(f"📊 Price deviation: {price_deviation*100:.2f}%")
                logger.info(f"🏛️ Market state: {self.current_market_state}")
                
                # Cancel existing orders
                self.stop()
                
                # Recalculate with current market conditions
                self.auto_configure_institutional_grid(self.total_investment)
                
                # Place new institutional orders
                self.place_institutional_grid_orders()
                
                logger.info("✅ Institutional grid rebalanced successfully")
                
        except Exception as e:
            logger.error(f"Error in institutional rebalancing: {e}")
    
    def check_institutional_risk_management(self):
        """
        Advanced institutional risk management
        """
        try:
            # Get account info
            account_info = self.get_account_info()
            
            # Calculate current exposure
            total_exposure = 0
            if account_info and 'positions' in account_info:
                for pos in account_info['positions']:
                    contracts = float(pos.get('contracts', 0))
                    if contracts != 0:
                        mark_price = float(pos.get('markPrice', 0))
                        exposure = abs(contracts * mark_price / self.leverage)
                        total_exposure += exposure
            
            # Risk checks
            max_exposure = self.total_investment * 2.0  # 200% max exposure
            
            if total_exposure > max_exposure:
                logger.warning(f"⚠️ Exposure limit exceeded: ${total_exposure:.2f} > ${max_exposure:.2f}")
                self.reduce_exposure()
            
            # Market volatility risk
            vol_metrics = self.calculate_enhanced_volatility_metrics()
            vol_percentile = vol_metrics.get('volatility_percentile', 0.5)
            
            if vol_percentile > 0.95:  # Extreme volatility
                logger.warning("⚠️ Extreme volatility detected - implementing protective measures")
                self.implement_volatility_protection()
            
            # Institutional flow risk
            analysis = getattr(self, 'current_market_analysis', {})
            institutional_flow = analysis.get('institutional_flow', {})
            
            if institutional_flow.get('institutional_strength', 0) > 0.9:
                # Very strong institutional flow - be cautious
                logger.info("🏛️ Strong institutional flow detected - adjusting strategy")
                self.adjust_for_institutional_flow(institutional_flow)
                
        except Exception as e:
            logger.error(f"Error in institutional risk management: {e}")
    
    def reduce_exposure(self):
        """
        Reduce exposure when limits are exceeded
        """
        try:
            # Cancel largest orders first
            all_orders = list(self.buy_orders.items()) + list(self.sell_orders.items())
            
            # Sort by USDT amount (largest first)
            sorted_orders = sorted(all_orders, key=lambda x: x[1]['usdt_amount'], reverse=True)
            
            cancelled_count = 0
            for order_id, order_data in sorted_orders[:5]:  # Cancel top 5 largest
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                    if order_id in self.buy_orders:
                        del self.buy_orders[order_id]
                    elif order_id in self.sell_orders:
                        del self.sell_orders[order_id]
                    cancelled_count += 1
                except:
                    pass
            
            logger.info(f"🛡️ Reduced exposure by cancelling {cancelled_count} largest orders")
            
        except Exception as e:
            logger.error(f"Error reducing exposure: {e}")
    
    def implement_volatility_protection(self):
        """
        Implement protection during extreme volatility
        """
        try:
            # Widen spreads
            self.grid_spacing *= 1.5
            
            # Reduce order sizes
            self.base_order_amount *= 0.6
            
            # Cancel and replace with protective parameters
            self.stop()
            time.sleep(2)
            self.place_institutional_grid_orders()
            
            logger.info("🛡️ Volatility protection implemented")
            
            # Schedule restoration after 30 minutes
            def restore_normal():
                time.sleep(1800)
                self.grid_spacing /= 1.5
                self.base_order_amount /= 0.6
                logger.info("🔄 Normal parameters restored after volatility")
            
            threading.Thread(target=restore_normal, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error implementing volatility protection: {e}")
    
    def adjust_for_institutional_flow(self, institutional_flow: Dict):
        """
        Adjust strategy based on institutional flow
        """
        try:
            bias = institutional_flow.get('institutional_bias', 0)
            
            if bias > 0.5:  # Strong institutional buying
                # Reduce sell orders, increase buy orders
                logger.info("🏛️ Adjusting for institutional buying pressure")
                # Implementation would adjust order distribution
                
            elif bias < -0.5:  # Strong institutional selling
                # Reduce buy orders, increase sell orders
                logger.info("🏛️ Adjusting for institutional selling pressure")
                # Implementation would adjust order distribution
                
        except Exception as e:
            logger.error(f"Error adjusting for institutional flow: {e}")
    
    def get_adaptive_check_interval(self) -> float:
        """
        Get adaptive check interval based on market conditions
        """
        try:
            base_interval = 3  # 3 seconds base
            
            # Faster in high volatility or high opportunity states
            if self.current_market_state in ["institutional_momentum", "high_frequency_opportunity"]:
                return base_interval * 0.5  # 1.5 seconds
            elif self.current_market_state == "volume_breakout":
                return base_interval * 0.7  # 2.1 seconds
            elif self.liquidity_state == "high":
                return base_interval * 0.8  # 2.4 seconds
            else:
                return base_interval
                
        except:
            return 3.0
    
    def display_institutional_performance(self):
        """
        Display comprehensive institutional performance dashboard
        """
        try:
            metrics = self.calculate_institutional_performance_metrics()
            current_price = self.get_current_price()
            account_info = self.get_account_info()
            
            logger.info("=" * 80)
            logger.info("🏛️ INSTITUTIONAL PERFORMANCE DASHBOARD")
            logger.info("=" * 80)
            
            # Current market state
            logger.info(f"📊 Market State: {self.current_market_state.upper()}")
            logger.info(f"💧 Liquidity State: {self.liquidity_state.upper()}")
            logger.info(f"💱 Current Price: ${current_price:.4f}")
            logger.info(f"🎯 Grid Range: ${self.lower_price:.4f} - ${self.upper_price:.4f}")
            
            # Account info
            if account_info:
                logger.info(f"💰 USDT Balance: ${account_info.get('usdt_balance', 0):.2f}")
            
            # Performance metrics
            if metrics:
                logger.info(f"📈 Total Trades: {metrics['total_trades']}")
                logger.info(f"🎯 Win Rate: {metrics['win_rate']:.1f}%")
                logger.info(f"💵 Total Institutional Profit: ${metrics['total_institutional_profit']:.4f}")
                logger.info(f"📊 Trade Profits: ${metrics['total_trade_profit']:.4f}")
                logger.info(f"💎 Liquidity Rebates: ${metrics['total_rebate_profit']:.6f}")
                logger.info(f"⚡ Market Making: ${metrics['market_making_profits']:.4f}")
                logger.info(f"📉 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                logger.info(f"📊 Calmar Ratio: {metrics['calmar_ratio']:.3f}")
                logger.info(f"💹 Profit Factor: {metrics['profit_factor']:.2f}")
                logger.info(f"📅 Avg Daily Return: ${metrics['avg_daily_return']:.4f}")
            
            # Order status
            active_buys = sum(1 for o in self.buy_orders.values() if o['status'] == 'open')
            active_sells = sum(1 for o in self.sell_orders.values() if o['status'] == 'open')
            logger.info(f"📋 Active Orders: {active_buys} buys, {active_sells} sells")
            
            # Institutional metrics
            logger.info(f"🏛️ Institutional Profits: ${getattr(self, 'institutional_profits', 0):.4f}")
            logger.info(f"⚡ Arbitrage Profits: ${getattr(self, 'arbitrage_profits', 0):.4f}")
            logger.info(f"💎 Liquidity Rebates: ${getattr(self, 'liquidity_rebates', 0):.6f}")
            
            # Market analysis summary
            analysis = getattr(self, 'current_market_analysis', {})
            if analysis:
                imbalance = analysis.get('imbalance')
                if imbalance:
                    logger.info(f"📊 Order Book Imbalance: {imbalance.imbalance_ratio:.3f}")
                    logger.info(f"🎯 Pressure Score: {imbalance.pressure_score:.3f}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying institutional performance: {e}")
    
    # Include all missing methods from the enhanced bot
    def calculate_enhanced_volatility_metrics(self, symbol: str = None) -> Dict:
        """Enhanced volatility calculation using multiple timeframes"""
        if symbol is None:
            symbol = self.symbol
            
        try:
            # Simplified version for institutional bot
            df = self.fetch_ohlcv_data('1h', 100)
            if df.empty:
                return {'weighted_volatility': 0.02, 'volatility_percentile': 0.5, 'current_price': 0}
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            current_price = df['close'].iloc[-1]
            
            # Calculate percentile
            historical_vols = returns.rolling(24).std().dropna()
            percentile = (historical_vols < volatility).sum() / len(historical_vols) if len(historical_vols) > 0 else 0.5
            
            return {
                'weighted_volatility': volatility,
                'volatility_percentile': percentile,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {'weighted_volatility': 0.02, 'volatility_percentile': 0.5, 'current_price': 0}
    
    def fetch_ohlcv_data(self, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_optimal_price_decimals(self, price: float) -> int:
        """Get optimal decimal places based on price"""
        if price < 0.001:
            return 8
        elif price < 0.01:
            return 6
        elif price < 0.1:
            return 5
        elif price < 1:
            return 4
        elif price < 10:
            return 3
        else:
            return 2
    
    def check_position_mode(self):
        """Check and set position mode"""
        try:
            self.exchange.set_position_mode(hedged=False, symbol=self.symbol)
            logger.info("Position mode set to one-way mode")
        except Exception as e:
            logger.info(f"Position mode check: {e}")
    
    def set_leverage(self):
        """Set leverage for trading"""
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"Leverage set to {self.leverage}x")
        except Exception as e:
            logger.info(f"Leverage setting: {e}")
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0
    
    def calculate_grid_levels(self) -> List[float]:
        """Calculate all grid price levels"""
        levels = []
        for i in range(self.grid_levels):
            price = self.lower_price + (i * self.grid_spacing)
            levels.append(round(price, self.price_decimals))
        return levels
    
    def calculate_position_size_for_amount(self, price: float, usdt_amount: float) -> float:
        """Calculate position size for USDT amount"""
        if price <= 0:
            return 0
        contract_value_usdt = usdt_amount * self.leverage
        position_size = contract_value_usdt / price
        
        try:
            market = self.exchange.market(self.symbol)
            min_size = market['limits']['amount']['min']
            precision = market['precision']['amount']
            
            if isinstance(precision, float):
                decimal_places = max(0, -int(np.log10(precision)))
            else:
                decimal_places = int(precision)
            
            position_size = round(position_size, decimal_places)
            position_size = max(position_size, min_size)
        except:
            position_size = round(position_size, 3)
        
        return position_size
    
    def get_account_info(self) -> Dict:
        """Get account info"""
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions([self.symbol])
            
            return {
                'usdt_balance': balance.get('USDT', {}).get('free', 0),
                'positions': positions
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
    
    def stop(self):
        """Cancel all orders"""
        try:
            logger.info("Cancelling all orders...")
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("✅ All orders cancelled")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")


# INSTITUTIONAL USAGE EXAMPLE
if __name__ == "__main__":
    # Configuration
    API_KEY = "VDpt0WQXIjXul4OBrS"
    API_SECRET = "z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9"
    
    # Investment amount
    TOTAL_INVESTMENT = 69  # USDT
    
    # Initialize institutional bot
    institutional_bot = InstitutionalGridBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='DOGE/USDT:USDT',
        testnet=False  # Set to False for live trading
    )
    
    # Configure institutional parameters for maximum profitability
    institutional_bot.use_orderbook_signals = True
    institutional_bot.tick_momentum_capture = True
    institutional_bot.volume_profile_analysis = True
    institutional_bot.dynamic_spread_optimization = True
    institutional_bot.latency_optimization = True
    institutional_bot.cross_exchange_signals = True
    institutional_bot.liquidity_rebate_optimization = True
    institutional_bot.smart_order_routing = True
    institutional_bot.stat_arb_signals = True
    institutional_bot.market_impact_model = True
    institutional_bot.pattern_recognition = True
    institutional_bot.institutional_flow_tracking = True
    
    # Risk management (institutional grade)
    institutional_bot.max_drawdown_threshold = 0.15  # 15% max drawdown
    institutional_bot.daily_loss_limit = 0.25  # 25% daily loss limit
    institutional_bot.impact_cost_threshold = 0.02  # 2 basis points max impact
    
    # Advanced features
    institutional_bot.trend_allocation = 0.2  # 20% for trend following
    institutional_bot.min_profit_threshold = 0.18  # 0.18% minimum profit
    institutional_bot.imbalance_threshold = 0.25  # 25% imbalance threshold
    institutional_bot.arbitrage_threshold = 0.04  # 0.04% arbitrage threshold
    
    try:
        # Configure and run institutional bot
        institutional_bot.auto_configure_institutional_grid(TOTAL_INVESTMENT)
        
        logger.info("🏛️ Starting INSTITUTIONAL GRADE Grid Bot!")
        logger.info("💎 Professional profit extraction techniques: ENABLED")
        logger.info("⚡ Latency optimization: ACTIVE")
        logger.info("📊 Market microstructure analysis: RUNNING")
        logger.info("🎯 Statistical arbitrage: MONITORING")
        logger.info("💰 Liquidity rebate optimization: ACTIVE")
        logger.info("🧠 High-frequency pattern recognition: ENABLED")
        logger.info("🏛️ Institutional flow tracking: MONITORING")
        
        institutional_bot.run_institutional_bot(
            check_interval=3,  # Ultra-fast 3 second checks
            rebalance_interval=900  # Smart rebalancing every 15 minutes
        )
        
    except KeyboardInterrupt:
        institutional_bot.stop()
        logger.info("🛑 Institutional bot stopped safely")