import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import logging
import requests
from typing import Dict, List, Tuple, Optional
import ta
from collections import deque
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detects market regime (trending/ranging) and volatility"""
    
    def __init__(self):
        self.atr_period = 14
        self.adx_period = 14
        self.volatility_lookback = 20
        
    def calculate_regime(self, df: pd.DataFrame) -> Dict:
        """Determine market regime and conditions"""
        # Calculate indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], self.atr_period)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], self.adx_period)
        df['rsi'] = ta.momentum.rsi(df['close'], 14)
        
        # Volatility analysis
        returns = df['close'].pct_change()
        current_volatility = returns.tail(self.volatility_lookback).std()
        avg_volatility = returns.std()
        volatility_ratio = current_volatility / avg_volatility
        
        # Trend strength
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        current_price = df['close'].iloc[-1]
        
        trend_strength = 0
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_strength = 1  # Bullish
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend_strength = -1  # Bearish
        
        # Market regime determination
        adx_value = df['adx'].iloc[-1]
        if adx_value > 25:
            regime = 'trending'
        else:
            regime = 'ranging'
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        return {
            'regime': regime,
            'trend_strength': trend_strength,
            'volatility_ratio': volatility_ratio,
            'adx': adx_value,
            'volume_ratio': volume_ratio,
            'rsi': df['rsi'].iloc[-1],
            'atr': df['atr'].iloc[-1]
        }

class LevelIdentifier:
    """Enhanced level identification with quality scoring"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.level_cache = {}
        self.level_performance = {}  # Track how well each level performs
        
    def find_volume_levels(self, df: pd.DataFrame, num_levels: int = 5) -> List[Dict]:
        """Find high volume levels with quality metrics"""
        price_range = df['close'].max() - df['close'].min()
        num_bins = 50
        bins = np.linspace(df['close'].min(), df['close'].max(), num_bins)
        
        volume_profile = []
        for i in range(len(bins) - 1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i + 1])
            filtered_df = df.loc[mask]
            
            if len(filtered_df) > 0:
                vol = filtered_df['volume'].sum()
                # Calculate level quality metrics
                touches = len(filtered_df)
                bounces = 0
                
                # Count bounces (price reversal at this level)
                for idx in filtered_df.index[1:-1]:
                    if (df.loc[idx-1, 'close'] < bins[i] and df.loc[idx+1, 'close'] > bins[i+1]) or \
                       (df.loc[idx-1, 'close'] > bins[i+1] and df.loc[idx+1, 'close'] < bins[i]):
                        bounces += 1
                
                volume_profile.append({
                    'price': (bins[i] + bins[i + 1]) / 2,
                    'volume': vol,
                    'touches': touches,
                    'bounces': bounces,
                    'quality': bounces / max(touches, 1)  # Bounce rate
                })
        
        # Sort by combined score
        volume_df = pd.DataFrame(volume_profile)
        volume_df['score'] = volume_df['volume'] * volume_df['quality']
        volume_df = volume_df.nlargest(num_levels, 'score')
        
        return volume_df.to_dict('records')
    
    def find_liquidity_levels(self, df: pd.DataFrame) -> List[float]:
        """Find levels with liquidity pools (stop loss clusters)"""
        levels = []
        
        # Identify potential stop loss areas
        # Below recent lows (long stop losses)
        recent_lows = df['low'].rolling(10).min()
        stop_zone_longs = recent_lows * 0.995  # 0.5% below lows
        
        # Above recent highs (short stop losses)
        recent_highs = df['high'].rolling(10).max()
        stop_zone_shorts = recent_highs * 1.005  # 0.5% above highs
        
        # Find areas where price wicked through these zones
        for i in range(10, len(df)):
            # Long stop hunts
            if df.loc[i, 'low'] < stop_zone_longs.iloc[i-1] and df.loc[i, 'close'] > stop_zone_longs.iloc[i-1]:
                levels.append(stop_zone_longs.iloc[i-1])
            
            # Short stop hunts
            if df.loc[i, 'high'] > stop_zone_shorts.iloc[i-1] and df.loc[i, 'close'] < stop_zone_shorts.iloc[i-1]:
                levels.append(stop_zone_shorts.iloc[i-1])
        
        return levels
    
    def find_order_block_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Identify order blocks (institutional levels)"""
        order_blocks = []
        
        for i in range(3, len(df) - 3):
            # Bullish order block: Strong move up after consolidation
            if df.loc[i+1, 'close'] > df.loc[i, 'high'] * 1.002:  # 0.2% breakout
                # Check if previous candles were consolidating
                consolidation = True
                for j in range(i-2, i+1):
                    if abs(df.loc[j, 'close'] - df.loc[j, 'open']) / df.loc[j, 'open'] > 0.002:
                        consolidation = False
                        break
                
                if consolidation:
                    order_blocks.append({
                        'price': (df.loc[i, 'high'] + df.loc[i, 'low']) / 2,
                        'type': 'bullish_ob',
                        'strength': df.loc[i, 'volume'] / df['volume'].rolling(20).mean().iloc[i]
                    })
            
            # Bearish order block: Strong move down after consolidation
            elif df.loc[i+1, 'close'] < df.loc[i, 'low'] * 0.998:  # 0.2% breakdown
                consolidation = True
                for j in range(i-2, i+1):
                    if abs(df.loc[j, 'close'] - df.loc[j, 'open']) / df.loc[j, 'open'] > 0.002:
                        consolidation = False
                        break
                
                if consolidation:
                    order_blocks.append({
                        'price': (df.loc[i, 'high'] + df.loc[i, 'low']) / 2,
                        'type': 'bearish_ob',
                        'strength': df.loc[i, 'volume'] / df['volume'].rolling(20).mean().iloc[i]
                    })
        
        return order_blocks
    
    def combine_levels(self, df: pd.DataFrame, market_regime: Dict, merge_threshold: float = 0.001) -> List[Dict]:
        """Combine all levels with regime-adjusted scoring"""
        all_levels = []
        
        # Get levels from different methods
        volume_levels = self.find_volume_levels(df)
        pivot_levels = self.find_pivot_levels(df)
        swing_levels = self.find_swing_levels(df)
        liquidity_levels = self.find_liquidity_levels(df)
        order_blocks = self.find_order_block_levels(df)
        
        # Add volume levels with quality metrics
        for level in volume_levels:
            all_levels.append({
                'price': level['price'],
                'type': 'volume',
                'strength': 3 * level['quality'],  # Weight by quality
                'touches': level['touches'],
                'bounces': level['bounces']
            })
        
        # Add other levels
        for key, level in pivot_levels.items():
            strength = 2
            # Pivots are more important in ranging markets
            if market_regime['regime'] == 'ranging':
                strength *= 1.5
            all_levels.append({'price': level, 'type': f'pivot_{key}', 'strength': strength})
        
        for level in swing_levels:
            all_levels.append({'price': level, 'type': 'swing', 'strength': 2})
        
        for level in liquidity_levels:
            all_levels.append({'price': level, 'type': 'liquidity', 'strength': 2.5})
        
        for ob in order_blocks:
            all_levels.append({
                'price': ob['price'],
                'type': ob['type'],
                'strength': 3 * ob['strength']
            })
        
        # Merge nearby levels and combine strengths
        merged_levels = self._merge_nearby_levels(all_levels, merge_threshold)
        
        # Adjust strength based on market regime
        for level in merged_levels:
            # In trending markets, levels in trend direction are stronger
            if market_regime['regime'] == 'trending':
                current_price = df['close'].iloc[-1]
                if market_regime['trend_strength'] > 0 and level['price'] < current_price:
                    level['strength'] *= 1.3  # Support in uptrend
                elif market_regime['trend_strength'] < 0 and level['price'] > current_price:
                    level['strength'] *= 1.3  # Resistance in downtrend
            
            # Add historical performance if available
            level_key = round(level['price'], 2)
            if level_key in self.level_performance:
                perf = self.level_performance[level_key]
                level['win_rate'] = perf['wins'] / max(perf['total'], 1)
                level['strength'] *= (1 + level['win_rate'])
        
        # Sort by strength
        merged_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return merged_levels
    
    def _merge_nearby_levels(self, levels: List[Dict], threshold: float) -> List[Dict]:
        """Merge nearby levels intelligently"""
        if not levels:
            return []
        
        # Sort by price
        levels.sort(key=lambda x: x['price'])
        
        merged = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] < threshold:
                current_cluster.append(level)
            else:
                # Merge current cluster
                if current_cluster:
                    merged_level = self._merge_cluster(current_cluster)
                    merged.append(merged_level)
                current_cluster = [level]
        
        # Don't forget the last cluster
        if current_cluster:
            merged_level = self._merge_cluster(current_cluster)
            merged.append(merged_level)
        
        return merged
    
    def _merge_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge a cluster of levels into one"""
        # Weighted average price by strength
        total_strength = sum(level['strength'] for level in cluster)
        weighted_price = sum(level['price'] * level['strength'] for level in cluster) / total_strength
        
        # Combine types
        types = []
        for level in cluster:
            if 'types' in level:
                types.extend(level['types'])
            else:
                types.append(level['type'])
        
        # Sum other metrics
        touches = sum(level.get('touches', 0) for level in cluster)
        bounces = sum(level.get('bounces', 0) for level in cluster)
        
        return {
            'price': weighted_price,
            'strength': total_strength,
            'types': list(set(types)),
            'touches': touches,
            'bounces': bounces,
            'cluster_size': len(cluster)
        }
    
    def find_pivot_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points"""
        last_candle = df.iloc[-1]
        high = last_candle['high']
        low = last_candle['low']
        close = last_candle['close']
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def find_swing_levels(self, df: pd.DataFrame, window: int = 10) -> List[float]:
        """Find swing highs and lows"""
        levels = []
        
        for i in range(window, len(df) - window):
            # Swing high
            if df.loc[i, 'high'] == df.loc[i-window:i+window, 'high'].max():
                levels.append(df.loc[i, 'high'])
            
            # Swing low
            if df.loc[i, 'low'] == df.loc[i-window:i+window, 'low'].min():
                levels.append(df.loc[i, 'low'])
        
        return list(set(levels))
    
    def update_level_performance(self, level_price: float, success: bool):
        """Track level performance for adaptive learning"""
        level_key = round(level_price, 2)
        if level_key not in self.level_performance:
            self.level_performance[level_key] = {'wins': 0, 'total': 0}
        
        self.level_performance[level_key]['total'] += 1
        if success:
            self.level_performance[level_key]['wins'] += 1

class MomentumAnalyzer:
    """Analyzes momentum and order flow for better entries"""
    
    def __init__(self):
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Comprehensive momentum analysis"""
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], self.rsi_period)
        current_rsi = df['rsi'].iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume momentum
        df['volume_sma'] = df['volume'].rolling(20).mean()
        volume_momentum = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
        
        # Price momentum
        returns = df['close'].pct_change()
        momentum_1h = returns.tail(4).sum()  # Last 4 periods (1 hour if 15m candles)
        momentum_4h = returns.tail(16).sum()  # Last 16 periods (4 hours if 15m candles)
        
        # Divergence detection
        divergence = self._detect_divergence(df)
        
        # Order flow imbalance
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        order_flow = df['buying_pressure'].tail(5).mean()
        
        return {
            'rsi': current_rsi,
            'rsi_oversold': current_rsi < 30,
            'rsi_overbought': current_rsi > 70,
            'macd_bullish': df['macd_diff'].iloc[-1] > 0,
            'macd_cross': self._detect_macd_cross(df),
            'stoch_oversold': df['stoch_k'].iloc[-1] < 20,
            'stoch_overbought': df['stoch_k'].iloc[-1] > 80,
            'volume_momentum': volume_momentum,
            'price_momentum_1h': momentum_1h,
            'price_momentum_4h': momentum_4h,
            'divergence': divergence,
            'order_flow': order_flow,
            'trend_strength': self._calculate_trend_strength(df)
        }
    
    def _detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Detect price/momentum divergence"""
        # Find recent peaks and troughs
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(len(df) - lookback, len(df) - 2):
            # Price peaks
            if df.loc[i, 'high'] > df.loc[i-1, 'high'] and df.loc[i, 'high'] > df.loc[i+1, 'high']:
                price_highs.append((i, df.loc[i, 'high']))
                rsi_highs.append((i, df.loc[i, 'rsi']))
            
            # Price troughs
            if df.loc[i, 'low'] < df.loc[i-1, 'low'] and df.loc[i, 'low'] < df.loc[i+1, 'low']:
                price_lows.append((i, df.loc[i, 'low']))
                rsi_lows.append((i, df.loc[i, 'rsi']))
        
        # Check for divergence
        bullish_div = False
        bearish_div = False
        
        # Bullish divergence: lower lows in price, higher lows in RSI
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                bullish_div = True
        
        # Bearish divergence: higher highs in price, lower highs in RSI
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                bearish_div = True
        
        return {
            'bullish': bullish_div,
            'bearish': bearish_div
        }
    
    def _detect_macd_cross(self, df: pd.DataFrame) -> str:
        """Detect MACD crossovers"""
        if len(df) < 2:
            return 'none'
        
        curr_diff = df['macd_diff'].iloc[-1]
        prev_diff = df['macd_diff'].iloc[-2]
        
        if prev_diff <= 0 and curr_diff > 0:
            return 'bullish'
        elif prev_diff >= 0 and curr_diff < 0:
            return 'bearish'
        else:
            return 'none'
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX and price action"""
        adx = ta.trend.adx(df['high'], df['low'], df['close'], 14)
        
        # Price position relative to moving averages
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        
        trend_score = 0
        
        # ADX strength
        if adx.iloc[-1] > 25:
            trend_score += 1
        if adx.iloc[-1] > 40:
            trend_score += 1
        
        # MA alignment
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_score += 2  # Bullish alignment
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend_score -= 2  # Bearish alignment
        
        return trend_score

class RiskManager:
    """Enhanced risk management with dynamic sizing"""
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_positions: int = 3):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.open_positions = 0
        self.daily_pnl = 0
        self.daily_trades = 0
        self.win_rate_history = deque(maxlen=20)  # Track last 20 trades
        
    def calculate_position_size(self, balance: float, entry_price: float, 
                               stop_loss_price: float, leverage: int = 1,
                               market_conditions: Dict = None) -> float:
        """Dynamic position sizing based on conditions"""
        # Base risk calculation
        risk_amount = balance * self.max_risk_per_trade
        
        # Adjust risk based on market conditions
        if market_conditions:
            # Reduce risk in high volatility
            if market_conditions.get('volatility_ratio', 1) > 1.5:
                risk_amount *= 0.7
            
            # Reduce risk if trend is against us
            if market_conditions.get('trend_strength', 0) < -1:
                risk_amount *= 0.8
            
            # Increase risk in optimal conditions
            if market_conditions.get('regime') == 'ranging' and \
               market_conditions.get('volatility_ratio', 1) < 0.8:
                risk_amount *= 1.2
        
        # Adjust based on recent performance
        if len(self.win_rate_history) >= 10:
            recent_win_rate = sum(self.win_rate_history) / len(self.win_rate_history)
            if recent_win_rate > 0.6:
                risk_amount *= 1.1
            elif recent_win_rate < 0.4:
                risk_amount *= 0.8
        
        # Calculate position size
        price_difference = abs(entry_price - stop_loss_price)
        risk_per_unit = price_difference / entry_price
        
        position_size = risk_amount / risk_per_unit / entry_price
        
        # Apply leverage
        position_size *= leverage
        
        # Apply daily loss limit
        if self.daily_pnl < -balance * 0.02:  # 2% daily loss limit
            position_size *= 0.5
        
        return position_size
    
    def can_open_position(self, balance: float) -> bool:
        """Check if we can open a new position"""
        # Check position limit
        if self.open_positions >= self.max_positions:
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -balance * 0.03:  # 3% daily loss limit
            return False
        
        # Check daily trade limit
        if self.daily_trades >= 10:  # Maximum 10 trades per day
            return False
        
        return True
    
    def update_trade_result(self, profit: float, success: bool):
        """Update risk parameters based on trade results"""
        self.daily_pnl += profit
        self.daily_trades += 1
        self.win_rate_history.append(1 if success else 0)
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0
        self.daily_trades = 0

class LevelTradingBot:
    """Enhanced trading bot with advanced features"""
    
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'BTC/USDT:USDT',
                 leverage: int = 5, testnet: bool = True):
        self.symbol = symbol
        self.leverage = leverage
        
        # Initialize exchange
        exchange_class = ccxt.bybit
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': testnet
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Initialize components
        self.level_identifier = LevelIdentifier()
        self.risk_manager = RiskManager()
        self.market_regime_detector = MarketRegimeDetector()
        self.momentum_analyzer = MomentumAnalyzer()
        
        # Enhanced trading parameters
        self.min_distance_to_level = 0.0005
        self.max_distance_to_level = 0.002  # Don't trade if too far from level
        self.take_profit_ratio = 2.0
        self.trailing_stop_activation = 0.01
        self.trailing_stop_distance = 0.005
        self.partial_tp_ratio = 0.5  # Take 50% profit at first target
        self.partial_tp_target = 1.0  # First target at 1:1 RR
        
        # State tracking
        self.current_levels = []
        self.positions = {}
        self.last_update_time = None
        self.market_conditions = {}
        self.trade_history = deque(maxlen=100)
        
        # Performance tracking
        self.session_start_balance = None
        self.session_trades = 0
        self.session_wins = 0
        
    def update_levels(self, timeframe: str = '15m', limit: int = 200):
        """Update levels with multiple timeframe analysis"""
        try:
            # Fetch data from multiple timeframes
            timeframes = ['15m', '1h', '4h']
            all_levels = []
            
            for tf in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Get market regime for this timeframe
                regime = self.market_regime_detector.calculate_regime(df)
                
                # Weight levels by timeframe importance
                weight = {'15m': 1, '1h': 1.5, '4h': 2}[tf]
                
                # Get levels for this timeframe
                levels = self.level_identifier.combine_levels(df, regime)
                
                # Apply timeframe weight
                for level in levels:
                    level['strength'] *= weight
                    level['timeframe'] = tf
                
                all_levels.extend(levels)
            
            # Merge levels from all timeframes
            self.current_levels = self.level_identifier._merge_nearby_levels(all_levels, 0.002)
            
            # Store market conditions from lowest timeframe
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '15m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.market_conditions = self.market_regime_detector.calculate_regime(df)
            
            self.last_update_time = datetime.now()
            
            # Log level analysis
            self._log_level_analysis()
            
            logger.info(f"Updated levels: {len(self.current_levels)} levels identified")
            logger.info(f"Market regime: {self.market_conditions['regime']}, ADX: {self.market_conditions['adx']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating levels: {e}")
    
    def check_entry_conditions(self, current_price: float) -> Optional[Dict]:
        """Enhanced entry checking with multiple confirmations"""
        # Find nearest levels
        nearest = self.find_nearest_levels(current_price)
        
        # Get momentum analysis
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, '15m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        momentum = self.momentum_analyzer.analyze_momentum(df)
        
        entry_signal = None
        
        # Check support levels
        if nearest['support']:
            distance_to_support = (current_price - nearest['support']['price']) / current_price
            
            if self.min_distance_to_level < distance_to_support < self.max_distance_to_level:
                # Calculate entry score
                score = self._calculate_entry_score('buy', nearest['support'], momentum)
                
                if score > 0.6:  # Minimum score threshold
                    entry_signal = {
                        'side': 'buy',
                        'level': nearest['support'],
                        'entry_price': current_price,
                        'stop_loss': nearest['support']['price'] * (1 - self._calculate_stop_distance(nearest['support'], momentum)),
                        'score': score,
                        'momentum': momentum,
                        'reason': self._generate_entry_reason('buy', nearest['support'], momentum)
                    }
        
        # Check resistance levels
        if nearest['resistance'] and not entry_signal:
            distance_to_resistance = (nearest['resistance']['price'] - current_price) / current_price
            
            if self.min_distance_to_level < distance_to_resistance < self.max_distance_to_level:
                # Calculate entry score
                score = self._calculate_entry_score('sell', nearest['resistance'], momentum)
                
                if score > 0.6:  # Minimum score threshold
                    entry_signal = {
                        'side': 'sell',
                        'level': nearest['resistance'],
                        'entry_price': current_price,
                        'stop_loss': nearest['resistance']['price'] * (1 + self._calculate_stop_distance(nearest['resistance'], momentum)),
                        'score': score,
                        'momentum': momentum,
                        'reason': self._generate_entry_reason('sell', nearest['resistance'], momentum)
                    }
        
        # Add take profit levels if signal found
        if entry_signal:
            self._add_take_profit_levels(entry_signal, nearest)
        
        return entry_signal
    
    def _calculate_entry_score(self, side: str, level: Dict, momentum: Dict) -> float:
        """Calculate comprehensive entry score"""
        score = 0.0
        
        # Level strength (0-0.3)
        score += min(level['strength'] / 10, 0.3)
        
        # Level quality metrics (0-0.2)
        if 'bounces' in level and 'touches' in level and level['touches'] > 0:
            quality = level['bounces'] / level['touches']
            score += quality * 0.2
        
        # Momentum confirmation (0-0.3)
        if side == 'buy':
            if momentum['rsi_oversold']:
                score += 0.1
            if momentum['divergence']['bullish']:
                score += 0.1
            if momentum['macd_cross'] == 'bullish':
                score += 0.1
            if momentum['order_flow'] > 0.6:
                score += 0.05
        else:  # sell
            if momentum['rsi_overbought']:
                score += 0.1
            if momentum['divergence']['bearish']:
                score += 0.1
            if momentum['macd_cross'] == 'bearish':
                score += 0.1
            if momentum['order_flow'] < 0.4:
                score += 0.05
        
        # Market regime bonus (0-0.2)
        if self.market_conditions['regime'] == 'ranging':
            score += 0.1  # Better for level trading
            if self.market_conditions['volatility_ratio'] < 1.0:
                score += 0.1  # Low volatility is good
        else:
            # In trending market, only trade with trend
            if (side == 'buy' and self.market_conditions['trend_strength'] > 0) or \
               (side == 'sell' and self.market_conditions['trend_strength'] < 0):
                score += 0.1
        
        # Volume confirmation (0-0.1)
        if momentum['volume_momentum'] > 1.2:
            score += 0.05
        if self.market_conditions['volume_ratio'] > 1.5:
            score += 0.05
        
        # Historical performance bonus
        if 'win_rate' in level and level['win_rate'] > 0.6:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_stop_distance(self, level: Dict, momentum: Dict) -> float:
        """Dynamic stop loss distance based on conditions"""
        base_distance = 0.005  # 0.5% base
        
        # Adjust for volatility
        atr_multiplier = self.market_conditions.get('atr', 0) / 100
        base_distance += atr_multiplier * 0.002
        
        # Tighter stops for stronger levels
        if level['strength'] > 5:
            base_distance *= 0.8
        
        # Wider stops in trending markets
        if self.market_conditions['regime'] == 'trending':
            base_distance *= 1.2
        
        return min(base_distance, 0.01)  # Max 1% stop
    
    def _add_take_profit_levels(self, signal: Dict, nearest_levels: Dict):
        """Add multiple take profit targets"""
        risk = abs(signal['entry_price'] - signal['stop_loss'])
        
        if signal['side'] == 'buy':
            # First TP at 1:1 RR
            signal['tp1'] = signal['entry_price'] + risk
            
            # Second TP at configured ratio or next resistance
            signal['tp2'] = signal['entry_price'] + (risk * self.take_profit_ratio)
            
            # If there's a resistance level between TP1 and TP2, use it
            if nearest_levels['resistance']:
                resistance_price = nearest_levels['resistance']['price']
                if signal['tp1'] < resistance_price < signal['tp2']:
                    signal['tp2'] = resistance_price * 0.998  # Just below resistance
        else:
            # First TP at 1:1 RR
            signal['tp1'] = signal['entry_price'] - risk
            
            # Second TP at configured ratio or next support
            signal['tp2'] = signal['entry_price'] - (risk * self.take_profit_ratio)
            
            # If there's a support level between TP1 and TP2, use it
            if nearest_levels['support']:
                support_price = nearest_levels['support']['price']
                if signal['tp2'] < support_price < signal['tp1']:
                    signal['tp2'] = support_price * 1.002  # Just above support
    
    def _generate_entry_reason(self, side: str, level: Dict, momentum: Dict) -> str:
        """Generate detailed entry reason"""
        reasons = []
        
        # Level info
        reasons.append(f"{side.upper()} at {level['price']:.2f} ({', '.join(level['types'])})")
        reasons.append(f"Level strength: {level['strength']:.1f}")
        
        # Momentum info
        if momentum['divergence']['bullish' if side == 'buy' else 'bearish']:
            reasons.append(f"{side.capitalize()} divergence detected")
        
        if momentum['macd_cross'] != 'none':
            reasons.append(f"MACD {momentum['macd_cross']} cross")
        
        # Market regime
        reasons.append(f"Market: {self.market_conditions['regime']}")
        
        return " | ".join(reasons)
    
    def execute_trade(self, signal: Dict):
        """Execute trade with enhanced order management"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            
            # Calculate position size with dynamic sizing
            position_size = self.risk_manager.calculate_position_size(
                usdt_balance,
                signal['entry_price'],
                signal['stop_loss'],
                self.leverage,
                self.market_conditions
            )
            
            # Set leverage
            self.exchange.set_leverage(self.leverage, self.symbol)
            
            # Place market order
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=signal['side'],
                amount=position_size
            )
            
            # Place stop loss order
            stop_side = 'sell' if signal['side'] == 'buy' else 'buy'
            stop_order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop',
                side=stop_side,
                amount=position_size,
                stopPrice=signal['stop_loss'],
                params={'reduce_only': True}
            )
            
            # Place partial take profit at TP1
            tp1_amount = position_size * self.partial_tp_ratio
            tp1_order = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=stop_side,
                amount=tp1_amount,
                price=signal['tp1'],
                params={'reduce_only': True}
            )
            
            # Place remaining take profit at TP2
            tp2_amount = position_size * (1 - self.partial_tp_ratio)
            tp2_order = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=stop_side,
                amount=tp2_amount,
                price=signal['tp2'],
                params={'reduce_only': True}
            )
            
            # Store position info
            position_id = order['id']
            self.positions[position_id] = {
                'entry_order': order,
                'stop_order': stop_order,
                'tp1_order': tp1_order,
                'tp2_order': tp2_order,
                'signal': signal,
                'status': 'open',
                'entry_time': datetime.now(),
                'partial_closed': False
            }
            
            self.risk_manager.open_positions += 1
            self.risk_manager.daily_trades += 1
            self.session_trades += 1
            
            logger.info(f"=== TRADE EXECUTED ===")
            logger.info(f"Side: {signal['side'].upper()}")
            logger.info(f"Entry: {signal['entry_price']:.2f}")
            logger.info(f"Stop Loss: {signal['stop_loss']:.2f}")
            logger.info(f"TP1 (50%): {signal['tp1']:.2f}")
            logger.info(f"TP2 (50%): {signal['tp2']:.2f}")
            logger.info(f"Score: {signal['score']:.2f}")
            logger.info(f"Reason: {signal['reason']}")
            logger.info(f"====================")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def update_trailing_stops(self):
        """Enhanced trailing stop management"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            
            for position in positions:
                if position['contracts'] > 0:
                    position_id = position['id']
                    if position_id in self.positions:
                        stored_position = self.positions[position_id]
                        entry_price = stored_position['signal']['entry_price']
                        current_price = position['markPrice']
                        side = stored_position['signal']['side']
                        
                        # Calculate profit percentage
                        if side == 'buy':
                            profit_pct = (current_price - entry_price) / entry_price
                            
                            # If TP1 hit, move stop to breakeven
                            if current_price >= stored_position['signal']['tp1'] and not stored_position.get('breakeven_set'):
                                new_stop = entry_price * 1.001  # Slight profit
                                self._update_stop_order(stored_position, new_stop, position['contracts'])
                                stored_position['breakeven_set'] = True
                                logger.info("Stop moved to breakeven")
                            
                            # Trail stop if profit exceeds activation
                            elif profit_pct > self.trailing_stop_activation:
                                new_stop = current_price * (1 - self.trailing_stop_distance)
                                current_stop = stored_position['stop_order']['stopPrice']
                                
                                if new_stop > current_stop:
                                    self._update_stop_order(stored_position, new_stop, position['contracts'])
                                    
                        else:  # sell
                            profit_pct = (entry_price - current_price) / entry_price
                            
                            # If TP1 hit, move stop to breakeven
                            if current_price <= stored_position['signal']['tp1'] and not stored_position.get('breakeven_set'):
                                new_stop = entry_price * 0.999  # Slight profit
                                self._update_stop_order(stored_position, new_stop, position['contracts'])
                                stored_position['breakeven_set'] = True
                                logger.info("Stop moved to breakeven")
                            
                            # Trail stop if profit exceeds activation
                            elif profit_pct > self.trailing_stop_activation:
                                new_stop = current_price * (1 + self.trailing_stop_distance)
                                current_stop = stored_position['stop_order']['stopPrice']
                                
                                if new_stop < current_stop:
                                    self._update_stop_order(stored_position, new_stop, position['contracts'])
                                    
        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")
    
    def _update_stop_order(self, position_data: Dict, new_stop: float, amount: float):
        """Update stop loss order"""
        try:
            # Cancel old stop order
            self.exchange.cancel_order(position_data['stop_order']['id'], self.symbol)
            
            # Place new stop order
            side = position_data['signal']['side']
            stop_side = 'sell' if side == 'buy' else 'buy'
            
            new_stop_order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop',
                side=stop_side,
                amount=amount,
                stopPrice=new_stop,
                params={'reduce_only': True}
            )
            
            position_data['stop_order'] = new_stop_order
            logger.info(f"Updated stop to {new_stop:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating stop order: {e}")
    
    def check_closed_positions(self):
        """Enhanced position tracking with performance analysis"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            open_position_ids = [p['id'] for p in positions if p['contracts'] > 0]
            
            for position_id, position_data in list(self.positions.items()):
                if position_data['status'] == 'open' and position_id not in open_position_ids:
                    # Position has been closed
                    position_data['status'] = 'closed'
                    position_data['close_time'] = datetime.now()
                    
                    # Calculate PnL
                    trades = self.exchange.fetch_my_trades(self.symbol, limit=50)
                    position_trades = [t for t in trades if t['order'] == position_id]
                    
                    if position_trades:
                        entry_trade = position_trades[0]
                        exit_trades = position_trades[1:]
                        
                        if exit_trades:
                            # Calculate profit
                            entry_price = entry_trade['price']
                            exit_price = sum(t['price'] * t['amount'] for t in exit_trades) / sum(t['amount'] for t in exit_trades)
                            
                            if position_data['signal']['side'] == 'buy':
                                profit_pct = (exit_price - entry_price) / entry_price
                            else:
                                profit_pct = (entry_price - exit_price) / entry_price
                            
                            profit_usd = profit_pct * entry_trade['cost']
                            
                            # Update statistics
                            success = profit_pct > 0
                            self.risk_manager.update_trade_result(profit_usd, success)
                            
                            if success:
                                self.session_wins += 1
                            
                            # Update level performance
                            self.level_identifier.update_level_performance(
                                position_data['signal']['level']['price'],
                                success
                            )
                            
                            # Log trade result
                            logger.info(f"=== TRADE CLOSED ===")
                            logger.info(f"Result: {'WIN' if success else 'LOSS'}")
                            logger.info(f"Profit: {profit_pct*100:.2f}% (${profit_usd:.2f})")
                            logger.info(f"Session: {self.session_wins}/{self.session_trades} "
                                      f"({self.session_wins/max(self.session_trades,1)*100:.1f}% win rate)")
                            logger.info(f"==================")
                    
                    self.risk_manager.open_positions -= 1
                    
        except Exception as e:
            logger.error(f"Error checking closed positions: {e}")
    
    def find_nearest_levels(self, current_price: float) -> Dict[str, Optional[Dict]]:
        """Find nearest support and resistance levels"""
        support = None
        resistance = None
        
        for level in self.current_levels:
            if level['price'] < current_price * 0.998:  # At least 0.2% below
                if support is None or level['price'] > support['price']:
                    support = level
            elif level['price'] > current_price * 1.002:  # At least 0.2% above
                if resistance is None or level['price'] < resistance['price']:
                    resistance = level
        
        return {'support': support, 'resistance': resistance}
    
    def _log_level_analysis(self):
        """Log detailed analysis of current levels"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            logger.info("="*60)
            logger.info("📊 POTENTIAL ENTRY LEVELS ANALYSIS")
            logger.info("="*60)
            logger.info(f"Current Price: ${current_price:.2f}")
            logger.info(f"Market: {self.market_conditions['regime'].upper()} "
                       f"(ADX: {self.market_conditions['adx']:.1f}, "
                       f"Vol Ratio: {self.market_conditions['volatility_ratio']:.2f})")
            logger.info("-"*60)
            
            # Separate and sort levels
            support_levels = [l for l in self.current_levels if l['price'] < current_price]
            resistance_levels = [l for l in self.current_levels if l['price'] > current_price]
            
            # Sort by distance from current price
            support_levels.sort(key=lambda x: x['price'], reverse=True)  # Nearest first
            resistance_levels.sort(key=lambda x: x['price'])  # Nearest first
            
            # Log resistance levels
            logger.info("🔴 RESISTANCE LEVELS (Potential SHORT entries):")
            logger.info("-"*60)
            for i, level in enumerate(resistance_levels[:5]):  # Top 5 nearest
                distance_pct = ((level['price'] - current_price) / current_price) * 100
                entry_score = self._calculate_potential_score('sell', level)
                
                logger.info(f"{i+1}. Price: ${level['price']:.2f} | "
                           f"Distance: {distance_pct:.2f}% | "
                           f"Strength: {level['strength']:.1f} | "
                           f"Score: {entry_score:.2f}")
                logger.info(f"   Types: {', '.join(level['types'])}")
                
                if 'win_rate' in level:
                    logger.info(f"   Historical: {level['win_rate']*100:.1f}% win rate")
                
                if distance_pct <= self.max_distance_to_level * 100:
                    logger.info(f"   ⚡ WITHIN TRADING RANGE - Monitoring for entry!")
                
                logger.info("")
            
            # Log support levels
            logger.info("🟢 SUPPORT LEVELS (Potential LONG entries):")
            logger.info("-"*60)
            for i, level in enumerate(support_levels[:5]):  # Top 5 nearest
                distance_pct = ((current_price - level['price']) / current_price) * 100
                entry_score = self._calculate_potential_score('buy', level)
                
                logger.info(f"{i+1}. Price: ${level['price']:.2f} | "
                           f"Distance: {distance_pct:.2f}% | "
                           f"Strength: {level['strength']:.1f} | "
                           f"Score: {entry_score:.2f}")
                logger.info(f"   Types: {', '.join(level['types'])}")
                
                if 'win_rate' in level:
                    logger.info(f"   Historical: {level['win_rate']*100:.1f}% win rate")
                
                if distance_pct <= self.max_distance_to_level * 100:
                    logger.info(f"   ⚡ WITHIN TRADING RANGE - Monitoring for entry!")
                
                logger.info("")
            
            # Summary of tradeable levels
            tradeable_supports = [l for l in support_levels 
                                if ((current_price - l['price']) / current_price) <= self.max_distance_to_level]
            tradeable_resistances = [l for l in resistance_levels 
                                   if ((l['price'] - current_price) / current_price) <= self.max_distance_to_level]
            
            logger.info("📌 SUMMARY:")
            logger.info(f"   Tradeable Support Levels: {len(tradeable_supports)}")
            logger.info(f"   Tradeable Resistance Levels: {len(tradeable_resistances)}")
            logger.info(f"   Strongest Level: ${max(self.current_levels, key=lambda x: x['strength'])['price']:.2f}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error logging level analysis: {e}")
    
    def _calculate_potential_score(self, side: str, level: Dict) -> float:
        """Calculate potential entry score for a level"""
        try:
            # Get current momentum
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '15m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            momentum = self.momentum_analyzer.analyze_momentum(df)
            
            # Calculate score
            return self._calculate_entry_score(side, level, momentum)
        except:
            return 0.0
    
    def log_performance(self):
        """Log detailed performance metrics"""
        try:
            balance = self.exchange.fetch_balance()
            current_balance = balance['USDT']['free']
            
            if self.session_start_balance:
                session_pnl = current_balance - self.session_start_balance
                session_pnl_pct = (session_pnl / self.session_start_balance) * 100
                
                logger.info(f"=== PERFORMANCE UPDATE ===")
                logger.info(f"Session P&L: ${session_pnl:.2f} ({session_pnl_pct:.2f}%)")
                logger.info(f"Win Rate: {self.session_wins}/{self.session_trades} "
                          f"({self.session_wins/max(self.session_trades,1)*100:.1f}%)")
                logger.info(f"Daily P&L: ${self.risk_manager.daily_pnl:.2f}")
                logger.info(f"Open Positions: {self.risk_manager.open_positions}")
                logger.info(f"======================")
                
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def log_live_levels(self):
        """Log current tradeable levels in real-time"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Find immediately tradeable levels
            immediate_levels = []
            
            for level in self.current_levels:
                if level['price'] < current_price:  # Support
                    distance = (current_price - level['price']) / current_price
                    if self.min_distance_to_level < distance < self.max_distance_to_level:
                        score = self._calculate_potential_score('buy', level)
                        immediate_levels.append({
                            'level': level,
                            'side': 'buy',
                            'distance_pct': distance * 100,
                            'score': score
                        })
                else:  # Resistance
                    distance = (level['price'] - current_price) / current_price
                    if self.min_distance_to_level < distance < self.max_distance_to_level:
                        score = self._calculate_potential_score('sell', level)
                        immediate_levels.append({
                            'level': level,
                            'side': 'sell',
                            'distance_pct': distance * 100,
                            'score': score
                        })
            
            if immediate_levels:
                logger.info("🎯 TRADEABLE LEVELS RIGHT NOW:")
                for item in sorted(immediate_levels, key=lambda x: x['score'], reverse=True):
                    emoji = "🟢" if item['side'] == 'buy' else "🔴"
                    logger.info(f"{emoji} {item['side'].upper()} @ ${item['level']['price']:.2f} "
                               f"(Distance: {item['distance_pct']:.2f}%, Score: {item['score']:.2f})")
            
        except Exception as e:
            logger.error(f"Error logging live levels: {e}")
    
    def _log_entry_analysis(self, current_price: float):
        """Log why we're not entering a position"""
        nearest = self.find_nearest_levels(current_price)
        
        # Only log periodically to avoid spam
        if not hasattr(self, '_last_entry_log') or time.time() - self._last_entry_log > 60:
            self._last_entry_log = time.time()
            
            reasons = []
            
            if nearest['support']:
                distance = ((current_price - nearest['support']['price']) / current_price) * 100
                if distance > self.max_distance_to_level * 100:
                    reasons.append(f"Support too far: {distance:.2f}% away")
                elif distance < self.min_distance_to_level * 100:
                    reasons.append(f"Too close to support: {distance:.2f}%")
                else:
                    score = self._calculate_potential_score('buy', nearest['support'])
                    if score < 0.6:
                        reasons.append(f"Support score too low: {score:.2f}")
            
            if nearest['resistance']:
                distance = ((nearest['resistance']['price'] - current_price) / current_price) * 100
                if distance > self.max_distance_to_level * 100:
                    reasons.append(f"Resistance too far: {distance:.2f}% away")
                elif distance < self.min_distance_to_level * 100:
                    reasons.append(f"Too close to resistance: {distance:.2f}%")
                else:
                    score = self._calculate_potential_score('sell', nearest['resistance'])
                    if score < 0.6:
                        reasons.append(f"Resistance score too low: {score:.2f}")
            
            if reasons:
                logger.info(f"⏳ Waiting for entry: {' | '.join(reasons)}")
    
    def run(self, update_interval: int = 300, check_interval: int = 10):
        """Main bot loop with enhanced monitoring"""
        logger.info("Starting Enhanced Level-to-Level Trading Bot")
        
        # Get initial balance
        try:
            balance = self.exchange.fetch_balance()
            self.session_start_balance = balance['USDT']['free']
            logger.info(f"Starting balance: ${self.session_start_balance:.2f}")
        except:
            pass
        
        # Initial level update
        self.update_levels()
        
        last_level_update = time.time()
        last_performance_log = time.time()
        last_daily_reset = datetime.now().date()
        last_live_level_log = time.time()
        
        while True:
            try:
                # Reset daily stats at midnight
                if datetime.now().date() > last_daily_reset:
                    self.risk_manager.reset_daily_stats()
                    last_daily_reset = datetime.now().date()
                    logger.info("Daily statistics reset")
                
                # Update levels periodically
                if time.time() - last_level_update > update_interval:
                    self.update_levels()
                    last_level_update = time.time()
                
                # Get current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                # Log live tradeable levels every 30 seconds
                if time.time() - last_live_level_log > 30:
                    self.log_live_levels()
                    last_live_level_log = time.time()
                
                # Check for entry signals
                if self.risk_manager.can_open_position(self.session_start_balance):
                    signal = self.check_entry_conditions(current_price)
                    if signal:
                        self.execute_trade(signal)
                    else:
                        # Log why we're not entering
                        self._log_entry_analysis(current_price)
                
                # Update trailing stops
                self.update_trailing_stops()
                
                # Check closed positions
                self.check_closed_positions()
                
                # Log performance periodically
                if time.time() - last_performance_log > 600:  # Every 10 minutes
                    self.log_performance()
                    last_performance_log = time.time()
                
                # Quick status
                logger.debug(f"Price: {current_price:.2f} | Positions: {self.risk_manager.open_positions} | "
                           f"Daily trades: {self.risk_manager.daily_trades}")
                
                # Sleep before next check
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.log_performance()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
    
    def _log_entry_analysis(self, current_price: float):
        """Log why we're not entering a position"""
        nearest = self.find_nearest_levels(current_price)
        
        # Only log periodically to avoid spam
        if not hasattr(self, '_last_entry_log') or time.time() - self._last_entry_log > 60:
            self._last_entry_log = time.time()
            
            reasons = []
            
            if nearest['support']:
                distance = ((current_price - nearest['support']['price']) / current_price) * 100
                if distance > self.max_distance_to_level * 100:
                    reasons.append(f"Support too far: {distance:.2f}% away")
                elif distance < self.min_distance_to_level * 100:
                    reasons.append(f"Too close to support: {distance:.2f}%")
                else:
                    score = self._calculate_potential_score('buy', nearest['support'])
                    if score < 0.6:
                        reasons.append(f"Support score too low: {score:.2f}")
            
            if nearest['resistance']:
                distance = ((nearest['resistance']['price'] - current_price) / current_price) * 100
                if distance > self.max_distance_to_level * 100:
                    reasons.append(f"Resistance too far: {distance:.2f}% away")
                elif distance < self.min_distance_to_level * 100:
                    reasons.append(f"Too close to resistance: {distance:.2f}%")
                else:
                    score = self._calculate_potential_score('sell', nearest['resistance'])
                    if score < 0.6:
                        reasons.append(f"Resistance score too low: {score:.2f}")
            
            if reasons:
                logger.info(f"⏳ Waiting for entry: {' | '.join(reasons)}")
    
    def log_live_levels(self):
        """Log current tradeable levels in real-time"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Find immediately tradeable levels
            immediate_levels = []
            
            for level in self.current_levels:
                if level['price'] < current_price:  # Support
                    distance = (current_price - level['price']) / current_price
                    if self.min_distance_to_level < distance < self.max_distance_to_level:
                        score = self._calculate_potential_score('buy', level)
                        immediate_levels.append({
                            'level': level,
                            'side': 'buy',
                            'distance_pct': distance * 100,
                            'score': score
                        })
                else:  # Resistance
                    distance = (level['price'] - current_price) / current_price
                    if self.min_distance_to_level < distance < self.max_distance_to_level:
                        score = self._calculate_potential_score('sell', level)
                        immediate_levels.append({
                            'level': level,
                            'side': 'sell',
                            'distance_pct': distance * 100,
                            'score': score
                        })
            
            if immediate_levels:
                logger.info("🎯 TRADEABLE LEVELS RIGHT NOW:")
                for item in sorted(immediate_levels, key=lambda x: x['score'], reverse=True):
                    emoji = "🟢" if item['side'] == 'buy' else "🔴"
                    logger.info(f"{emoji} {item['side'].upper()} @ ${item['level']['price']:.2f} "
                               f"(Distance: {item['distance_pct']:.2f}%, Score: {item['score']:.2f})")
            
        except Exception as e:
            logger.error(f"Error logging live levels: {e}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'api_key': 'VDpt0WQXIjXul4OBrS',
        'api_secret': 'z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9',
        'symbol': 'SOL/USDT:USDT',  # Futures perpetual
        'leverage': 39,
        'testnet': False  # Use testnet for testing
    }
    
    # Create and run bot
    bot = LevelTradingBot(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        symbol=config['symbol'],
        leverage=config['leverage'],
        testnet=config['testnet']
    )
    
    # Run the bot
    bot.run(update_interval=300, check_interval=10)