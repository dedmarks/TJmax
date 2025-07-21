import ccxt
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import asyncio
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
import heapq
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingOpportunity:
    """Trading opportunity data structure"""
    strategy: str
    action: str  # 'buy', 'sell', 'close'
    confidence: float
    expected_profit: float
    risk_score: float
    entry_price: float
    exit_price: float
    stop_loss: float
    position_size: float
    timeframe: str
    metadata: Dict

@dataclass
class MarketRegime:
    """Market regime classification"""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: str  # 'low', 'medium', 'high', 'extreme'
    volume: str  # 'low', 'normal', 'high'
    momentum: float
    mean_reversion_strength: float
    breakout_probability: float

class MultiStrategyAdaptiveBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'DOGE/USDT:USDT', testnet: bool = True):
        """
        Multi-Strategy Adaptive Trading Bot
        Combines multiple profitable strategies and adapts to market conditions
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
        
        # MULTI-STRATEGY COMPONENTS
        
        # 1. MOMENTUM BREAKOUT STRATEGY
        self.momentum_strategy = True
        self.momentum_threshold = 2.0  # Standard deviations
        self.momentum_lookback = 20
        self.momentum_profit_target = 0.015  # 1.5%
        self.momentum_stop_loss = 0.008  # 0.8%
        
        # 2. MEAN REVERSION STRATEGY
        self.mean_reversion_strategy = True
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.mean_reversion_profit = 0.012  # 1.2%
        
        # 3. SCALPING STRATEGY (High Frequency)
        self.scalping_strategy = True
        self.scalp_profit_target = 0.003  # 0.3%
        self.scalp_stop_loss = 0.002  # 0.2%
        self.scalp_max_duration = 300  # 5 minutes max hold
        self.scalp_volume_threshold = 1.5  # Volume spike multiplier
        
        # 4. ARBITRAGE STRATEGY
        self.arbitrage_strategy = True
        self.funding_rate_threshold = 0.01  # 1% funding rate
        self.spot_futures_threshold = 0.001  # 0.1% price difference
        
        # 5. NEWS/SENTIMENT STRATEGY
        self.sentiment_strategy = True
        self.social_sentiment_weight = 0.3
        self.news_impact_multiplier = 1.5
        
        # 6. MACHINE LEARNING PREDICTIONS
        self.ml_strategy = True
        self.ml_model = None
        self.feature_scaler = StandardScaler()
        self.ml_retrain_interval = 3600  # Retrain every hour
        self.ml_confidence_threshold = 0.7
        
        # 7. DELTA NEUTRAL STRATEGY
        self.delta_neutral_strategy = True
        self.hedge_ratio = 0.8
        self.rebalance_threshold = 0.1
        
        # ADAPTIVE POSITION SIZING
        self.kelly_criterion = True
        self.max_position_size = 0.15  # 15% of capital per trade
        self.volatility_adjustment = True
        self.correlation_adjustment = True
        
        # RISK MANAGEMENT
        self.max_concurrent_trades = 8
        self.max_daily_trades = 50
        self.max_drawdown = 0.12  # 12%
        self.profit_lock_threshold = 0.05  # Lock profits at 5%
        
        # Portfolio tracking
        self.active_positions = {}
        self.trade_history = []
        self.daily_pnl = defaultdict(float)
        self.strategy_performance = defaultdict(lambda: {'trades': 0, 'profit': 0, 'win_rate': 0})
        
        # Market data cache
        self.market_data_cache = {}
        self.last_data_update = 0
        self.data_update_interval = 10  # 10 seconds
        
        # Capital allocation
        self.total_capital = 0
        self.available_capital = 0
        self.allocated_capital = defaultdict(float)
        
        # Performance metrics
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
        # Market regime tracking
        self.current_regime = None
        self.regime_history = deque(maxlen=100)
        
        self.check_position_mode()
    
    def analyze_market_regime(self) -> MarketRegime:
        """
        Analyze current market regime for strategy selection
        """
        try:
            # Fetch multiple timeframes
            df_1m = self.fetch_ohlcv_data('1m', 100)
            df_5m = self.fetch_ohlcv_data('5m', 100)
            df_15m = self.fetch_ohlcv_data('15m', 100)
            df_1h = self.fetch_ohlcv_data('1h', 100)
            
            if any(df.empty for df in [df_1m, df_5m, df_15m, df_1h]):
                return MarketRegime('sideways', 'medium', 'normal', 0, 0, 0)
            
            # TREND ANALYSIS
            # Multiple timeframe trend
            ema_fast_1h = df_1h['close'].ewm(span=10).mean().iloc[-1]
            ema_slow_1h = df_1h['close'].ewm(span=30).mean().iloc[-1]
            ema_fast_15m = df_15m['close'].ewm(span=10).mean().iloc[-1]
            ema_slow_15m = df_15m['close'].ewm(span=30).mean().iloc[-1]
            
            current_price = df_1m['close'].iloc[-1]
            
            # Trend strength
            trend_1h = 1 if ema_fast_1h > ema_slow_1h else -1
            trend_15m = 1 if ema_fast_15m > ema_slow_15m else -1
            trend_strength = (trend_1h + trend_15m) / 2
            
            if trend_strength > 0.5:
                trend = 'bullish'
            elif trend_strength < -0.5:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            # VOLATILITY ANALYSIS
            returns_1h = df_1h['close'].pct_change().dropna()
            volatility = returns_1h.std()
            vol_percentiles = np.percentile(returns_1h.rolling(24).std().dropna(), [25, 75, 95])
            
            if volatility < vol_percentiles[0]:
                volatility_regime = 'low'
            elif volatility < vol_percentiles[1]:
                volatility_regime = 'medium'
            elif volatility < vol_percentiles[2]:
                volatility_regime = 'high'
            else:
                volatility_regime = 'extreme'
            
            # VOLUME ANALYSIS
            avg_volume = df_1h['volume'].rolling(24).mean().iloc[-1]
            current_volume = df_1h['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio < 0.7:
                volume_regime = 'low'
            elif volume_ratio < 1.5:
                volume_regime = 'normal'
            else:
                volume_regime = 'high'
            
            # MOMENTUM ANALYSIS
            rsi = ta.momentum.RSIIndicator(df_15m['close'], window=14).rsi().iloc[-1]
            momentum_score = (rsi - 50) / 50  # Normalize to -1 to 1
            
            # MEAN REVERSION STRENGTH
            bb_indicator = ta.volatility.BollingerBands(df_15m['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband().iloc[-1]
            bb_lower = bb_indicator.bollinger_lband().iloc[-1]
            bb_mid = bb_indicator.bollinger_mavg().iloc[-1]
            
            price_position = (current_price - bb_mid) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0
            mean_reversion_strength = abs(price_position)
            
            # BREAKOUT PROBABILITY
            atr = ta.volatility.AverageTrueRange(df_15m['high'], df_15m['low'], df_15m['close'], window=14).average_true_range().iloc[-1]
            recent_range = df_15m['high'].tail(10).max() - df_15m['low'].tail(10).min()
            compression_ratio = atr / recent_range if recent_range > 0 else 0
            breakout_probability = min(compression_ratio * 2, 1.0)  # Normalize
            
            regime = MarketRegime(
                trend=trend,
                volatility=volatility_regime,
                volume=volume_regime,
                momentum=momentum_score,
                mean_reversion_strength=mean_reversion_strength,
                breakout_probability=breakout_probability
            )
            
            self.current_regime = regime
            self.regime_history.append(regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime('sideways', 'medium', 'normal', 0, 0, 0)
    
    def detect_momentum_opportunities(self) -> List[TradingOpportunity]:
        """
        STRATEGY 1: Momentum Breakout Detection
        """
        opportunities = []
        
        try:
            df = self.fetch_ohlcv_data('5m', 50)
            if df.empty:
                return opportunities
            
            # Calculate momentum indicators
            returns = df['close'].pct_change()
            momentum = returns.rolling(self.momentum_lookback).sum()
            volatility = returns.rolling(self.momentum_lookback).std()
            
            current_momentum = momentum.iloc[-1]
            current_volatility = volatility.iloc[-1]
            
            # Z-score momentum
            momentum_zscore = (current_momentum - momentum.mean()) / momentum.std()
            
            current_price = df['close'].iloc[-1]
            
            # Strong bullish momentum
            if momentum_zscore > self.momentum_threshold:
                entry_price = current_price * 1.001  # Slight premium
                exit_price = entry_price * (1 + self.momentum_profit_target)
                stop_loss = entry_price * (1 - self.momentum_stop_loss)
                
                confidence = min(abs(momentum_zscore) / 3, 1.0)
                expected_profit = self.momentum_profit_target * 100
                risk_score = self.momentum_stop_loss / self.momentum_profit_target
                
                position_size = self.calculate_position_size('momentum', confidence, risk_score)
                
                opportunities.append(TradingOpportunity(
                    strategy='momentum_long',
                    action='buy',
                    confidence=confidence,
                    expected_profit=expected_profit,
                    risk_score=risk_score,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    timeframe='5m',
                    metadata={'momentum_zscore': momentum_zscore, 'volatility': current_volatility}
                ))
            
            # Strong bearish momentum
            elif momentum_zscore < -self.momentum_threshold:
                entry_price = current_price * 0.999  # Slight discount
                exit_price = entry_price * (1 - self.momentum_profit_target)
                stop_loss = entry_price * (1 + self.momentum_stop_loss)
                
                confidence = min(abs(momentum_zscore) / 3, 1.0)
                expected_profit = self.momentum_profit_target * 100
                risk_score = self.momentum_stop_loss / self.momentum_profit_target
                
                position_size = self.calculate_position_size('momentum', confidence, risk_score)
                
                opportunities.append(TradingOpportunity(
                    strategy='momentum_short',
                    action='sell',
                    confidence=confidence,
                    expected_profit=expected_profit,
                    risk_score=risk_score,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    timeframe='5m',
                    metadata={'momentum_zscore': momentum_zscore, 'volatility': current_volatility}
                ))
            
        except Exception as e:
            logger.error(f"Error detecting momentum opportunities: {e}")
        
        return opportunities
    
    def detect_mean_reversion_opportunities(self) -> List[TradingOpportunity]:
        """
        STRATEGY 2: Mean Reversion with Multiple Indicators
        """
        opportunities = []
        
        try:
            df = self.fetch_ohlcv_data('15m', 100)
            if df.empty:
                return opportunities
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=self.bollinger_period, window_dev=self.bollinger_std)
            bb_upper = bb_indicator.bollinger_hband()
            bb_lower = bb_indicator.bollinger_lband()
            bb_mid = bb_indicator.bollinger_mavg()
            
            # RSI
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            stoch_k = stoch.stoch()
            
            current_price = df['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_stoch = stoch_k.iloc[-1]
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            current_bb_mid = bb_mid.iloc[-1]
            
            # Mean reversion signals
            signals = []
            
            # Oversold conditions
            if (current_price < current_bb_lower and 
                current_rsi < self.rsi_oversold and 
                current_stoch < 20):
                
                entry_price = current_price
                exit_price = current_bb_mid
                stop_loss = current_price * 0.995
                
                profit_pct = (exit_price - entry_price) / entry_price
                confidence = (self.rsi_oversold - current_rsi) / self.rsi_oversold + \
                           (20 - current_stoch) / 20
                confidence = min(confidence / 2, 1.0)
                
                risk_score = 0.005 / profit_pct if profit_pct > 0 else 1.0
                position_size = self.calculate_position_size('mean_reversion', confidence, risk_score)
                
                opportunities.append(TradingOpportunity(
                    strategy='mean_reversion_long',
                    action='buy',
                    confidence=confidence,
                    expected_profit=profit_pct * 100,
                    risk_score=risk_score,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    timeframe='15m',
                    metadata={'rsi': current_rsi, 'stoch': current_stoch, 'bb_position': 'below_lower'}
                ))
            
            # Overbought conditions
            elif (current_price > current_bb_upper and 
                  current_rsi > self.rsi_overbought and 
                  current_stoch > 80):
                
                entry_price = current_price
                exit_price = current_bb_mid
                stop_loss = current_price * 1.005
                
                profit_pct = (entry_price - exit_price) / entry_price
                confidence = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + \
                           (current_stoch - 80) / 20
                confidence = min(confidence / 2, 1.0)
                
                risk_score = 0.005 / profit_pct if profit_pct > 0 else 1.0
                position_size = self.calculate_position_size('mean_reversion', confidence, risk_score)
                
                opportunities.append(TradingOpportunity(
                    strategy='mean_reversion_short',
                    action='sell',
                    confidence=confidence,
                    expected_profit=profit_pct * 100,
                    risk_score=risk_score,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    position_size=position_size,
                    timeframe='15m',
                    metadata={'rsi': current_rsi, 'stoch': current_stoch, 'bb_position': 'above_upper'}
                ))
            
        except Exception as e:
            logger.error(f"Error detecting mean reversion opportunities: {e}")
        
        return opportunities
    
    def detect_scalping_opportunities(self) -> List[TradingOpportunity]:
        """
        STRATEGY 3: High-Frequency Scalping
        """
        opportunities = []
        
        try:
            df_1m = self.fetch_ohlcv_data('1m', 30)
            if df_1m.empty:
                return opportunities
            
            # Volume analysis
            volume_ma = df_1m['volume'].rolling(10).mean()
            current_volume = df_1m['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # Price action analysis
            current_price = df_1m['close'].iloc[-1]
            price_change = df_1m['close'].pct_change().iloc[-1]
            
            # Volatility analysis
            atr = ta.volatility.AverageTrueRange(df_1m['high'], df_1m['low'], df_1m['close'], window=10).average_true_range()
            current_atr = atr.iloc[-1]
            
            # Order book analysis (simplified)
            try:
                orderbook = self.exchange.fetch_order_book(self.symbol, limit=10)
                bid_volume = sum([bid[1] for bid in orderbook['bids'][:5]])
                ask_volume = sum([ask[1] for ask in orderbook['asks'][:5]])
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            except:
                imbalance = 0
            
            # Scalping conditions
            if (volume_ratio > self.scalp_volume_threshold and 
                abs(price_change) > 0.001 and 
                current_atr > 0):
                
                # Direction based on imbalance and momentum
                if imbalance > 0.2 and price_change > 0:
                    # Bullish scalp
                    entry_price = current_price * 1.0005
                    exit_price = entry_price * (1 + self.scalp_profit_target)
                    stop_loss = entry_price * (1 - self.scalp_stop_loss)
                    
                    confidence = min(volume_ratio / 3 + abs(imbalance), 1.0)
                    
                    opportunities.append(TradingOpportunity(
                        strategy='scalp_long',
                        action='buy',
                        confidence=confidence,
                        expected_profit=self.scalp_profit_target * 100,
                        risk_score=self.scalp_stop_loss / self.scalp_profit_target,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=stop_loss,
                        position_size=self.calculate_position_size('scalping', confidence, 0.67),
                        timeframe='1m',
                        metadata={'volume_ratio': volume_ratio, 'imbalance': imbalance, 'atr': current_atr}
                    ))
                
                elif imbalance < -0.2 and price_change < 0:
                    # Bearish scalp
                    entry_price = current_price * 0.9995
                    exit_price = entry_price * (1 - self.scalp_profit_target)
                    stop_loss = entry_price * (1 + self.scalp_stop_loss)
                    
                    confidence = min(volume_ratio / 3 + abs(imbalance), 1.0)
                    
                    opportunities.append(TradingOpportunity(
                        strategy='scalp_short',
                        action='sell',
                        confidence=confidence,
                        expected_profit=self.scalp_profit_target * 100,
                        risk_score=self.scalp_stop_loss / self.scalp_profit_target,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=stop_loss,
                        position_size=self.calculate_position_size('scalping', confidence, 0.67),
                        timeframe='1m',
                        metadata={'volume_ratio': volume_ratio, 'imbalance': imbalance, 'atr': current_atr}
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting scalping opportunities: {e}")
        
        return opportunities
    
    def detect_arbitrage_opportunities(self) -> List[TradingOpportunity]:
        """
        STRATEGY 4: Arbitrage Opportunities
        """
        opportunities = []
        
        try:
            # Funding rate arbitrage
            try:
                funding_rate = self.exchange.fetch_funding_rate(self.symbol)
                current_funding = funding_rate['fundingRate']
                
                if abs(current_funding) > self.funding_rate_threshold:
                    current_price = self.get_current_price()
                    
                    if current_funding > self.funding_rate_threshold:
                        # High positive funding - short futures, long spot
                        entry_price = current_price
                        expected_profit = abs(current_funding) * 100 * 3  # 3 funding periods per day
                        
                        opportunities.append(TradingOpportunity(
                            strategy='funding_arbitrage_short',
                            action='sell',
                            confidence=0.9,  # High confidence for arbitrage
                            expected_profit=expected_profit,
                            risk_score=0.1,  # Low risk for arbitrage
                            entry_price=entry_price,
                            exit_price=entry_price,  # Hold for funding
                            stop_loss=entry_price * 1.01,  # Wide stop
                            position_size=self.calculate_position_size('arbitrage', 0.9, 0.1),
                            timeframe='8h',  # Funding period
                            metadata={'funding_rate': current_funding, 'type': 'funding_arbitrage'}
                        ))
                    
                    elif current_funding < -self.funding_rate_threshold:
                        # High negative funding - long futures, short spot
                        entry_price = current_price
                        expected_profit = abs(current_funding) * 100 * 3
                        
                        opportunities.append(TradingOpportunity(
                            strategy='funding_arbitrage_long',
                            action='buy',
                            confidence=0.9,
                            expected_profit=expected_profit,
                            risk_score=0.1,
                            entry_price=entry_price,
                            exit_price=entry_price,
                            stop_loss=entry_price * 0.99,
                            position_size=self.calculate_position_size('arbitrage', 0.9, 0.1),
                            timeframe='8h',
                            metadata={'funding_rate': current_funding, 'type': 'funding_arbitrage'}
                        ))
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")
        
        return opportunities
    
    def generate_ml_predictions(self) -> List[TradingOpportunity]:
        """
        STRATEGY 5: Machine Learning Predictions (Fixed)
        """
        opportunities = []
        
        try:
            if not self.ml_strategy:
                return opportunities
            
            # Prepare features using standardized method
            features = self.prepare_ml_features()
            if features is None or len(features) != 6:
                return opportunities
            
            # Train model if needed
            if self.ml_model is None:
                logger.info("🧠 Training ML model for first time...")
                self.train_ml_model()
            
            if self.ml_model is None:
                return opportunities
            
            # Generate prediction
            try:
                features_scaled = self.feature_scaler.transform([features])
                prediction = self.ml_model.predict(features_scaled)[0]
                prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
                confidence = prediction_proba.max()
                
                current_price = self.get_current_price()
                
                if confidence > self.ml_confidence_threshold:
                    if prediction > 0:  # Bullish prediction
                        entry_price = current_price * 1.001
                        exit_price = entry_price * 1.008  # 0.8% target
                        stop_loss = entry_price * 0.996  # 0.4% stop
                        
                        opportunities.append(TradingOpportunity(
                            strategy='ml_long',
                            action='buy',
                            confidence=confidence,
                            expected_profit=0.8,
                            risk_score=0.5,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            position_size=self.calculate_position_size('ml', confidence, 0.5),
                            timeframe='1h',
                            metadata={'ml_prediction': prediction, 'ml_confidence': confidence}
                        ))
                    
                    elif prediction < 0:  # Bearish prediction
                        entry_price = current_price * 0.999
                        exit_price = entry_price * 0.992
                        stop_loss = entry_price * 1.004
                        
                        opportunities.append(TradingOpportunity(
                            strategy='ml_short',
                            action='sell',
                            confidence=confidence,
                            expected_profit=0.8,
                            risk_score=0.5,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            position_size=self.calculate_position_size('ml', confidence, 0.5),
                            timeframe='1h',
                            metadata={'ml_prediction': prediction, 'ml_confidence': confidence}
                        ))
                        
            except Exception as pred_error:
                logger.warning(f"ML prediction error: {pred_error}")
                # Temporarily disable ML if there are issues
                return opportunities
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
        
        return opportunities
    
    def prepare_ml_features(self) -> Optional[List[float]]:
        """
        Prepare features for ML model (standardized version)
        """
        try:
            df = self.fetch_ohlcv_data('1h', 100)
            if df.empty or len(df) < 50:
                return None
            
            # Use the same feature extraction as training
            features = self.extract_features_from_subset(df.tail(50))
            return features
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None
    
    def train_ml_model(self):
        """
        Train ML model for predictions (Fixed)
        """
        try:
            # Fetch training data
            df = self.fetch_ohlcv_data('1h', 1000)
            if df.empty or len(df) < 200:
                logger.warning("Insufficient data for ML training")
                return
            
            # Prepare training data
            X = []
            y = []
            
            for i in range(50, len(df) - 10):  # Need lookback and lookahead
                # Features for time i
                subset = df.iloc[i-50:i]
                features = self.extract_features_from_subset(subset)
                if features and len(features) == 6:  # Ensure exactly 6 features
                    X.append(features)
                    
                    # Target: future return (next 10 periods)
                    future_return = (df['close'].iloc[i+10] - df['close'].iloc[i]) / df['close'].iloc[i]
                    
                    # Create discrete targets for classification
                    if future_return > 0.004:  # 0.4% gain
                        y.append(1)  # Bullish
                    elif future_return < -0.004:  # 0.4% loss
                        y.append(-1)  # Bearish
                    else:
                        y.append(0)  # Neutral
            
            if len(X) < 50:
                logger.warning("Not enough valid training samples")
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Remove any rows with NaN or infinite values
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 30:
                logger.warning("Not enough valid samples after cleaning")
                return
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            from sklearn.ensemble import RandomForestClassifier
            self.ml_model = RandomForestClassifier(
                n_estimators=50,  # Reduced for faster training
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
            
            self.ml_model.fit(X_scaled, y)
            
            # Calculate training accuracy
            train_score = self.ml_model.score(X_scaled, y)
            
            logger.info(f"✅ ML model trained with {len(X)} samples, accuracy: {train_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.ml_model = None
    
    def extract_features_from_subset(self, df_subset):
        """Extract standardized features from a subset of data"""
        try:
            if len(df_subset) < 20:
                return None
            
            features = []
            
            # 1. Price return features
            returns = df_subset['close'].pct_change().dropna()
            if len(returns) > 0:
                features.append(returns.mean())  # Average return
                features.append(returns.std())   # Return volatility
                features.append(returns.iloc[-1])  # Latest return
            else:
                features.extend([0, 0, 0])
            
            # 2. RSI (technical indicator)
            if len(df_subset) >= 14:
                rsi = ta.momentum.RSIIndicator(df_subset['close'], window=14).rsi().iloc[-1]
                features.append(rsi / 100 if not np.isnan(rsi) else 0.5)
            else:
                features.append(0.5)
            
            # 3. Volume ratio
            volume_mean = df_subset['volume'].mean()
            volume_ratio = df_subset['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1
            features.append(min(volume_ratio, 5))  # Cap at 5x
            
            # 4. Trend strength (EMA crossover)
            if len(df_subset) >= 12:
                ema_fast = df_subset['close'].ewm(span=5).mean().iloc[-1]
                ema_slow = df_subset['close'].ewm(span=10).mean().iloc[-1]
                trend = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
                features.append(trend)
            else:
                features.append(0)
            
            # Ensure we always return exactly 6 features
            while len(features) < 6:
                features.append(0)
            
            return features[:6]  # Truncate to exactly 6 features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0, 0, 0, 0, 0, 0]  # Return default 6 features
    
    def calculate_position_size(self, strategy: str, confidence: float, risk_score: float) -> float:
        """
        Calculate optimal position size using multiple methods
        """
        try:
            # Base position size
            base_size = self.available_capital * 0.05  # 5% base
            
            # Kelly criterion adjustment
            if self.kelly_criterion and strategy in self.strategy_performance:
                perf = self.strategy_performance[strategy]
                if perf['trades'] > 10:
                    win_rate = perf['win_rate']
                    avg_win = perf.get('avg_win', 0.01)
                    avg_loss = perf.get('avg_loss', 0.005)
                    
                    if avg_win > 0 and avg_loss > 0:
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                        base_size *= (1 + kelly_fraction)
            
            # Confidence adjustment
            size_with_confidence = base_size * confidence
            
            # Risk adjustment
            risk_adjusted_size = size_with_confidence / (1 + risk_score)
            
            # Volatility adjustment
            if self.volatility_adjustment:
                regime = self.current_regime
                if regime and regime.volatility == 'high':
                    risk_adjusted_size *= 0.7
                elif regime and regime.volatility == 'extreme':
                    risk_adjusted_size *= 0.4
            
            # Maximum position size limit
            final_size = min(risk_adjusted_size, self.available_capital * self.max_position_size)
            
            return max(final_size, 10)  # Minimum $10 position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 50  # Default size
    
    def execute_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """
        Execute a trading opportunity
        """
        try:
            # Check if we can take this trade
            if not self.can_take_trade(opportunity):
                return False
            
            current_price = self.get_current_price()
            position_size_usdt = opportunity.position_size
            
            # Calculate actual position size in contracts
            position_size_contracts = self.calculate_position_size_for_amount(current_price, position_size_usdt)
            
            # Place order based on action
            order = None
            if opportunity.action == 'buy':
                order = self.place_market_buy_order(position_size_contracts, opportunity)
            elif opportunity.action == 'sell':
                order = self.place_market_sell_order(position_size_contracts, opportunity)
            
            if order:
                # Store position
                position_id = f"{opportunity.strategy}_{int(time.time())}"
                self.active_positions[position_id] = {
                    'opportunity': opportunity,
                    'order': order,
                    'entry_time': time.time(),
                    'entry_price': current_price,
                    'size': position_size_contracts,
                    'side': opportunity.action,
                    'status': 'open'
                }
                
                # Update capital allocation
                self.available_capital -= position_size_usdt
                self.allocated_capital[opportunity.strategy] += position_size_usdt
                
                logger.info(f"✅ Executed {opportunity.strategy}: {opportunity.action} {position_size_contracts:.4f} @ ${current_price:.4f}")
                logger.info(f"💰 Expected profit: {opportunity.expected_profit:.2f}%, Confidence: {opportunity.confidence:.2f}")
                
                return True
            
        except Exception as e:
            logger.error(f"Error executing opportunity: {e}")
        
        return False
    
    def can_take_trade(self, opportunity: TradingOpportunity) -> bool:
        """
        Check if we can take this trade
        """
        try:
            # Check maximum concurrent trades
            if len(self.active_positions) >= self.max_concurrent_trades:
                return False
            
            # Check daily trade limit
            today = datetime.now().date()
            today_trades = len([t for t in self.trade_history if t.get('date') == today])
            if today_trades >= self.max_daily_trades:
                return False
            
            # Check available capital
            if self.available_capital < opportunity.position_size:
                return False
            
            # Check maximum drawdown
            if self.is_max_drawdown_exceeded():
                return False
            
            # Check strategy allocation limits
            strategy_allocation = self.allocated_capital[opportunity.strategy]
            max_strategy_allocation = self.total_capital * 0.3  # 30% per strategy
            if strategy_allocation + opportunity.position_size > max_strategy_allocation:
                return False
            
            # Check correlation limits (simplified)
            if self.correlation_adjustment:
                similar_positions = [p for p in self.active_positions.values() 
                                   if p['opportunity'].strategy.startswith(opportunity.strategy.split('_')[0])]
                if len(similar_positions) >= 3:  # Max 3 similar positions
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade eligibility: {e}")
            return False
    
    def place_market_buy_order(self, position_size: float, opportunity: TradingOpportunity):
        """
        Place market buy order
        """
        try:
            params = {
                'timeInForce': 'IOC',  # Immediate or Cancel
                'clientOrderId': f"{opportunity.strategy}_{int(time.time() * 1000)}"
            }
            
            order = self.exchange.create_market_buy_order(
                symbol=self.symbol,
                amount=position_size,
                params=params
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return None
    
    def place_market_sell_order(self, position_size: float, opportunity: TradingOpportunity):
        """
        Place market sell order
        """
        try:
            params = {
                'timeInForce': 'IOC',
                'clientOrderId': f"{opportunity.strategy}_{int(time.time() * 1000)}"
            }
            
            order = self.exchange.create_market_sell_order(
                symbol=self.symbol,
                amount=position_size,
                params=params
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return None
    
    def manage_active_positions(self):
        """
        Manage active positions - exits, stop losses, profit taking
        """
        try:
            current_price = self.get_current_price()
            current_time = time.time()
            
            positions_to_close = []
            
            for position_id, position in self.active_positions.items():
                opportunity = position['opportunity']
                entry_price = position['entry_price']
                entry_time = position['entry_time']
                side = position['side']
                
                # Calculate current P&L
                if side == 'buy':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:  # sell
                    unrealized_pnl = (entry_price - current_price) / entry_price
                
                position['unrealized_pnl'] = unrealized_pnl
                
                # Check exit conditions
                should_close = False
                close_reason = ""
                
                # Profit target hit
                if unrealized_pnl >= (opportunity.expected_profit / 100):
                    should_close = True
                    close_reason = "profit_target"
                
                # Stop loss hit
                elif side == 'buy' and current_price <= opportunity.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif side == 'sell' and current_price >= opportunity.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                
                # Time-based exits
                elif opportunity.strategy.startswith('scalp') and (current_time - entry_time) > self.scalp_max_duration:
                    should_close = True
                    close_reason = "time_limit"
                
                # Trailing stop for profitable trades
                elif unrealized_pnl > 0.01:  # 1% profit
                    trailing_stop_distance = 0.005  # 0.5%
                    if side == 'buy' and current_price <= (entry_price * (1 + unrealized_pnl - trailing_stop_distance)):
                        should_close = True
                        close_reason = "trailing_stop"
                    elif side == 'sell' and current_price >= (entry_price * (1 - unrealized_pnl + trailing_stop_distance)):
                        should_close = True
                        close_reason = "trailing_stop"
                
                # Strategy-specific exits
                if opportunity.strategy == 'funding_arbitrage_short' or opportunity.strategy == 'funding_arbitrage_long':
                    # Hold until next funding time (simplified)
                    if (current_time - entry_time) > 28800:  # 8 hours
                        should_close = True
                        close_reason = "funding_period_end"
                
                if should_close:
                    positions_to_close.append((position_id, close_reason))
            
            # Close positions
            for position_id, close_reason in positions_to_close:
                self.close_position(position_id, close_reason)
                
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def close_position(self, position_id: str, reason: str):
        """
        Close a position
        """
        try:
            position = self.active_positions[position_id]
            opportunity = position['opportunity']
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']
            entry_time = position['entry_time']
            
            current_price = self.get_current_price()
            
            # Place closing order
            if side == 'buy':
                close_order = self.exchange.create_market_sell_order(self.symbol, size)
            else:
                close_order = self.exchange.create_market_buy_order(self.symbol, size)
            
            # Calculate realized P&L
            if side == 'buy':
                realized_pnl = (current_price - entry_price) / entry_price
            else:
                realized_pnl = (entry_price - current_price) / entry_price
            
            realized_pnl_usd = realized_pnl * opportunity.position_size
            
            # Update performance tracking
            self.total_profit += realized_pnl_usd
            self.total_trades += 1
            
            if realized_pnl_usd > 0:
                self.winning_trades += 1
                self.current_consecutive_losses = 0
            else:
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_consecutive_losses)
            
            # Update strategy performance
            strategy = opportunity.strategy.split('_')[0]  # Base strategy name
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {'trades': 0, 'profit': 0, 'wins': 0, 'losses': 0}
            
            self.strategy_performance[strategy]['trades'] += 1
            self.strategy_performance[strategy]['profit'] += realized_pnl_usd
            
            if realized_pnl_usd > 0:
                self.strategy_performance[strategy]['wins'] += 1
            else:
                self.strategy_performance[strategy]['losses'] += 1
            
            self.strategy_performance[strategy]['win_rate'] = (
                self.strategy_performance[strategy]['wins'] / 
                self.strategy_performance[strategy]['trades']
            )
            
            # Update capital
            self.available_capital += opportunity.position_size
            self.allocated_capital[opportunity.strategy] -= opportunity.position_size
            
            # Store trade record
            trade_record = {
                'date': datetime.now().date(),
                'strategy': opportunity.strategy,
                'side': side,
                'entry_price': entry_price,
                'exit_price': current_price,
                'size': size,
                'duration': time.time() - entry_time,
                'realized_pnl': realized_pnl_usd,
                'realized_pnl_pct': realized_pnl * 100,
                'close_reason': reason,
                'confidence': opportunity.confidence
            }
            
            self.trade_history.append(trade_record)
            
            # Update daily PnL
            today = datetime.now().date()
            self.daily_pnl[today] += realized_pnl_usd
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            logger.info(f"🔄 Closed {opportunity.strategy}: {side} position")
            logger.info(f"💰 P&L: ${realized_pnl_usd:.4f} ({realized_pnl*100:.2f}%) - Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def is_max_drawdown_exceeded(self) -> bool:
        """
        Check if maximum drawdown is exceeded
        """
        try:
            if not self.daily_pnl:
                return False
            
            # Calculate cumulative returns
            cumulative_pnl = 0
            peak = 0
            max_dd = 0
            
            for pnl in self.daily_pnl.values():
                cumulative_pnl += pnl
                peak = max(peak, cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / self.total_capital if self.total_capital > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            return max_dd > self.max_drawdown
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return False
    
    def get_all_opportunities(self) -> List[TradingOpportunity]:
        """
        Get opportunities from all strategies
        """
        all_opportunities = []
        
        try:
            # Update market regime
            regime = self.analyze_market_regime()
            
            # Get opportunities from each strategy based on market regime
            if self.momentum_strategy and regime.trend != 'sideways':
                all_opportunities.extend(self.detect_momentum_opportunities())
            
            if self.mean_reversion_strategy and regime.trend == 'sideways':
                all_opportunities.extend(self.detect_mean_reversion_opportunities())
            
            if self.scalping_strategy and regime.volatility in ['medium', 'high']:
                all_opportunities.extend(self.detect_scalping_opportunities())
            
            if self.arbitrage_strategy:
                all_opportunities.extend(self.detect_arbitrage_opportunities())
            
            if self.ml_strategy:
                all_opportunities.extend(self.generate_ml_predictions())
            
            # Sort by expected profit * confidence
            all_opportunities.sort(key=lambda x: x.expected_profit * x.confidence, reverse=True)
            
            return all_opportunities[:5]  # Top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error getting opportunities: {e}")
            return []
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        try:
            if not self.trade_history:
                return {}
            
            # Basic metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['realized_pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = sum(t['realized_pnl'] for t in self.trade_history)
            avg_win = np.mean([t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
            
            # Risk metrics
            daily_returns = list(self.daily_pnl.values())
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Drawdown
            cumulative_pnl = 0
            peak = 0
            max_dd = 0
            
            for pnl in daily_returns:
                cumulative_pnl += pnl
                peak = max(peak, cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / self.total_capital if self.total_capital > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            # Strategy breakdown
            strategy_stats = {}
            for strategy, perf in self.strategy_performance.items():
                if perf['trades'] > 0:
                    strategy_stats[strategy] = {
                        'trades': perf['trades'],
                        'profit': round(perf['profit'], 4),
                        'win_rate': round(perf['win_rate'] * 100, 1)
                    }
            
            return {
                'total_trades': total_trades,
                'win_rate': round(win_rate * 100, 1),
                'total_profit': round(total_pnl, 4),
                'avg_win': round(avg_win, 4),
                'avg_loss': round(avg_loss, 4),
                'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf'),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_dd * 100, 2),
                'active_positions': len(self.active_positions),
                'available_capital': round(self.available_capital, 2),
                'total_capital': round(self.total_capital, 2),
                'capital_utilization': round((self.total_capital - self.available_capital) / self.total_capital * 100, 1) if self.total_capital > 0 else 0,
                'strategy_performance': strategy_stats,
                'max_consecutive_losses': self.max_consecutive_losses
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def display_performance_dashboard(self):
        """
        Display comprehensive performance dashboard
        """
        try:
            metrics = self.calculate_performance_metrics()
            current_price = self.get_current_price()
            
            logger.info("=" * 80)
            logger.info("🚀 MULTI-STRATEGY ADAPTIVE BOT DASHBOARD")
            logger.info("=" * 80)
            
            # Market information
            if self.current_regime:
                logger.info(f"📊 Market Regime: {self.current_regime.trend.upper()} | "
                          f"Vol: {self.current_regime.volatility.upper()} | "
                          f"Volume: {self.current_regime.volume.upper()}")
                logger.info(f"📈 Momentum: {self.current_regime.momentum:.3f} | "
                          f"Mean Rev: {self.current_regime.mean_reversion_strength:.3f} | "
                          f"Breakout Prob: {self.current_regime.breakout_probability:.3f}")
            
            logger.info(f"💱 Current Price: ${current_price:.4f}")
            logger.info(f"💰 Available Capital: ${metrics.get('available_capital', 0):.2f}")
            logger.info(f"📊 Capital Utilization: {metrics.get('capital_utilization', 0):.1f}%")
            
            # Performance metrics
            if metrics:
                logger.info(f"📈 Total Trades: {metrics['total_trades']}")
                logger.info(f"🎯 Win Rate: {metrics['win_rate']:.1f}%")
                logger.info(f"💵 Total Profit: ${metrics['total_profit']:.4f}")
                logger.info(f"📊 Profit Factor: {metrics['profit_factor']:.2f}")
                logger.info(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                logger.info(f"📉 Max Drawdown: {metrics['max_drawdown']:.2f}%")
                logger.info(f"🔄 Active Positions: {metrics['active_positions']}")
            
            # Active positions summary
            if self.active_positions:
                logger.info("\n📋 ACTIVE POSITIONS:")
                for pos_id, pos in self.active_positions.items():
                    pnl = pos.get('unrealized_pnl', 0) * 100
                    logger.info(f"  {pos['opportunity'].strategy}: {pos['side']} | "
                              f"P&L: {pnl:.2f}% | "
                              f"Size: ${pos['opportunity'].position_size:.0f}")
            
            # Strategy performance
            if metrics.get('strategy_performance'):
                logger.info("\n📊 STRATEGY PERFORMANCE:")
                for strategy, stats in metrics['strategy_performance'].items():
                    logger.info(f"  {strategy.upper()}: {stats['trades']} trades | "
                              f"${stats['profit']:.2f} profit | "
                              f"{stats['win_rate']:.1f}% win rate")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
    
    def run_adaptive_bot(self, total_capital: float, check_interval: int = 5):
        """
        Run the multi-strategy adaptive bot
        """
        try:
            logger.info("🚀 Starting Multi-Strategy Adaptive Trading Bot...")
            
            # Initialize capital
            self.total_capital = total_capital
            self.available_capital = total_capital
            
            # Set leverage
            self.set_leverage()
            
            # Initial ML model training
            if self.ml_strategy:
                logger.info("🧠 Training initial ML model...")
                self.train_ml_model()
            
            logger.info(f"💰 Total Capital: ${self.total_capital:.2f}")
            logger.info(f"⚙️ Strategies Active: Momentum, Mean Reversion, Scalping, Arbitrage, ML")
            logger.info(f"🔄 Check Interval: {check_interval}s")
            
            last_ml_training = time.time()
            last_dashboard_update = time.time()
            iteration = 0
            
            while True:
                try:
                    iteration += 1
                    current_time = time.time()
                    
                    # 1. Manage existing positions
                    self.manage_active_positions()
                    
                    # 2. Look for new opportunities
                    opportunities = self.get_all_opportunities()
                    
                    # 3. Execute best opportunities
                    for opportunity in opportunities[:3]:  # Top 3
                        if opportunity.confidence > 0.6:  # Minimum confidence
                            executed = self.execute_opportunity(opportunity)
                            if executed:
                                time.sleep(1)  # Brief pause between executions
                    
                    # 4. Retrain ML model periodically
                    if (self.ml_strategy and 
                        current_time - last_ml_training > self.ml_retrain_interval):
                        logger.info("🧠 Retraining ML model...")
                        self.train_ml_model()
                        last_ml_training = current_time
                    
                    # 5. Display dashboard
                    if current_time - last_dashboard_update > 30:  # Every 30 seconds
                        self.display_performance_dashboard()
                        last_dashboard_update = current_time
                    
                    # 6. Emergency stops
                    if self.is_max_drawdown_exceeded():
                        logger.warning("🚨 Maximum drawdown exceeded - stopping bot")
                        break
                    
                    # 7. Log opportunities (every 10 iterations)
                    if iteration % 10 == 0 and opportunities:
                        logger.info(f"🎯 Found {len(opportunities)} opportunities")
                        best = opportunities[0]
                        logger.info(f"   Best: {best.strategy} | "
                                  f"Confidence: {best.confidence:.2f} | "
                                  f"Expected: {best.expected_profit:.2f}%")
                    
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    time.sleep(check_interval * 2)
                    
        except Exception as e:
            logger.error(f"💥 Fatal error in adaptive bot: {e}")
            raise
        finally:
            self.stop_bot()
    
    def stop_bot(self):
        """
        Clean shutdown of the bot
        """
        try:
            logger.info("🛑 Stopping Multi-Strategy Adaptive Bot...")
            
            # Close all active positions
            logger.info(f"📋 Closing {len(self.active_positions)} active positions...")
            for position_id in list(self.active_positions.keys()):
                self.close_position(position_id, "bot_shutdown")
            
            # Final performance summary
            final_metrics = self.calculate_performance_metrics()
            if final_metrics:
                logger.info("📊 FINAL PERFORMANCE SUMMARY:")
                logger.info(f"   Total Profit: ${final_metrics['total_profit']:.4f}")
                logger.info(f"   Win Rate: {final_metrics['win_rate']:.1f}%")
                logger.info(f"   Total Trades: {final_metrics['total_trades']}")
                logger.info(f"   Sharpe Ratio: {final_metrics['sharpe_ratio']:.3f}")
            
            logger.info("✅ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    # Utility methods
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
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0
    
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


# USAGE EXAMPLE
if __name__ == "__main__":
    # Configuration
    API_KEY = "VDpt0WQXIjXul4OBrS"
    API_SECRET = "z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9"
    
    # Investment amount
    TOTAL_CAPITAL = 110  # USDT
    
    # Initialize multi-strategy bot
    adaptive_bot = MultiStrategyAdaptiveBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='DOGE/USDT:USDT',
        testnet=False  # Set to False for live trading
    )
    
    # Configure strategies (all enabled by default)
    adaptive_bot.momentum_strategy = True
    adaptive_bot.mean_reversion_strategy = True
    adaptive_bot.scalping_strategy = True
    adaptive_bot.arbitrage_strategy = True
    adaptive_bot.ml_strategy = True
    adaptive_bot.sentiment_strategy = False  # Disabled for now
    adaptive_bot.delta_neutral_strategy = False  # Disabled for now
    
    # Risk management settings
    adaptive_bot.max_concurrent_trades = 6
    adaptive_bot.max_daily_trades = 40
    adaptive_bot.max_drawdown = 0.15  # 15%
    adaptive_bot.max_position_size = 0.12  # 12% per trade
    
    # Strategy-specific settings
    adaptive_bot.momentum_threshold = 1.8  # Slightly more sensitive
    adaptive_bot.scalp_profit_target = 0.004  # 0.4% for DOGE
    adaptive_bot.mean_reversion_profit = 0.015  # 1.5%
    adaptive_bot.ml_confidence_threshold = 0.65  # Lower for more trades
    
    # Position sizing settings
    adaptive_bot.kelly_criterion = True
    adaptive_bot.volatility_adjustment = True
    adaptive_bot.correlation_adjustment = True
    
    try:
        logger.info("🚀 MULTI-STRATEGY ADAPTIVE TRADING BOT")
        logger.info("💎 Strategies: Momentum, Mean Reversion, Scalping, Arbitrage, ML")
        logger.info("🧠 Machine Learning: ENABLED")
        logger.info("📊 Market Regime Analysis: ACTIVE")
        logger.info("💰 Kelly Criterion Position Sizing: ENABLED")
        logger.info("⚡ Real-time Opportunity Detection: RUNNING")
        logger.info("🎯 Adaptive Strategy Selection: ACTIVE")
        
        # Run the adaptive bot
        adaptive_bot.run_adaptive_bot(
            total_capital=TOTAL_CAPITAL,
            check_interval=5  # Check every 5 seconds
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
    finally:
        adaptive_bot.stop_bot()