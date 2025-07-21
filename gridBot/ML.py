"""
Ultra High-Performance AI Cryptocurrency Trading Bot
Designed for extreme risk tolerance (40% per trade) with 10-20x leverage
Uses ensemble deep learning models optimized for CPU execution
"""

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import asyncio
import websocket
import threading
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import ta
from scipy import stats

# System imports
import redis
import pickle
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import hmac
import requests
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

class Config:
    """Central configuration for the trading bot"""
    
    # API Configuration
    BYBIT_API_KEY = "YOUR_API_KEY"
    BYBIT_API_SECRET = "YOUR_API_SECRET"
    BYBIT_TESTNET = True
    
    # Trading Parameters
    MAX_RISK_PER_TRADE = 0.40  # 40% risk per trade
    DEFAULT_LEVERAGE = 20
    MIN_LEVERAGE = 10
    MAX_LEVERAGE = 25
    
    # Model Parameters
    CONFIDENCE_THRESHOLD = 0.65
    ENSEMBLE_MODELS = 3
    PREDICTION_WINDOW = 5  # 5 minutes ahead
    
    # Risk Management
    MAX_DAILY_LOSS = 0.20  # 20% daily loss limit
    MAX_POSITIONS = 8
    CORRELATION_THRESHOLD = 0.8
    CIRCUIT_BREAKER_THRESHOLD = 0.10  # 10% rapid market move
    
    # Execution Parameters
    TARGET_LATENCY_MS = 10
    MIN_VOLUME_USD = 1_000_000  # $1M daily volume minimum
    
    # Data Configuration
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # Trading Pairs
    TRADING_PAIRS = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "DOGE/USDT:USDT",
        "AVAX/USDT:USDT",
        "MATIC/USDT:USDT",
        "LINK/USDT:USDT",
        "DOT/USDT:USDT"
    ]
    
    # Timeframes
    PRIMARY_TIMEFRAME = "1m"
    ANALYSIS_TIMEFRAMES = ["1m", "5m", "15m"]

# ==================== NEURAL NETWORK MODELS ====================

class TemporalAttention(nn.Module):
    """Multi-head temporal attention mechanism"""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.fc(context)

class LightweightGRU(nn.Module):
    """Fast GRU model for short-term predictions"""
    
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = TemporalAttention(hidden_sizes[1])
        self.fc = nn.Linear(hidden_sizes[1], 3)  # Buy, Hold, Sell
        
    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, hidden = self.gru2(x)
        x = self.attention(x)
        x = x[:, -1, :]  # Take last timestep
        return torch.softmax(self.fc(x), dim=1)

class EnhancedBiLSTM(nn.Module):
    """Bidirectional LSTM with attention for long-term patterns"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attention = TemporalAttention(hidden_sizes[1] * 2)
        self.fc = nn.Linear(hidden_sizes[1] * 2, 3)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.attention(x)
        x = x[:, -1, :]
        return torch.softmax(self.fc(x), dim=1)

class MomentumPredictor(nn.Module):
    """Linear momentum model for rapid movements"""
    
    def __init__(self, input_size, lookback=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size * lookback, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.softmax(self.fc3(x), dim=1)

class EnsembleMetaLearner(nn.Module):
    """Meta-learner to combine predictions"""
    
    def __init__(self, num_models=3):
        super().__init__()
        self.fc1 = nn.Linear(num_models * 3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        
    def forward(self, predictions):
        x = torch.cat(predictions, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

# ==================== DEEP Q-NETWORK ====================

class CryptoDQN(nn.Module):
    """Deep Q-Network for reinforcement learning"""
    
    def __init__(self, state_size=50, action_size=3):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    """DQN Agent for trading decisions"""
    
    def __init__(self, state_size=50, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
        self.model = CryptoDQN(state_size, action_size)
        self.target_model = CryptoDQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return np.argmax(q_values.detach().numpy())
        
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
                
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            loss = F.mse_loss(self.model(state_tensor), target_f)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==================== FEATURE ENGINEERING ====================

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self, lookback_periods=100):
        self.lookback_periods = lookback_periods
        self.scaler = RobustScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_to_high'] = df['close'] / df['high']
        df['close_to_low'] = df['close'] / df['low']
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_delta'] = df['volume'] - df['volume'].shift(1)
        df['price_volume'] = df['close'] * df['volume']
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = df['bb_high'] - df['bb_low']
        df['bb_position'] = (df['close'] - df['bb_low']) / df['bb_width']
        
        # ATR and volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Microstructure features
        df['bid_ask_spread'] = df['high'] - df['low']
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Order flow imbalance
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.0001)
        df['order_flow_imbalance'] = df['buy_pressure'] - df['sell_pressure']
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Pattern recognition
        df['hammer'] = ((df['low'] - df['close']).abs() < 0.1 * (df['high'] - df['low'])) & \
                       (df['close'] > df['open'])
        df['doji'] = (df['close'] - df['open']).abs() < 0.1 * (df['high'] - df['low'])
        
        # Market regime
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['regime_volatility'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # Correlation features (calculated separately for each pair)
        # These would be added in the main trading loop
        
        return df.dropna()
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU models"""
        
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df[feature_cols])
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(scaled_features) - 5):  # -5 for 5-minute prediction
            sequences.append(scaled_features[i-sequence_length:i])
            
            # Create target: 1 if price increases >0.1% in next 5 minutes, -1 if decreases, 0 otherwise
            future_return = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
            
            if future_return > 0.001:  # 0.1% threshold
                targets.append(0)  # Buy
            elif future_return < -0.001:
                targets.append(2)  # Sell
            else:
                targets.append(1)  # Hold
                
        return np.array(sequences), np.array(targets)

# ==================== MARKET DATA MANAGER ====================

class MarketDataManager:
    """Handles all market data operations"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.exchange = None
        self.ws_connections = {}
        self.data_buffer = {symbol: deque(maxlen=1000) for symbol in Config.TRADING_PAIRS}
        
    def initialize_exchange(self):
        """Initialize Bybit exchange connection"""
        self.exchange = ccxt.bybit({
            'apiKey': Config.BYBIT_API_KEY,
            'secret': Config.BYBIT_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'testnet': Config.BYBIT_TESTNET
            }
        })
        
        if Config.BYBIT_TESTNET:
            self.exchange.set_sandbox_mode(True)
            
    def start_websocket_feeds(self):
        """Start WebSocket connections for real-time data"""
        for symbol in Config.TRADING_PAIRS:
            threading.Thread(target=self._websocket_handler, args=(symbol,), daemon=True).start()
            
    def _websocket_handler(self, symbol):
        """Handle WebSocket connection for a symbol"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    self._process_tick_data(symbol, data['data'])
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error for {symbol}: {error}")
            
        def on_close(ws):
            logger.info(f"WebSocket closed for {symbol}")
            # Reconnect after 5 seconds
            time.sleep(5)
            self._websocket_handler(symbol)
            
        # Bybit WebSocket URL
        ws_url = "wss://stream-testnet.bybit.com/v5/public/linear" if Config.BYBIT_TESTNET else "wss://stream.bybit.com/v5/public/linear"
        
        ws = websocket.WebSocketApp(ws_url,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        
        # Subscribe to ticker
        ws.on_open = lambda ws: ws.send(json.dumps({
            "op": "subscribe",
            "args": [f"tickers.{symbol.replace('/', '').replace(':USDT', '')}"]
        }))
        
        self.ws_connections[symbol] = ws
        ws.run_forever()
        
    def _process_tick_data(self, symbol, data):
        """Process incoming tick data"""
        timestamp = datetime.now()
        
        tick = {
            'timestamp': timestamp,
            'symbol': symbol,
            'bid': float(data.get('bid1Price', 0)),
            'ask': float(data.get('ask1Price', 0)),
            'last': float(data.get('lastPrice', 0)),
            'volume': float(data.get('volume24h', 0))
        }
        
        # Add to buffer
        self.data_buffer[symbol].append(tick)
        
        # Store in Redis
        key = f"tick:{symbol}:{timestamp.timestamp()}"
        self.redis_client.setex(key, 3600, json.dumps(tick))  # Expire after 1 hour
        
        # Update latest price
        self.redis_client.set(f"latest:{symbol}", json.dumps(tick))
        
    def get_historical_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get current orderbook"""
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {}

# ==================== MODEL MANAGER ====================

class ModelManager:
    """Manages all ML models and predictions"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.onnx_sessions = {}
        self.feature_engineer = FeatureEngineer()
        self.dqn_agent = DQNAgent()
        self.performance_tracker = deque(maxlen=1000)
        
    def initialize_models(self):
        """Initialize all trading models"""
        # Create PyTorch models
        self.models['gru_short'] = LightweightGRU(input_size=50)
        self.models['lstm_long'] = EnhancedBiLSTM(input_size=50)
        self.models['momentum'] = MomentumPredictor(input_size=50)
        self.models['meta_learner'] = EnsembleMetaLearner()
        
        # Load pre-trained weights if available
        self._load_model_weights()
        
        # Convert to ONNX for faster inference
        self._convert_to_onnx()
        
    def _load_model_weights(self):
        """Load pre-trained model weights"""
        for model_name, model in self.models.items():
            try:
                model.load_state_dict(torch.load(f'models/{model_name}.pth', map_location=self.device))
                model.eval()
                logger.info(f"Loaded weights for {model_name}")
            except:
                logger.info(f"No pre-trained weights found for {model_name}")
                
    def _convert_to_onnx(self):
        """Convert models to ONNX format for CPU optimization"""
        for model_name, model in self.models.items():
            try:
                # Create dummy input
                if 'gru' in model_name or 'lstm' in model_name:
                    dummy_input = torch.randn(1, 60, 50)  # batch_size, sequence_length, features
                elif 'momentum' in model_name:
                    dummy_input = torch.randn(1, 10, 50)
                else:  # meta_learner
                    dummy_input = [torch.randn(1, 3) for _ in range(3)]
                    dummy_input = torch.cat(dummy_input, dim=1)
                
                # Export to ONNX
                onnx_path = f'models/{model_name}.onnx'
                torch.onnx.export(model, dummy_input, onnx_path,
                                 export_params=True,
                                 opset_version=11,
                                 do_constant_folding=True,
                                 input_names=['input'],
                                 output_names=['output'])
                
                # Create ONNX session
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.intra_op_num_threads = mp.cpu_count()
                
                self.onnx_sessions[model_name] = ort.InferenceSession(onnx_path, sess_options)
                logger.info(f"Converted {model_name} to ONNX")
                
            except Exception as e:
                logger.error(f"Error converting {model_name} to ONNX: {e}")
                
    def predict(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading prediction for a symbol"""
        
        # Feature engineering
        features_df = self.feature_engineer.create_features(market_data)
        
        # Create sequences
        sequences, _ = self.feature_engineer.create_sequences(features_df)
        
        if len(sequences) == 0:
            return {'action': 'hold', 'confidence': 0.0}
        
        # Get latest sequence
        latest_sequence = sequences[-1:].astype(np.float32)
        
        # Get predictions from each model
        predictions = []
        
        # GRU short-term prediction
        if 'gru_short' in self.onnx_sessions:
            gru_pred = self.onnx_sessions['gru_short'].run(
                None, {'input': latest_sequence})[0]
            predictions.append(gru_pred)
        
        # LSTM long-term prediction  
        if 'lstm_long' in self.onnx_sessions:
            lstm_pred = self.onnx_sessions['lstm_long'].run(
                None, {'input': latest_sequence})[0]
            predictions.append(lstm_pred)
            
        # Momentum prediction (last 10 timesteps)
        if 'momentum' in self.onnx_sessions:
            momentum_sequence = latest_sequence[:, -10:, :].astype(np.float32)
            momentum_pred = self.onnx_sessions['momentum'].run(
                None, {'input': momentum_sequence})[0]
            predictions.append(momentum_pred)
            
        # Ensemble prediction
        if len(predictions) >= 2 and 'meta_learner' in self.onnx_sessions:
            ensemble_input = np.concatenate(predictions, axis=1).astype(np.float32)
            final_pred = self.onnx_sessions['meta_learner'].run(
                None, {'input': ensemble_input})[0]
        else:
            # Average predictions if meta learner not available
            final_pred = np.mean(predictions, axis=0)
            
        # Get action and confidence
        action_idx = np.argmax(final_pred[0])
        confidence = float(final_pred[0][action_idx])
        
        actions = ['buy', 'hold', 'sell']
        action = actions[action_idx]
        
        # DQN recommendation
        state = self._create_state_vector(features_df)
        dqn_action = self.dqn_agent.act(state)
        dqn_confidence = 0.7  # Fixed confidence for DQN
        
        # Combine predictions
        if confidence < Config.CONFIDENCE_THRESHOLD and dqn_confidence >= Config.CONFIDENCE_THRESHOLD:
            action = actions[dqn_action]
            confidence = dqn_confidence
            
        return {
            'action': action,
            'confidence': confidence,
            'models_agree': len(set([np.argmax(p[0]) for p in predictions])) == 1,
            'timestamp': datetime.now()
        }
        
    def _create_state_vector(self, features_df: pd.DataFrame, state_size: int = 50) -> np.ndarray:
        """Create state vector for DQN"""
        # Use last row of features
        last_features = features_df.iloc[-1]
        
        # Select most important features
        important_features = [
            'returns', 'rsi', 'macd_diff', 'bb_position', 'volume_ratio',
            'order_flow_imbalance', 'volatility', 'atr', 'momentum_5', 'momentum_10'
        ]
        
        state = []
        for feature in important_features[:state_size]:
            if feature in last_features:
                state.append(float(last_features[feature]))
                
        # Pad if necessary
        while len(state) < state_size:
            state.append(0.0)
            
        return np.array(state)
    
    def update_models(self, trading_results: List[Dict]):
        """Update models with recent trading results"""
        # Update DQN with experiences
        for result in trading_results:
            if 'state' in result and 'action' in result and 'reward' in result:
                self.dqn_agent.remember(
                    result['state'],
                    result['action'],
                    result['reward'],
                    result['next_state'],
                    result['done']
                )
                
        # Train DQN
        if len(self.dqn_agent.memory) > 1000:
            self.dqn_agent.replay(batch_size=32)
            
        # Update target network periodically
        if len(trading_results) % 100 == 0:
            self.dqn_agent.update_target_model()
            
        # Track performance
        for result in trading_results:
            self.performance_tracker.append({
                'timestamp': result.get('timestamp', datetime.now()),
                'profit': result.get('profit', 0),
                'accuracy': result.get('prediction_correct', False)
            })
            
        # Retrain models if performance drops
        if len(self.performance_tracker) >= 100:
            recent_accuracy = sum([p['accuracy'] for p in list(self.performance_tracker)[-100:]]) / 100
            if recent_accuracy < 0.55:  # Below 55% accuracy
                logger.warning(f"Model accuracy dropped to {recent_accuracy:.2%}, considering retraining")
                
# ==================== RISK MANAGER ====================

class ExtremeRiskManager:
    """Manages extreme risk parameters (40% per trade, 10-20x leverage)"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_starting_capital = initial_capital
        self.positions = {}
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.correlation_matrix = pd.DataFrame()
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, market_conditions: Dict) -> Dict:
        """Calculate position size for extreme risk tolerance"""
        
        # Base risk amount (40% of capital)
        risk_percent = Config.MAX_RISK_PER_TRADE
        
        # Adjust for market conditions
        if market_conditions.get('volatility_ratio', 1.0) > 1.5:
            risk_percent *= 0.6  # Reduce to 24% in high volatility
            
        if market_conditions.get('correlation_spike', False):
            risk_percent *= 0.5  # Reduce to 20% if correlations are high
            
        # Calculate risk amount
        risk_amount = self.current_capital * risk_percent
        
        # Calculate position details
        price_distance = abs(entry_price - stop_loss_price) / entry_price
        
        # Determine leverage based on volatility
        volatility = market_conditions.get('atr', 0) / entry_price
        if volatility > 0.02:  # >2% ATR
            leverage = Config.MIN_LEVERAGE
        elif volatility < 0.01:  # <1% ATR
            leverage = Config.MAX_LEVERAGE
        else:
            leverage = Config.DEFAULT_LEVERAGE
            
        # Calculate position size
        position_size = risk_amount / price_distance
        required_margin = position_size / leverage
        
        # Check if we have enough margin
        available_margin = self._get_available_margin()
        if required_margin > available_margin:
            # Scale down position
            scale_factor = available_margin / required_margin
            position_size *= scale_factor
            required_margin = available_margin
            
        return {
            'position_size': position_size,
            'contracts': position_size / entry_price,
            'leverage': leverage,
            'required_margin': required_margin,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent
        }
        
    def check_risk_limits(self, symbol: str, proposed_position: Dict) -> Tuple[bool, str]:
        """Check if position meets risk criteria"""
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_starting_capital * Config.MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
            
        # Check maximum positions
        if len(self.positions) >= Config.MAX_POSITIONS:
            return False, "Maximum positions reached"
            
        # Check correlation limits
        if self._check_correlation_risk(symbol):
            return False, "Correlation risk too high"
            
        # Check available margin
        if proposed_position['required_margin'] > self._get_available_margin():
            return False, "Insufficient margin"
            
        return True, "OK"
        
    def _get_available_margin(self) -> float:
        """Calculate available margin"""
        used_margin = sum(pos.get('margin', 0) for pos in self.positions.values())
        return self.current_capital - used_margin
        
    def _check_correlation_risk(self, symbol: str) -> bool:
        """Check if adding position increases correlation risk"""
        if len(self.positions) == 0:
            return False
            
        # This would check correlation matrix
        # Simplified for now
        return False
        
    def update_position(self, symbol: str, position_data: Dict):
        """Update position tracking"""
        self.positions[symbol] = position_data
        
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """Close position and calculate P&L"""
        if symbol not in self.positions:
            return {'error': 'Position not found'}
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        contracts = position['contracts']
        side = position['side']
        
        # Calculate P&L
        if side == 'buy':
            pnl = (exit_price - entry_price) * contracts
        else:
            pnl = (entry_price - exit_price) * contracts
            
        pnl_percent = pnl / position['margin']
        
        # Update tracking
        self.daily_pnl += pnl
        self.current_capital += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            
        # Remove position
        del self.positions[symbol]
        
        return {
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_price': exit_price,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }
        
    def check_circuit_breaker(self, market_data: Dict) -> bool:
        """Check if circuit breaker should trigger"""
        
        # Check rapid market movement
        for symbol, data in market_data.items():
            if 'price_change_5m' in data:
                if abs(data['price_change_5m']) > Config.CIRCUIT_BREAKER_THRESHOLD:
                    logger.warning(f"Circuit breaker triggered: {symbol} moved {data['price_change_5m']:.2%}")
                    return True
                    
        # Check account drawdown
        current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        if current_drawdown > 0.25:  # 25% drawdown
            logger.warning(f"Circuit breaker triggered: Account drawdown {current_drawdown:.2%}")
            return True
            
        return False
        
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0
        self.daily_starting_capital = self.current_capital

# ==================== ORDER MANAGER ====================

class HighFrequencyOrderManager:
    """Manages order execution with ultra-low latency"""
    
    def __init__(self, exchange, risk_manager):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.active_orders = {}
        self.execution_times = deque(maxlen=1000)
        
    async def execute_signal(self, symbol: str, signal: Dict, market_data: Dict) -> Dict:
        """Execute trading signal with <10ms latency target"""
        
        start_time = time.time()
        
        try:
            # Get current market price
            ticker = await self._get_ticker_async(symbol)
            current_price = ticker['last']
            
            # Calculate entry and exit prices
            if signal['action'] == 'buy':
                entry_price = ticker['ask']  # Market buy at ask
                stop_loss = entry_price * 0.95  # 5% stop loss
                take_profit = entry_price * 1.002  # 0.2% take profit
            elif signal['action'] == 'sell':
                entry_price = ticker['bid']  # Market sell at bid
                stop_loss = entry_price * 1.05  # 5% stop loss
                take_profit = entry_price * 0.998  # 0.2% take profit
            else:
                return {'status': 'skipped', 'reason': 'hold signal'}
                
            # Calculate position size
            position_details = self.risk_manager.calculate_position_size(
                symbol, entry_price, stop_loss, market_data
            )
            
            # Check risk limits
            can_trade, reason = self.risk_manager.check_risk_limits(symbol, position_details)
            if not can_trade:
                return {'status': 'rejected', 'reason': reason}
                
            # Place market order
            order_params = {
                'symbol': symbol,
                'type': 'market',
                'side': signal['action'],
                'amount': position_details['contracts'],
                'params': {
                    'leverage': position_details['leverage'],
                    'timeInForce': 'IOC'  # Immediate or cancel
                }
            }
            
            order_result = await self._place_order_async(order_params)
            
            if order_result['status'] == 'closed':
                # Order filled, place stop loss and take profit
                sl_order = await self._place_stop_loss_async(
                    symbol, signal['action'], position_details['contracts'], stop_loss
                )
                
                tp_order = await self._place_take_profit_async(
                    symbol, signal['action'], position_details['contracts'], take_profit
                )
                
                # Update risk manager
                self.risk_manager.update_position(symbol, {
                    'entry_price': order_result['average'],
                    'contracts': position_details['contracts'],
                    'side': signal['action'],
                    'margin': position_details['required_margin'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'sl_order_id': sl_order['id'],
                    'tp_order_id': tp_order['id'],
                    'timestamp': datetime.now()
                })
                
                # Track execution time
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                self.execution_times.append(execution_time)
                
                return {
                    'status': 'executed',
                    'order_id': order_result['id'],
                    'entry_price': order_result['average'],
                    'contracts': position_details['contracts'],
                    'execution_time_ms': execution_time
                }
                
        except Exception as e:
            logger.error(f"Order execution error for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
            
    async def _get_ticker_async(self, symbol: str) -> Dict:
        """Get ticker data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.exchange.fetch_ticker, symbol)
        
    async def _place_order_async(self, order_params: Dict) -> Dict:
        """Place order asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.exchange.create_order(**order_params)
        )
        
    async def _place_stop_loss_async(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict:
        """Place stop loss order"""
        sl_side = 'sell' if side == 'buy' else 'buy'
        
        order_params = {
            'symbol': symbol,
            'type': 'stop',
            'side': sl_side,
            'amount': amount,
            'stopPrice': stop_price,
            'params': {
                'reduceOnly': True,
                'timeInForce': 'GTC'
            }
        }
        
        return await self._place_order_async(order_params)
        
    async def _place_take_profit_async(self, symbol: str, side: str, amount: float, limit_price: float) -> Dict:
        """Place take profit order"""
        tp_side = 'sell' if side == 'buy' else 'buy'
        
        order_params = {
            'symbol': symbol,
            'type': 'limit',
            'side': tp_side,
            'amount': amount,
            'price': limit_price,
            'params': {
                'reduceOnly': True,
                'timeInForce': 'GTC'
            }
        }
        
        return await self._place_order_async(order_params)
        
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_times:
            return {}
            
        times = list(self.execution_times)
        return {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'p99_ms': np.percentile(times, 99),
            'below_10ms_pct': sum(1 for t in times if t < 10) / len(times) * 100
        }

# ==================== MAIN TRADING BOT ====================

class UltraHighPerformanceTradingBot:
    """Main orchestrator for the trading system"""
    
    def __init__(self):
        # Initialize components
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB
        )
        
        self.market_data_manager = MarketDataManager(self.redis_client)
        self.model_manager = ModelManager()
        self.risk_manager = ExtremeRiskManager(initial_capital=10000)  # $10k starting capital
        self.order_manager = None  # Initialized after exchange
        
        self.is_running = False
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'start_time': datetime.now()
        }
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Ultra High-Performance Trading Bot...")
        
        # Initialize exchange
        self.market_data_manager.initialize_exchange()
        self.order_manager = HighFrequencyOrderManager(
            self.market_data_manager.exchange,
            self.risk_manager
        )
        
        # Initialize models
        self.model_manager.initialize_models()
        
        # Start market data feeds
        self.market_data_manager.start_websocket_feeds()
        
        logger.info("Initialization complete!")
        
    async def run(self):
        """Main trading loop"""
        self.is_running = True
        
        # Reset daily stats at midnight
        last_reset = datetime.now().date()
        
        while self.is_running:
            try:
                # Check if we need to reset daily stats
                if datetime.now().date() > last_reset:
                    self.risk_manager.reset_daily_stats()
                    last_reset = datetime.now().date()
                    
                # Check circuit breaker
                market_conditions = self._analyze_market_conditions()
                if self.risk_manager.check_circuit_breaker(market_conditions):
                    logger.warning("Circuit breaker active, pausing trading for 5 minutes")
                    await asyncio.sleep(300)
                    continue
                    
                # Process each trading pair
                tasks = []
                for symbol in Config.TRADING_PAIRS:
                    task = self._process_symbol(symbol, market_conditions)
                    tasks.append(task)
                    
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks)
                
                # Update model with results
                valid_results = [r for r in results if r and 'status' in r and r['status'] == 'executed']
                if valid_results:
                    self.model_manager.update_models(valid_results)
                    
                # Log performance
                self._log_performance()
                
                # Sleep briefly before next iteration
                await asyncio.sleep(1)  # 1 second between cycles for 1-minute candles
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
                
        self.shutdown()
        
    async def _process_symbol(self, symbol: str, market_conditions: Dict) -> Optional[Dict]:
        """Process trading logic for a symbol"""
        
        try:
            # Get historical data
            df = self.market_data_manager.get_historical_data(symbol, Config.PRIMARY_TIMEFRAME, limit=200)
            
            if df.empty or len(df) < 100:
                return None
                
            # Get prediction
            prediction = self.model_manager.predict(symbol, df)
            
            # Check confidence threshold
            if prediction['confidence'] < Config.CONFIDENCE_THRESHOLD:
                return None
                
            # Check if we already have a position
            if symbol in self.risk_manager.positions:
                # Check for exit signals
                position = self.risk_manager.positions[symbol]
                
                # Simple exit logic - could be enhanced
                current_price = df['close'].iloc[-1]
                entry_price = position['entry_price']
                
                if position['side'] == 'buy':
                    current_pnl = (current_price - entry_price) / entry_price
                else:
                    current_pnl = (entry_price - current_price) / entry_price
                    
                # Exit if target reached or stop hit
                if current_pnl > 0.002 or current_pnl < -0.05:
                    result = self.risk_manager.close_position(symbol, current_price)
                    self.trading_stats['total_pnl'] += result['pnl']
                    return result
                    
                return None
                
            # Execute signal if no position
            if prediction['action'] in ['buy', 'sell']:
                result = await self.order_manager.execute_signal(
                    symbol, prediction, market_conditions
                )
                
                if result['status'] == 'executed':
                    self.trading_stats['total_trades'] += 1
                    
                return result
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
            
    def _analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions"""
        
        conditions = {}
        
        try:
            # Get BTC data as market proxy
            btc_df = self.market_data_manager.get_historical_data('BTC/USDT:USDT', '5m', limit=100)
            
            if not btc_df.empty:
                # Calculate volatility
                returns = btc_df['close'].pct_change()
                current_volatility = returns.tail(20).std()
                avg_volatility = returns.std()
                
                conditions['volatility_ratio'] = current_volatility / avg_volatility
                conditions['atr'] = ta.volatility.AverageTrueRange(
                    btc_df['high'], btc_df['low'], btc_df['close']
                ).average_true_range().iloc[-1]
                
                # Price change
                conditions['price_change_5m'] = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-6]) / btc_df['close'].iloc[-6]
                
                # Check correlation spike
                # Simplified - would calculate actual correlations between pairs
                conditions['correlation_spike'] = False
                
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            
        return conditions
        
    def _log_performance(self):
        """Log current performance metrics"""
        
        # Get execution stats
        exec_stats = self.order_manager.get_execution_stats()
        
        # Calculate metrics
        runtime = (datetime.now() - self.trading_stats['start_time']).total_seconds() / 3600
        win_rate = self.risk_manager.winning_trades / max(self.risk_manager.total_trades, 1)
        
        logger.info(f"""
        ===== PERFORMANCE UPDATE =====
        Runtime: {runtime:.2f} hours
        Total Trades: {self.trading_stats['total_trades']}
        Win Rate: {win_rate:.2%}
        Total P&L: ${self.trading_stats['total_pnl']:.2f}
        Current Capital: ${self.risk_manager.current_capital:.2f}
        Daily P&L: ${self.risk_manager.daily_pnl:.2f}
        Active Positions: {len(self.risk_manager.positions)}
        
        Execution Stats:
        - Mean: {exec_stats.get('mean_ms', 0):.2f}ms
        - P99: {exec_stats.get('p99_ms', 0):.2f}ms
        - <10ms: {exec_stats.get('below_10ms_pct', 0):.1f}%
        =============================
        """)
        
    def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down trading bot...")
        self.is_running = False
        
        # Close all positions
        for symbol in list(self.risk_manager.positions.keys()):
            try:
                df = self.market_data_manager.get_historical_data(symbol, '1m', limit=1)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    self.risk_manager.close_position(symbol, current_price)
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
                
        # Save model weights
        for name, model in self.model_manager.models.items():
            try:
                torch.save(model.state_dict(), f'models/{name}_final.pth')
            except Exception as e:
                logger.error(f"Error saving model {name}: {e}")
                
        logger.info("Shutdown complete")

# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point"""
    
    # Create bot instance
    bot = UltraHighPerformanceTradingBot()
    
    # Initialize
    bot.initialize()
    
    # Wait for data to accumulate
    logger.info("Waiting 10 seconds for initial data...")
    await asyncio.sleep(10)
    
    # Run bot
    await bot.run()

if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the bot
    asyncio.run(main())