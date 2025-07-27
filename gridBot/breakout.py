import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MomentumConfig:
    """Configuration for high-risk momentum trading"""
    api_key: str
    api_secret: str
    
    # Risk Parameters
    leverage: int = 10  # Starting leverage
    max_leverage: int = 25  # Maximum leverage for strongest signals
    position_size: float = 0.2  # 20% of capital per trade
    max_positions: int = 3  # Maximum concurrent positions
    stop_loss: float = 0.03  # 3% stop loss (tight)
    
    # Momentum Parameters
    volume_spike_threshold: float = 3.0  # 3x average volume
    min_price_change: float = 0.02  # Minimum 2% move to consider
    momentum_lookback: int = 20  # Candles for momentum calculation
    
    # Profit Taking
    take_profit_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])  # 5%, 10%, 20%
    take_profit_percentages: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])  # Take 30%, 30%, 40%
    trailing_stop_activation: float = 0.05  # Activate trailing stop at 5% profit
    trailing_stop_distance: float = 0.02  # Trail by 2%
    
    # Scanner Settings
    symbols_to_scan: List[str] = field(default_factory=lambda: [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT',
        'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'FTM/USDT',
        'NEAR/USDT', 'ALGO/USDT', 'FIL/USDT', 'ICP/USDT', 'SAND/USDT',
        'MANA/USDT', 'AXS/USDT', 'GALA/USDT', 'APE/USDT', 'OP/USDT',
        'ARB/USDT', 'INJ/USDT', 'SUI/USDT', 'SEI/USDT', 'TIA/USDT'
    ])
    
    # Risk Limits
    max_daily_loss: float = 0.15  # 15% max daily loss
    max_daily_trades: int = 10  # Maximum trades per day
    
    testnet: bool = True  # Start with testnet

@dataclass
class MomentumSignal:
    """Data class for momentum signals"""
    symbol: str
    strength: float  # 0-100 score
    volume_ratio: float
    price_change: float
    timeframe_alignment: Dict[str, bool]
    entry_price: float
    suggested_leverage: int
    timestamp: datetime

class MomentumScanner:
    """Scans multiple symbols for momentum breakouts"""
    
    def __init__(self, exchange: ccxt.Exchange, config: MomentumConfig):
        self.exchange = exchange
        self.config = config
        self.volume_history = {symbol: deque(maxlen=100) for symbol in config.symbols_to_scan}
        self.price_history = {symbol: {} for symbol in config.symbols_to_scan}
        
    async def scan_symbol(self, symbol: str) -> Optional[MomentumSignal]:
        """Scan a single symbol for momentum"""
        try:
            # Fetch data for multiple timeframes
            timeframes = ['5m', '15m', '1h']
            momentum_scores = {}
            
            for tf in timeframes:
                df = await self.fetch_ohlcv_async(symbol, tf, limit=50)
                if df.empty:
                    continue
                
                # Calculate momentum indicators
                momentum_data = self.calculate_momentum(df)
                momentum_scores[tf] = momentum_data
            
            # Check if we have a breakout
            signal = self.evaluate_breakout(symbol, momentum_scores)
            return signal
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    async def fetch_ohlcv_async(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Async fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return pd.DataFrame()
    
    def calculate_momentum(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        if len(df) < 20:
            return {}
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price breakout
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Momentum strength
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # RSI for overbought/oversold
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        return {
            'volume_ratio': latest['volume_ratio'],
            'price_change': df['close'].pct_change(periods=5).iloc[-1],
            'price_position': latest['price_position'],
            'momentum': latest['momentum'],
            'rsi': latest['rsi'],
            'is_breakout': latest['close'] >= latest['high_20'],
            'volume_increasing': df['volume'].iloc[-3:].mean() > df['volume_sma'].iloc[-1]
        }
    
    def evaluate_breakout(self, symbol: str, momentum_scores: Dict) -> Optional[MomentumSignal]:
        """Evaluate if we have a strong breakout signal"""
        if not momentum_scores:
            return None
        
        # Check 15m timeframe as primary
        primary_tf = momentum_scores.get('15m', {})
        if not primary_tf:
            return None
        
        # Breakout criteria
        volume_spike = primary_tf.get('volume_ratio', 0) >= self.config.volume_spike_threshold
        price_breakout = primary_tf.get('is_breakout', False)
        momentum_positive = primary_tf.get('momentum', 0) > self.config.min_price_change
        not_overbought = primary_tf.get('rsi', 50) < 80
        
        if not (volume_spike and price_breakout and momentum_positive and not_overbought):
            return None
        
        # Calculate signal strength (0-100)
        strength = 0
        
        # Volume component (0-40 points)
        volume_score = min(primary_tf.get('volume_ratio', 0) / 5, 1) * 40
        strength += volume_score
        
        # Momentum component (0-30 points)
        momentum_score = min(primary_tf.get('momentum', 0) / 0.10, 1) * 30
        strength += momentum_score
        
        # Timeframe alignment (0-30 points)
        timeframe_alignment = {}
        alignment_score = 0
        
        for tf, data in momentum_scores.items():
            if data.get('momentum', 0) > 0 and data.get('volume_ratio', 1) > 1.5:
                timeframe_alignment[tf] = True
                alignment_score += 10
            else:
                timeframe_alignment[tf] = False
        
        strength += alignment_score
        
        # Determine leverage based on signal strength
        if strength >= 80:
            suggested_leverage = self.config.max_leverage
        elif strength >= 60:
            suggested_leverage = 15
        else:
            suggested_leverage = self.config.leverage
        
        return MomentumSignal(
            symbol=symbol,
            strength=strength,
            volume_ratio=primary_tf.get('volume_ratio', 0),
            price_change=primary_tf.get('momentum', 0),
            timeframe_alignment=timeframe_alignment,
            entry_price=0,  # Will be set when placing order
            suggested_leverage=suggested_leverage,
            timestamp=datetime.now()
        )

class PositionManager:
    """Manages open positions with trailing stops and profit taking"""
    
    def __init__(self, exchange: ccxt.Exchange, config: MomentumConfig):
        self.exchange = exchange
        self.config = config
        self.positions = {}
        self.daily_pnl = 0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
    
    def add_position(self, symbol: str, order: Dict, signal: MomentumSignal):
        """Add a new position to track"""
        self.positions[symbol] = {
            'entry_price': order['price'],
            'amount': order['amount'],
            'leverage': signal.suggested_leverage,
            'stop_loss': order['price'] * (1 - self.config.stop_loss),
            'take_profit_levels': [
                order['price'] * (1 + tp) for tp in self.config.take_profit_levels
            ],
            'take_profit_remaining': list(self.config.take_profit_percentages),
            'trailing_stop_active': False,
            'trailing_stop_price': None,
            'highest_price': order['price'],
            'entry_time': datetime.now(),
            'signal_strength': signal.strength
        }
        self.daily_trades += 1
    
    async def update_positions(self):
        """Update all positions with current prices"""
        for symbol, position in list(self.positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Update highest price
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                
                # Check for trailing stop activation
                profit_percent = (current_price - position['entry_price']) / position['entry_price']
                
                if profit_percent >= self.config.trailing_stop_activation and not position['trailing_stop_active']:
                    position['trailing_stop_active'] = True
                    position['trailing_stop_price'] = current_price * (1 - self.config.trailing_stop_distance)
                    logger.info(f"{symbol}: Trailing stop activated at {position['trailing_stop_price']:.2f}")
                
                # Update trailing stop
                if position['trailing_stop_active']:
                    new_stop = current_price * (1 - self.config.trailing_stop_distance)
                    if new_stop > position['trailing_stop_price']:
                        position['trailing_stop_price'] = new_stop
                
                # Check stop conditions
                should_close = False
                close_reason = ""
                
                # Stop loss
                if current_price <= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop loss hit"
                
                # Trailing stop
                elif position['trailing_stop_active'] and current_price <= position['trailing_stop_price']:
                    should_close = True
                    close_reason = "Trailing stop hit"
                
                # Time-based exit (optional)
                elif (datetime.now() - position['entry_time']).total_seconds() > 86400:  # 24 hours
                    should_close = True
                    close_reason = "Time limit reached"
                
                if should_close:
                    await self.close_position(symbol, close_reason)
                else:
                    # Check for partial profit taking
                    await self.check_profit_targets(symbol, current_price)
                
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    async def check_profit_targets(self, symbol: str, current_price: float):
        """Check and execute partial profit taking"""
        position = self.positions[symbol]
        
        for i, (tp_level, tp_percent) in enumerate(zip(position['take_profit_levels'], position['take_profit_remaining'])):
            if current_price >= tp_level and tp_percent > 0:
                # Calculate amount to sell
                sell_amount = position['amount'] * tp_percent
                
                try:
                    order = self.exchange.create_market_sell_order(
                        symbol=symbol,
                        amount=sell_amount
                    )
                    
                    position['take_profit_remaining'][i] = 0
                    position['amount'] -= sell_amount
                    
                    logger.info(f"{symbol}: Took {tp_percent*100}% profit at level {i+1} ({current_price:.2f})")
                    
                    # If all profit targets hit, close remaining
                    if all(tp == 0 for tp in position['take_profit_remaining']):
                        await self.close_position(symbol, "All profit targets hit")
                        
                except Exception as e:
                    logger.error(f"Error taking profit for {symbol}: {e}")
    
    async def close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Market sell
            order = self.exchange.create_market_sell_order(
                symbol=symbol,
                amount=position['amount']
            )
            
            # Calculate PnL
            exit_price = order['price']
            pnl_percent = ((exit_price - position['entry_price']) / position['entry_price']) * position['leverage']
            pnl_amount = pnl_percent * (position['amount'] * position['entry_price'])
            
            self.daily_pnl += pnl_amount
            
            logger.info(f"""
Position Closed: {symbol}
Reason: {reason}
Entry: {position['entry_price']:.2f}
Exit: {exit_price:.2f}
PnL: {pnl_percent*100:.2f}% (${pnl_amount:.2f})
Duration: {datetime.now() - position['entry_time']}
            """)
            
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
    
    def check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Reset daily counters
        if datetime.now().date() > self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.last_reset = datetime.now().date()
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss * self.get_account_balance():
            logger.warning("Daily loss limit reached")
            return False
        
        # Check trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check position limit
        if len(self.positions) >= self.config.max_positions:
            return False
        
        return True
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free'] + balance['USDT']['used']
        except:
            return 10000  # Default for testing

class MomentumBreakoutBot:
    """Main bot coordinating scanning and position management"""
    
    def __init__(self, config: MomentumConfig):
        self.config = config
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear'
            }
        })
        
        if config.testnet:
            self.exchange.set_sandbox_mode(True)
        
        self.scanner = MomentumScanner(self.exchange, config)
        self.position_manager = PositionManager(self.exchange, config)
        self.running = False
        self.scan_interval = 30  # Scan every 30 seconds
        
    async def execute_signal(self, signal: MomentumSignal):
        """Execute a trading signal"""
        try:
            # Double-check risk limits
            if not self.position_manager.check_risk_limits():
                return
            
            # Get current price
            ticker = self.exchange.fetch_ticker(signal.symbol)
            current_price = ticker['last']
            signal.entry_price = current_price
            
            # Calculate position size
            balance = self.position_manager.get_account_balance()
            position_value = balance * self.config.position_size
            amount = (position_value * signal.suggested_leverage) / current_price
            
            # Set leverage
            self.exchange.set_leverage(signal.suggested_leverage, signal.symbol)
            
            # Place order
            order = self.exchange.create_market_buy_order(
                symbol=signal.symbol,
                amount=amount,
                params={
                    'stopLoss': current_price * (1 - self.config.stop_loss)
                }
            )
            
            # Add to position manager
            self.position_manager.add_position(signal.symbol, order, signal)
            
            logger.info(f"""
🚀 NEW POSITION OPENED
Symbol: {signal.symbol}
Signal Strength: {signal.strength:.1f}/100
Entry Price: {current_price:.2f}
Leverage: {signal.suggested_leverage}x
Position Size: ${position_value:.2f}
Volume Spike: {signal.volume_ratio:.1f}x
Price Change: {signal.price_change*100:.2f}%
Timeframes Aligned: {sum(signal.timeframe_alignment.values())}/3
            """)
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def scanning_loop(self):
        """Main scanning loop"""
        while self.running:
            try:
                # Scan all symbols concurrently
                tasks = [self.scanner.scan_symbol(symbol) for symbol in self.config.symbols_to_scan]
                signals = await asyncio.gather(*tasks)
                
                # Filter out None values and sort by strength
                valid_signals = [s for s in signals if s is not None]
                valid_signals.sort(key=lambda x: x.strength, reverse=True)
                
                # Execute the strongest signal if any
                if valid_signals and valid_signals[0].strength >= 60:
                    # Check if we already have a position in this symbol
                    if valid_signals[0].symbol not in self.position_manager.positions:
                        await self.execute_signal(valid_signals[0])
                
                # Log scanning status
                if valid_signals:
                    logger.info(f"Scan complete. Top signal: {valid_signals[0].symbol} ({valid_signals[0].strength:.1f}/100)")
                
            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
            
            await asyncio.sleep(self.scan_interval)
    
    async def position_management_loop(self):
        """Position management loop"""
        while self.running:
            try:
                await self.position_manager.update_positions()
                
                # Log current status
                if self.position_manager.positions:
                    logger.info(f"""
📊 POSITION UPDATE
Active Positions: {len(self.position_manager.positions)}
Daily PnL: ${self.position_manager.daily_pnl:.2f}
Daily Trades: {self.position_manager.daily_trades}/{self.config.max_daily_trades}
                    """)
                
            except Exception as e:
                logger.error(f"Error in position management: {e}")
            
            await asyncio.sleep(5)  # Update every 5 seconds
    
    async def run(self):
        """Run the bot"""
        self.running = True
        logger.info("""
🔥 HIGH-RISK MOMENTUM BREAKOUT BOT STARTED 🔥
Leverage: {}-{}x
Stop Loss: {}%
Scanning {} symbols
Max Daily Loss: {}%
        """.format(
            self.config.leverage,
            self.config.max_leverage,
            self.config.stop_loss * 100,
            len(self.config.symbols_to_scan),
            self.config.max_daily_loss * 100
        ))
        
        # Run both loops concurrently
        await asyncio.gather(
            self.scanning_loop(),
            self.position_management_loop()
        )
    
    def start(self):
        """Start the bot"""
        asyncio.run(self.run())
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        logger.info("Shutting down bot...")
        
        # Close all positions
        for symbol in list(self.position_manager.positions.keys()):
            asyncio.run(self.position_manager.close_position(symbol, "Bot shutdown"))

# Example usage
if __name__ == "__main__":
    # WARNING: This is a HIGH RISK bot. Only use money you can afford to lose!
    
    config = MomentumConfig(
        api_key="YOUR_BYBIT_API_KEY",
        api_secret="YOUR_BYBIT_API_SECRET",
        
        # Start conservative, increase as you gain confidence
        leverage=10,
        max_leverage=25,
        position_size=0.2,  # 20% of capital per trade
        
        # Tight risk management is CRUCIAL
        stop_loss=0.03,  # 3% stop loss
        max_daily_loss=0.15,  # 15% daily loss limit
        
        testnet=True  # ALWAYS start with testnet!
    )
    
    bot = MomentumBreakoutBot(config)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()