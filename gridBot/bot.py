import ccxt
import pandas as pd
import numpy as np
import time
import json
import websocket
import threading
from datetime import datetime, timedelta
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional
import hashlib
import hmac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderFlowAnalyzer:
    """Analyze real-time order flow and market microstructure"""
    
    def __init__(self):
        self.order_book = {'bids': {}, 'asks': {}}
        self.trades = deque(maxlen=10000)
        self.imbalances = deque(maxlen=1000)
        self.aggressive_orders = {'buy': 0, 'sell': 0}
        self.passive_orders = {'buy': 0, 'sell': 0}
        self.footprint = {}
        
    def update_orderbook(self, data: Dict) -> None:
        """Update order book with real-time data"""
        # Process bids
        if 'bids' in data:
            for bid in data['bids']:
                if isinstance(bid, list) and len(bid) >= 2:
                    price, size = float(bid[0]), float(bid[1])
                elif isinstance(bid, str):
                    # Sometimes Bybit sends strings, need to parse
                    parts = bid.split(',')
                    if len(parts) >= 2:
                        price, size = float(parts[0]), float(parts[1])
                    else:
                        continue
                else:
                    continue
                    
                if size > 0:
                    self.order_book['bids'][price] = size
                elif price in self.order_book['bids']:
                    del self.order_book['bids'][price]
                    
        # Process asks
        if 'asks' in data:
            for ask in data['asks']:
                if isinstance(ask, list) and len(ask) >= 2:
                    price, size = float(ask[0]), float(ask[1])
                elif isinstance(ask, str):
                    # Sometimes Bybit sends strings, need to parse
                    parts = ask.split(',')
                    if len(parts) >= 2:
                        price, size = float(parts[0]), float(parts[1])
                    else:
                        continue
                else:
                    continue
                    
                if size > 0:
                    self.order_book['asks'][price] = size
                elif price in self.order_book['asks']:
                    del self.order_book['asks'][price]
    
    def analyze_trade(self, trade: Dict) -> None:
        """Analyze individual trade for aggression"""
        price = float(trade['price'])
        size = float(trade['size'])
        side = trade['side']
        
        # Store trade
        self.trades.append({
            'price': price,
            'size': size,
            'side': side,
            'timestamp': trade.get('timestamp', int(time.time() * 1000))
        })
        
        # Determine if aggressive or passive
        best_bid = max(self.order_book['bids'].keys()) if self.order_book['bids'] else 0
        best_ask = min(self.order_book['asks'].keys()) if self.order_book['asks'] else float('inf')
        
        if side == 'buy' and price >= best_ask:
            self.aggressive_orders['buy'] += size
        elif side == 'sell' and price <= best_bid:
            self.aggressive_orders['sell'] += size
        else:
            self.passive_orders[side] += size
    
    def calculate_order_flow_imbalance(self) -> float:
        """Calculate order flow imbalance"""
        if not self.trades:
            return 0
        
        recent_trades = list(self.trades)[-100:]  # Last 100 trades
        buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'buy')
        sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'sell')
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
        
        imbalance = (buy_volume - sell_volume) / total_volume
        self.imbalances.append(imbalance)
        
        return imbalance
    
    def get_cumulative_delta(self) -> float:
        """Calculate cumulative delta volume"""
        if not self.trades:
            return 0
        
        delta = 0
        for trade in self.trades:
            if trade['side'] == 'buy':
                delta += trade['size']
            else:
                delta -= trade['size']
        
        return delta
    
    def detect_absorption(self) -> Dict[str, bool]:
        """Detect absorption patterns in order flow"""
        if len(self.trades) < 50:
            return {'buy_absorption': False, 'sell_absorption': False}
        
        recent_trades = list(self.trades)[-50:]
        
        # Group trades by price level
        price_levels = {}
        for trade in recent_trades:
            price = round(trade['price'], 2)
            if price not in price_levels:
                price_levels[price] = {'buy': 0, 'sell': 0}
            price_levels[price][trade['side']] += trade['size']
        
        # Check for absorption
        buy_absorption = False
        sell_absorption = False
        
        for price, volumes in price_levels.items():
            # Buy absorption: heavy selling but price doesn't drop
            if volumes['sell'] > volumes['buy'] * 2:
                current_price = recent_trades[-1]['price']
                if current_price >= price:
                    buy_absorption = True
            
            # Sell absorption: heavy buying but price doesn't rise
            if volumes['buy'] > volumes['sell'] * 2:
                current_price = recent_trades[-1]['price']
                if current_price <= price:
                    sell_absorption = True
        
        return {'buy_absorption': buy_absorption, 'sell_absorption': sell_absorption}
    
    def get_liquidity_levels(self) -> Dict[str, List[float]]:
        """Identify major liquidity levels from order book"""
        bid_liquidity = []
        ask_liquidity = []
        
        # Sort order book
        sorted_bids = sorted(self.order_book['bids'].items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.order_book['asks'].items(), key=lambda x: x[0])
        
        # Find large liquidity pools
        if sorted_bids:
            avg_bid_size = np.mean([size for _, size in sorted_bids[:20]])
            bid_liquidity = [price for price, size in sorted_bids if size > avg_bid_size * 3]
        
        if sorted_asks:
            avg_ask_size = np.mean([size for _, size in sorted_asks[:20]])
            ask_liquidity = [price for price, size in sorted_asks if size > avg_ask_size * 3]
        
        return {'bid_liquidity': bid_liquidity[:5], 'ask_liquidity': ask_liquidity[:5]}

class MarketMicrostructure:
    """Analyze market microstructure for institutional behavior"""
    
    def __init__(self):
        self.tick_data = deque(maxlen=1000)
        self.spread_history = deque(maxlen=1000)
        self.volume_clusters = {}
        
    def update_tick(self, price: float, volume: float, timestamp: int) -> None:
        """Update tick data"""
        self.tick_data.append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
    
    def calculate_vwap(self, period: int = 100) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(self.tick_data) < period:
            return 0
        
        recent_ticks = list(self.tick_data)[-period:]
        total_volume = sum(t['volume'] for t in recent_ticks)
        
        if total_volume == 0:
            return 0
        
        vwap = sum(t['price'] * t['volume'] for t in recent_ticks) / total_volume
        return vwap
    
    def detect_iceberg_orders(self, order_book: Dict) -> Dict[str, List[float]]:
        """Detect potential iceberg orders from repetitive small orders"""
        iceberg_levels = {'bids': [], 'asks': []}
        
        # Look for price levels with consistent refilling
        for side in ['bids', 'asks']:
            price_refills = {}
            
            # This would need historical order book data to properly implement
            # For now, we'll look for unusually consistent order sizes
            if side in order_book:
                sizes = list(order_book[side].values())
                if sizes:
                    median_size = np.median(sizes)
                    
                    for price, size in order_book[side].items():
                        # Iceberg orders often show as consistent medium sizes
                        if median_size * 0.8 <= size <= median_size * 1.2:
                            if price not in price_refills:
                                price_refills[price] = 0
                            price_refills[price] += 1
            
            # Prices that consistently refill might be icebergs
            iceberg_levels[side] = [p for p, count in price_refills.items() if count > 3]
        
        return iceberg_levels
    
    def calculate_spread_metrics(self, best_bid: float, best_ask: float) -> Dict[str, float]:
        """Calculate spread metrics"""
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000  # Basis points
        
        self.spread_history.append({
            'spread': spread,
            'spread_bps': spread_bps,
            'timestamp': time.time()
        })
        
        # Calculate average spread
        avg_spread = np.mean([s['spread'] for s in self.spread_history])
        current_vs_avg = spread / avg_spread if avg_spread > 0 else 1
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'avg_spread': avg_spread,
            'spread_ratio': current_vs_avg
        }

class InstitutionalStrategy:
    """Trading strategy based on institutional order flow"""
    
    def __init__(self):
        self.order_flow = OrderFlowAnalyzer()
        self.microstructure = MarketMicrostructure()
        self.position_tracker = {'long': 0, 'short': 0}
        self.risk_per_trade = 0.005  # 0.5% risk per trade
        
    def analyze_market_state(self, current_price: float) -> Dict:
        """Comprehensive market analysis"""
        # Order flow metrics
        flow_imbalance = self.order_flow.calculate_order_flow_imbalance()
        cumulative_delta = self.order_flow.get_cumulative_delta()
        absorption = self.order_flow.detect_absorption()
        liquidity_levels = self.order_flow.get_liquidity_levels()
        
        # Microstructure metrics
        vwap = self.microstructure.calculate_vwap()
        
        # Calculate aggressive ratio
        total_aggressive = self.order_flow.aggressive_orders['buy'] + self.order_flow.aggressive_orders['sell']
        aggressive_buy_ratio = self.order_flow.aggressive_orders['buy'] / total_aggressive if total_aggressive > 0 else 0.5
        
        return {
            'flow_imbalance': flow_imbalance,
            'cumulative_delta': cumulative_delta,
            'absorption': absorption,
            'liquidity_levels': liquidity_levels,
            'vwap': vwap,
            'aggressive_buy_ratio': aggressive_buy_ratio,
            'current_price': current_price
        }
    
    def generate_signal(self, analysis: Dict) -> Optional[Dict]:
        """Generate trading signals based on institutional flow"""
        current_price = analysis['current_price']
        
        # Strong bullish signal
        if (analysis['flow_imbalance'] > 0.3 and 
            analysis['cumulative_delta'] > 0 and
            analysis['absorption']['buy_absorption'] and
            analysis['aggressive_buy_ratio'] > 0.65):
            
            # Find stop loss at nearest bid liquidity
            bid_liquidity = analysis['liquidity_levels']['bid_liquidity']
            stop_loss = bid_liquidity[0] if bid_liquidity else current_price * 0.997
            
            return {
                'type': 'long',
                'entry': current_price,
                'stop_loss': stop_loss,
                'confidence': analysis['flow_imbalance'],
                'reason': 'Strong buy flow with absorption'
            }
        
        # Strong bearish signal
        elif (analysis['flow_imbalance'] < -0.3 and
              analysis['cumulative_delta'] < 0 and
              analysis['absorption']['sell_absorption'] and
              analysis['aggressive_buy_ratio'] < 0.35):
            
            # Find stop loss at nearest ask liquidity
            ask_liquidity = analysis['liquidity_levels']['ask_liquidity']
            stop_loss = ask_liquidity[0] if ask_liquidity else current_price * 1.003
            
            return {
                'type': 'short',
                'entry': current_price,
                'stop_loss': stop_loss,
                'confidence': abs(analysis['flow_imbalance']),
                'reason': 'Strong sell flow with absorption'
            }
        
        # VWAP reversion trade
        elif analysis['vwap'] > 0:
            vwap_distance = (current_price - analysis['vwap']) / analysis['vwap']
            
            # Long if price significantly below VWAP with positive delta
            if vwap_distance < -0.002 and analysis['cumulative_delta'] > 0:
                return {
                    'type': 'long',
                    'entry': current_price,
                    'stop_loss': current_price * 0.998,
                    'target': analysis['vwap'],
                    'confidence': 0.6,
                    'reason': 'VWAP reversion long'
                }
            
            # Short if price significantly above VWAP with negative delta
            elif vwap_distance > 0.002 and analysis['cumulative_delta'] < 0:
                return {
                    'type': 'short',
                    'entry': current_price,
                    'stop_loss': current_price * 1.002,
                    'target': analysis['vwap'],
                    'confidence': 0.6,
                    'reason': 'VWAP reversion short'
                }
        
        return None

class BybitWebSocketManager:
    """Manage WebSocket connections for real-time data"""
    
    def __init__(self, strategy: InstitutionalStrategy, testnet: bool = True):
        self.strategy = strategy
        self.testnet = testnet
        self.ws = None
        self.running = False
        
        if testnet:
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmation
            if 'success' in data:
                if data['success']:
                    logger.info(f"Successfully subscribed to: {data.get('ret_msg', 'channels')}")
                return
            
            # Handle data messages
            if 'topic' in data and 'data' in data:
                topic = data['topic']
                
                # Order book updates
                if 'orderbook' in topic:
                    orderbook_data = data['data']
                    
                    # Bybit sends snapshot and delta updates
                    if 'b' in orderbook_data:  # bids
                        self.strategy.order_flow.update_orderbook({
                            'bids': orderbook_data['b'],
                            'asks': orderbook_data.get('a', [])
                        })
                    
                # Trade updates
                elif 'publicTrade' in topic:
                    trades_data = data['data']
                    
                    for trade in trades_data:
                        # Bybit trade format
                        trade_info = {
                            'price': trade.get('p', 0),  # price
                            'size': trade.get('v', 0),   # volume
                            'side': 'buy' if trade.get('S') == 'Buy' else 'sell',  # side
                            'timestamp': trade.get('T', int(time.time() * 1000))  # timestamp
                        }
                        
                        if trade_info['price'] and trade_info['size']:
                            self.strategy.order_flow.analyze_trade(trade_info)
                            self.strategy.microstructure.update_tick(
                                float(trade_info['price']),
                                float(trade_info['size']),
                                trade_info['timestamp']
                            )
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            logger.debug(f"Message content: {message[:200]}...")  # Log first 200 chars for debugging
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        self.running = False
    
    def on_open(self, ws):
        logger.info("WebSocket connection opened")
        
        # Subscribe to order book and trades
        subscribe_message = {
            "op": "subscribe",
            "args": [
                "orderbook.50.BTCUSDT",
                "publicTrade.BTCUSDT"
            ]
        }
        ws.send(json.dumps(subscribe_message))
    
    def start(self):
        """Start WebSocket connection"""
        self.running = True
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

class AdvancedBybitBot:
    """Advanced trading bot using institutional order flow"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Initialize strategy and WebSocket
        self.strategy = InstitutionalStrategy()
        self.ws_manager = BybitWebSocketManager(self.strategy, testnet)
        
        self.active_position = None
        self.performance = {'wins': 0, 'losses': 0, 'pnl': 0}
    
    def calculate_position_size(self, entry: float, stop_loss: float, confidence: float = 1.0) -> float:
        """Calculate position size with confidence adjustment"""
        try:
            balance = self.exchange.fetch_balance()
            account_balance = balance['USDT']['free'] if 'USDT' in balance else 0
            
            # Adjust risk based on confidence
            adjusted_risk = self.strategy.risk_per_trade * confidence
            risk_amount = account_balance * adjusted_risk
            
            # Calculate position size
            risk_per_contract = abs(entry - stop_loss)
            position_size = risk_amount / risk_per_contract
            
            # Apply maximum position size limit
            max_position = account_balance * 0.1  # Max 10% of account
            position_size = min(position_size, max_position / entry)
            
            return round(position_size, 3)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def execute_trade(self, signal: Dict, symbol: str = 'BTC/USDT:USDT') -> bool:
        """Execute trade based on signal"""
        try:
            # Check if we already have a position
            if self.active_position:
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(
                signal['entry'], 
                signal['stop_loss'],
                signal.get('confidence', 1.0)
            )
            
            if position_size == 0:
                return False
            
            # Place market order for immediate execution
            side = 'buy' if signal['type'] == 'long' else 'sell'
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_size
            )
            
            if order:
                self.active_position = {
                    'entry': signal['entry'],
                    'stop_loss': signal['stop_loss'],
                    'size': position_size,
                    'side': signal['type'],
                    'reason': signal['reason'],
                    'entry_time': time.time()
                }
                
                # Set stop loss
                sl_side = 'sell' if signal['type'] == 'long' else 'buy'
                sl_params = {
                    'stopPrice': signal['stop_loss'],
                    'triggerBy': 'LastPrice',
                    'reduceOnly': True
                }
                
                self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=sl_side,
                    amount=position_size,
                    price=signal['stop_loss'],
                    params=sl_params
                )
                
                logger.info(f"Position opened: {self.active_position}")
                return True
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def manage_position(self, current_price: float, symbol: str = 'BTC/USDT:USDT') -> None:
        """Manage active position"""
        if not self.active_position:
            return
        
        try:
            position = self.active_position
            
            # Calculate P&L
            if position['side'] == 'long':
                pnl_pct = (current_price - position['entry']) / position['entry']
            else:
                pnl_pct = (position['entry'] - current_price) / position['entry']
            
            # Dynamic exit conditions
            time_in_position = time.time() - position['entry_time']
            
            # Quick scalp exit - take profit at 0.2% for flow-based trades
            if pnl_pct > 0.002 and time_in_position < 300:  # 5 minutes
                self.close_position(symbol, reason="Quick scalp target")
            
            # Time-based exit - close if no movement after 15 minutes
            elif abs(pnl_pct) < 0.0005 and time_in_position > 900:
                self.close_position(symbol, reason="Time-based exit")
            
            # Trailing stop for profitable positions
            elif pnl_pct > 0.003:  # 0.3% profit
                new_stop = current_price * (0.998 if position['side'] == 'long' else 1.002)
                if position['side'] == 'long' and new_stop > position['stop_loss']:
                    self.update_stop_loss(symbol, new_stop)
                elif position['side'] == 'short' and new_stop < position['stop_loss']:
                    self.update_stop_loss(symbol, new_stop)
                    
        except Exception as e:
            logger.error(f"Error managing position: {e}")
    
    def update_stop_loss(self, symbol: str, new_stop: float) -> None:
        """Update stop loss order"""
        try:
            # Cancel existing stop orders
            orders = self.exchange.fetch_open_orders(symbol)
            for order in orders:
                if order['type'] == 'stop':
                    self.exchange.cancel_order(order['id'], symbol)
            
            # Place new stop
            sl_side = 'sell' if self.active_position['side'] == 'long' else 'buy'
            sl_params = {
                'stopPrice': new_stop,
                'triggerBy': 'LastPrice',
                'reduceOnly': True
            }
            
            self.exchange.create_order(
                symbol=symbol,
                type='stop',
                side=sl_side,
                amount=self.active_position['size'],
                price=new_stop,
                params=sl_params
            )
            
            self.active_position['stop_loss'] = new_stop
            logger.info(f"Stop loss updated to: {new_stop}")
            
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
    
    def close_position(self, symbol: str = 'BTC/USDT:USDT', reason: str = "") -> None:
        """Close active position"""
        if not self.active_position:
            return
        
        try:
            # Cancel all orders
            self.exchange.cancel_all_orders(symbol)
            
            # Close position
            side = 'sell' if self.active_position['side'] == 'long' else 'buy'
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=self.active_position['size'],
                params={'reduceOnly': True}
            )
            
            # Get fill price
            fill_price = float(order['average']) if order and 'average' in order else 0
            
            # Calculate final P&L
            if self.active_position['side'] == 'long':
                pnl = (fill_price - self.active_position['entry']) * self.active_position['size']
            else:
                pnl = (self.active_position['entry'] - fill_price) * self.active_position['size']
            
            # Update performance
            if pnl > 0:
                self.performance['wins'] += 1
            else:
                self.performance['losses'] += 1
            self.performance['pnl'] += pnl
            
            logger.info(f"Position closed: {reason}, P&L: ${pnl:.2f}")
            logger.info(f"Performance: {self.performance}")
            
            self.active_position = None
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def run(self, symbol: str = 'BTC/USDT:USDT'):
        """Main trading loop"""
        logger.info("Starting advanced trading bot...")
        logger.info(f"Trading {symbol}")
        
        # Start WebSocket data feed
        self.ws_manager.start()
        
        # Wait for initial data
        logger.info("Collecting initial market data...")
        time.sleep(10)
        
        last_signal_time = 0
        signal_cooldown = 60  # Minimum seconds between signals
        
        while True:
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Analyze market
                analysis = self.strategy.analyze_market_state(current_price)
                
                # Manage existing position
                self.manage_position(current_price, symbol)
                
                # Generate new signal only if no position and cooldown passed
                if not self.active_position and time.time() - last_signal_time > signal_cooldown:
                    signal = self.strategy.generate_signal(analysis)
                    
                    if signal:
                        logger.info(f"Signal generated: {signal}")
                        if self.execute_trade(signal, symbol):
                            last_signal_time = time.time()
                
                # Log market state every 30 seconds
                if int(time.time()) % 30 == 0:
                    logger.info(f"Market state - Price: {current_price:.2f}, "
                              f"Delta: {analysis['cumulative_delta']:.0f}, "
                              f"Imbalance: {analysis['flow_imbalance']:.3f}, "
                              f"Aggressive Buy%: {analysis['aggressive_buy_ratio']:.2%}")
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Shutting down bot...")
                self.close_position(symbol, reason="Manual shutdown")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

# Configuration
def main():
    config = {
        'api_key': 'VDpt0WQXIjXul4OBrS',
        'api_secret': 'z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9',
        'testnet': False,
        'symbol': 'BTC/USDT:USDT'
    }
    
    bot = AdvancedBybitBot(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        testnet=config['testnet']
    )
    
    bot.run(symbol=config['symbol'])

if __name__ == "__main__":
    main()