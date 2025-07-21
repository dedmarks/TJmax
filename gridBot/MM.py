import ccxt
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Optional
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketMakingBot:
    """Market making bot for small cap altcoins"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot for altcoins
                'adjustForTimeDifference': True
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        # Market making parameters
        self.spread_percentage = 0.005  # 0.5% minimum spread
        self.order_levels = 5  # Number of orders on each side
        self.order_amount = 50  # USD value per order
        self.max_position = 500  # Maximum position in USD
        self.rebalance_threshold = 0.7  # Rebalance if 70% on one side
        
        # Risk management
        self.max_spread = 0.03  # 3% maximum spread
        self.volatility_threshold = 0.05  # 5% volatility threshold
        self.min_liquidity = 10000  # Minimum 24h volume in USD
        
        # Tracking
        self.active_orders = {'buy': [], 'sell': []}
        self.position = {'base': 0, 'quote': 0}
        self.trades_history = deque(maxlen=1000)
        self.pnl = {'realized': 0, 'fees': 0}
        self.price_history = deque(maxlen=100)
        
    def select_suitable_altcoin(self, min_volume: float = 10000, max_spread: float = 0.05) -> Optional[str]:
        """Find suitable altcoin for market making"""
        try:
            # Fetch all markets
            markets = self.exchange.fetch_markets()
            tickers = self.exchange.fetch_tickers()
            
            suitable_coins = []
            
            for market in markets:
                symbol = market['symbol']
                
                # Skip if not spot or not active
                if market['type'] != 'spot' or not market['active']:
                    continue
                
                # Skip major pairs
                if any(base in symbol for base in ['BTC/', 'ETH/', 'BNB/']):
                    continue
                
                # Must be USDT pair
                if not symbol.endswith('/USDT'):
                    continue
                
                # Check ticker data
                if symbol in tickers:
                    ticker = tickers[symbol]
                    
                    # Check volume
                    volume_usd = ticker.get('quoteVolume', 0)
                    if volume_usd < min_volume or volume_usd > min_volume * 100:
                        continue
                    
                    # Check spread
                    if ticker['bid'] and ticker['ask']:
                        spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                        if spread > max_spread:
                            continue
                        
                        suitable_coins.append({
                            'symbol': symbol,
                            'volume': volume_usd,
                            'spread': spread,
                            'price': ticker['last'],
                            'volatility': ticker.get('percentage', 0)
                        })
            
            # Sort by spread (wider = more profit potential)
            suitable_coins.sort(key=lambda x: x['spread'], reverse=True)
            
            if suitable_coins:
                logger.info(f"Found {len(suitable_coins)} suitable coins")
                for coin in suitable_coins[:5]:
                    logger.info(f"{coin['symbol']}: Volume=${coin['volume']:.0f}, "
                              f"Spread={coin['spread']:.2%}, Price=${coin['price']}")
                
                return suitable_coins[0]['symbol']
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting altcoin: {e}")
            return None
    
    def calculate_spread_parameters(self, symbol: str) -> Dict:
        """Calculate optimal spread based on market conditions"""
        try:
            # Fetch order book
            order_book = self.exchange.fetch_order_book(symbol, 20)
            
            # Calculate current spread
            best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
            best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
            
            if not best_bid or not best_ask:
                return None
            
            current_spread = (best_ask - best_bid) / best_bid
            
            # Calculate order book depth
            bid_depth = sum(bid[1] * bid[0] for bid in order_book['bids'][:10])
            ask_depth = sum(ask[1] * ask[0] for ask in order_book['asks'][:10])
            
            # Calculate recent volatility
            if len(self.price_history) > 20:
                prices = [p['price'] for p in list(self.price_history)[-20:]]
                volatility = np.std(prices) / np.mean(prices)
            else:
                volatility = 0.01
            
            # Adjust spread based on conditions
            # Wider spread for higher volatility
            volatility_adjustment = 1 + (volatility * 10)
            
            # Wider spread for thin order books
            depth_ratio = min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
            depth_adjustment = 1 + (1 - depth_ratio) * 0.5
            
            # Calculate our spread
            our_spread = max(
                self.spread_percentage * volatility_adjustment * depth_adjustment,
                current_spread * 0.8,  # At least 80% of current spread
                0.003  # Minimum 0.3%
            )
            
            # Don't exceed max spread
            our_spread = min(our_spread, self.max_spread)
            
            return {
                'spread': our_spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'volatility': volatility,
                'mid_price': (best_bid + best_ask) / 2
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return None
    
    def calculate_order_sizes(self, symbol: str, spread_params: Dict) -> List[Tuple[float, float]]:
        """Calculate order sizes for each level"""
        mid_price = spread_params['mid_price']
        spread = spread_params['spread']
        
        # Get balance
        balance = self.exchange.fetch_balance()
        
        # Parse symbol
        base_currency = symbol.split('/')[0]
        quote_currency = symbol.split('/')[1]
        
        base_balance = balance.get(base_currency, {}).get('free', 0)
        quote_balance = balance.get(quote_currency, {}).get('free', 0)
        
        # Calculate position value
        position_value = base_balance * mid_price + quote_balance
        
        # Order sizes (smaller at extremes)
        order_sizes = []
        size_multipliers = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing sizes
        
        for i in range(self.order_levels):
            size = (self.order_amount / mid_price) * size_multipliers[i]
            order_sizes.append(size)
        
        return order_sizes
    
    def place_orders(self, symbol: str) -> bool:
        """Place market making orders"""
        try:
            # Cancel existing orders first
            self.cancel_all_orders(symbol)
            
            # Calculate spread parameters
            spread_params = self.calculate_spread_parameters(symbol)
            if not spread_params:
                logger.warning("Could not calculate spread parameters")
                return False
            
            mid_price = spread_params['mid_price']
            spread = spread_params['spread']
            volatility = spread_params['volatility']
            
            # Skip if too volatile
            if volatility > self.volatility_threshold:
                logger.warning(f"Volatility too high: {volatility:.2%}")
                return False
            
            # Calculate order sizes
            order_sizes = self.calculate_order_sizes(symbol, spread_params)
            
            # Place buy orders
            buy_orders = []
            for i in range(self.order_levels):
                price_multiplier = 1 - (spread/2) - (i * spread * 0.3)
                price = mid_price * price_multiplier
                size = order_sizes[i]
                
                try:
                    order = self.exchange.create_limit_buy_order(symbol, size, price)
                    buy_orders.append(order)
                    logger.info(f"Buy order placed: {size:.4f} @ {price:.4f}")
                except Exception as e:
                    logger.error(f"Error placing buy order: {e}")
            
            # Place sell orders
            sell_orders = []
            for i in range(self.order_levels):
                price_multiplier = 1 + (spread/2) + (i * spread * 0.3)
                price = mid_price * price_multiplier
                size = order_sizes[i]
                
                try:
                    order = self.exchange.create_limit_sell_order(symbol, size, price)
                    sell_orders.append(order)
                    logger.info(f"Sell order placed: {size:.4f} @ {price:.4f}")
                except Exception as e:
                    logger.error(f"Error placing sell order: {e}")
            
            self.active_orders = {'buy': buy_orders, 'sell': sell_orders}
            
            logger.info(f"Placed {len(buy_orders)} buy and {len(sell_orders)} sell orders")
            logger.info(f"Spread: {spread:.2%}, Mid: {mid_price:.4f}, Volatility: {volatility:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing orders: {e}")
            return False
    
    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all active orders"""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
            
            self.active_orders = {'buy': [], 'sell': []}
            logger.info(f"Cancelled {len(open_orders)} orders")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def check_filled_orders(self, symbol: str) -> None:
        """Check for filled orders and track P&L"""
        try:
            # Check buy orders
            for order in self.active_orders['buy'][:]:
                updated_order = self.exchange.fetch_order(order['id'], symbol)
                
                if updated_order['status'] == 'closed':
                    # Order filled
                    fill_price = updated_order['average']
                    fill_amount = updated_order['filled']
                    fee = updated_order['fee']['cost'] if updated_order['fee'] else 0
                    
                    self.trades_history.append({
                        'side': 'buy',
                        'price': fill_price,
                        'amount': fill_amount,
                        'fee': fee,
                        'timestamp': time.time()
                    })
                    
                    self.position['base'] += fill_amount
                    self.position['quote'] -= fill_amount * fill_price
                    self.pnl['fees'] += fee
                    
                    self.active_orders['buy'].remove(order)
                    logger.info(f"Buy filled: {fill_amount:.4f} @ {fill_price:.4f}")
            
            # Check sell orders
            for order in self.active_orders['sell'][:]:
                updated_order = self.exchange.fetch_order(order['id'], symbol)
                
                if updated_order['status'] == 'closed':
                    # Order filled
                    fill_price = updated_order['average']
                    fill_amount = updated_order['filled']
                    fee = updated_order['fee']['cost'] if updated_order['fee'] else 0
                    
                    self.trades_history.append({
                        'side': 'sell',
                        'price': fill_price,
                        'amount': fill_amount,
                        'fee': fee,
                        'timestamp': time.time()
                    })
                    
                    self.position['base'] -= fill_amount
                    self.position['quote'] += fill_amount * fill_price
                    self.pnl['fees'] += fee
                    
                    # Calculate profit
                    if len(self.trades_history) > 1:
                        # Find matching buy
                        buy_trades = [t for t in self.trades_history if t['side'] == 'buy']
                        if buy_trades:
                            avg_buy_price = np.mean([t['price'] for t in buy_trades[-5:]])
                            profit = (fill_price - avg_buy_price) * fill_amount
                            self.pnl['realized'] += profit
                    
                    self.active_orders['sell'].remove(order)
                    logger.info(f"Sell filled: {fill_amount:.4f} @ {fill_price:.4f}")
                    
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
    
    def rebalance_inventory(self, symbol: str, mid_price: float) -> None:
        """Rebalance inventory if too skewed"""
        try:
            # Calculate current position ratio
            base_value = self.position['base'] * mid_price
            total_value = base_value + self.position['quote']
            
            if total_value <= 0:
                return
            
            base_ratio = base_value / total_value
            
            # Rebalance if too skewed
            if base_ratio > self.rebalance_threshold:
                # Too much base currency, market sell some
                sell_amount = self.position['base'] * 0.2
                order = self.exchange.create_market_sell_order(symbol, sell_amount)
                logger.info(f"Rebalancing: Market sell {sell_amount:.4f}")
                
            elif base_ratio < (1 - self.rebalance_threshold):
                # Too much quote currency, market buy some
                buy_value = self.position['quote'] * 0.2
                buy_amount = buy_value / mid_price
                order = self.exchange.create_market_buy_order(symbol, buy_amount)
                logger.info(f"Rebalancing: Market buy {buy_amount:.4f}")
                
        except Exception as e:
            logger.error(f"Error rebalancing: {e}")
    
    def update_price_history(self, symbol: str) -> None:
        """Update price history for volatility calculation"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.price_history.append({
                'price': ticker['last'],
                'volume': ticker['quoteVolume'],
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error updating price history: {e}")
    
    def log_performance(self, symbol: str, mid_price: float) -> None:
        """Log bot performance"""
        try:
            # Calculate unrealized P&L
            base_value = self.position['base'] * mid_price
            total_value = base_value + self.position['quote']
            
            # Count trades
            recent_trades = [t for t in self.trades_history if time.time() - t['timestamp'] < 3600]
            trades_per_hour = len(recent_trades)
            
            # Calculate spread capture
            buy_fills = [t for t in recent_trades if t['side'] == 'buy']
            sell_fills = [t for t in recent_trades if t['side'] == 'sell']
            
            if buy_fills and sell_fills:
                avg_buy = np.mean([t['price'] for t in buy_fills])
                avg_sell = np.mean([t['price'] for t in sell_fills])
                spread_capture = (avg_sell - avg_buy) / avg_buy
            else:
                spread_capture = 0
            
            logger.info(f"=== Performance Update ===")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Position: {self.position['base']:.4f} base, {self.position['quote']:.2f} quote")
            logger.info(f"Total Value: ${total_value:.2f}")
            logger.info(f"Realized P&L: ${self.pnl['realized']:.2f}")
            logger.info(f"Fees Paid: ${self.pnl['fees']:.2f}")
            logger.info(f"Net P&L: ${self.pnl['realized'] - self.pnl['fees']:.2f}")
            logger.info(f"Trades/Hour: {trades_per_hour}")
            logger.info(f"Avg Spread Capture: {spread_capture:.2%}")
            logger.info(f"=======================")
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def run(self, symbol: str = None, update_interval: int = 10):
        """Main bot loop"""
        # Auto-select coin if not specified
        if not symbol:
            logger.info("Auto-selecting suitable altcoin...")
            symbol = self.select_suitable_altcoin()
            if not symbol:
                logger.error("No suitable altcoin found")
                return
        
        logger.info(f"Starting market maker for {symbol}")
        
        # Main loop
        order_update_counter = 0
        
        while True:
            try:
                # Update price history
                self.update_price_history(symbol)
                
                # Check filled orders
                self.check_filled_orders(symbol)
                
                # Update orders every 6 iterations (1 minute if 10s interval)
                order_update_counter += 1
                if order_update_counter >= 6:
                    self.place_orders(symbol)
                    order_update_counter = 0
                
                # Get current mid price
                ticker = self.exchange.fetch_ticker(symbol)
                mid_price = ticker['last']
                
                # Rebalance if needed
                self.rebalance_inventory(symbol, mid_price)
                
                # Log performance every 5 minutes
                if int(time.time()) % 300 < update_interval:
                    self.log_performance(symbol, mid_price)
                
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.cancel_all_orders(symbol)
                self.log_performance(symbol, mid_price)
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(update_interval)

class AdvancedMarketMaker(MarketMakingBot):
    """Advanced market maker with additional features"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        # Advanced parameters
        self.use_dynamic_sizing = True
        self.use_order_clustering = True
        self.whale_detection = True
        self.momentum_adjustment = True
        
        # Tracking
        self.momentum = deque(maxlen=20)
        self.whale_orders = []
        
    def detect_whale_orders(self, order_book: Dict, threshold: float = 0.1) -> List[Dict]:
        """Detect large orders that might impact price"""
        whale_orders = []
        
        total_bid_volume = sum(bid[1] for bid in order_book['bids'][:20])
        total_ask_volume = sum(ask[1] for ask in order_book['asks'][:20])
        
        # Check for large bids
        for bid in order_book['bids'][:10]:
            if bid[1] > total_bid_volume * threshold:
                whale_orders.append({
                    'side': 'bid',
                    'price': bid[0],
                    'size': bid[1],
                    'impact': 'support'
                })
        
        # Check for large asks
        for ask in order_book['asks'][:10]:
            if ask[1] > total_ask_volume * threshold:
                whale_orders.append({
                    'side': 'ask',
                    'price': ask[0],
                    'size': ask[1],
                    'impact': 'resistance'
                })
        
        return whale_orders
    
    def calculate_momentum(self) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < 10:
            return 0
        
        prices = [p['price'] for p in list(self.price_history)[-10:]]
        returns = np.diff(prices) / prices[:-1]
        momentum = np.mean(returns) * 100
        
        self.momentum.append(momentum)
        return momentum
    
    def adjust_for_momentum(self, spread: float, momentum: float) -> float:
        """Adjust spread based on momentum"""
        if abs(momentum) < 0.1:  # Low momentum
            return spread
        
        # Increase spread in direction of momentum
        if momentum > 0:  # Upward momentum
            # Wider spread on sell side
            return spread * (1 + abs(momentum))
        else:  # Downward momentum
            # Wider spread on buy side
            return spread * (1 + abs(momentum))

# Main execution
def main():
    # Configuration
    config = {
        'api_key': 'VDpt0WQXIjXul4OBrS',
        'api_secret': 'z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9',
        'testnet': False,  # Use real trading for altcoins
        'symbol': 'XRP/USDT',  # Auto-select if None
        'update_interval': 10  # seconds
    }
    
    # Create bot instance
    bot = AdvancedMarketMaker(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        testnet=config['testnet']
    )
    
    # Configure parameters
    bot.spread_percentage = 0.005  # 0.5% minimum spread
    bot.order_levels = 5  # 5 orders each side
    bot.order_amount = 50  # $50 per order
    bot.max_position = 500  # $500 max position
    
    # Run bot
    bot.run(
        symbol=config['symbol'],
        update_interval=config['update_interval']
    )

if __name__ == "__main__":
    main()