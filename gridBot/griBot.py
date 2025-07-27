import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
from collections import deque
import statistics

# Enhanced imports for advanced strategies
from scipy.optimize import minimize
from scipy.stats import jarque_bera, normaltest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_grid_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedFuturesGridTrader:
    def __init__(self, symbol: str, api_key: str, secret: str, capital: float, 
                 leverage: int = 40, base_grid_levels: int = 8, 
                 use_testnet: bool = False, max_position_ratio: float = 0.9):
        """
        Enhanced Futures Grid Trading Bot focused on profitability
        
        Key improvements:
        - Dynamic grid sizing based on market microstructure
        - Orderbook imbalance analysis
        - Funding rate arbitrage optimization
        - Volatility regime detection
        - Adaptive position sizing
        - Market making vs taking decisions
        """
        self.bybit = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': use_testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap'
            }
        })
        
        self.symbol = symbol
        self.capital = capital
        self.leverage = leverage
        self.base_grid_levels = base_grid_levels
        self.max_position_ratio = max_position_ratio
        
        # Enhanced trading state
        self.active_orders: Dict[str, Dict] = {}
        self.filled_orders: List[Dict] = []
        self.current_position: float = 0.0
        self.total_profit: float = 0.0
        self.is_running: bool = False
        self.trades_count: int = 0
        self.start_time: Optional[datetime] = None
        
        # Market microstructure data
        self.orderbook_history = deque(maxlen=100)
        self.trade_flow_history = deque(maxlen=200)
        self.funding_rate_history = deque(maxlen=48)  # 48 hours of funding rates
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=200)
        
        # Dynamic parameters
        self.current_spread_multiplier = 1.0
        self.volatility_regime = 'normal'  # 'low', 'normal', 'high', 'extreme'
        self.market_regime = 'neutral'  # 'bullish', 'bearish', 'neutral', 'choppy'
        self.funding_bias = 0.0
        
        # Performance tracking
        self.funding_fees_collected = 0.0
        self.maker_rebates = 0.0
        self.slippage_costs = 0.0
        self.opportunity_cost = 0.0
        
        # Risk management
        self.max_drawdown = 0.0
        self.peak_equity = capital
        self.daily_pnl_history = deque(maxlen=30)
        
        logger.info(f"Enhanced Grid Trader initialized for {symbol}")
        logger.info(f"Capital: ${capital}, Leverage: {leverage}x, Base Grid Levels: {base_grid_levels}")
    
    async def get_enhanced_market_data(self) -> Dict:
        """Get comprehensive market data including orderbook depth"""
        try:
            # Get basic market data
            ticker = self.bybit.fetch_ticker(self.symbol)
            funding_rate = self.bybit.fetch_funding_rate(self.symbol)
            
            # Get orderbook for microstructure analysis
            orderbook = self.bybit.fetch_order_book(self.symbol, limit=20)
            
            # Get recent trades for flow analysis
            recent_trades = self.bybit.fetch_trades(self.symbol, limit=100)
            
            market_data = {
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': ticker['ask'] - ticker['bid'],
                'spread_bps': ((ticker['ask'] - ticker['bid']) / ticker['last']) * 10000,
                'volume': ticker['baseVolume'],
                'funding_rate': funding_rate['fundingRate'],
                'funding_time': funding_rate['fundingDatetime'],
                'orderbook': orderbook,
                'recent_trades': recent_trades,
                'timestamp': datetime.now()
            }
            
            # Update histories
            self.price_history.append(market_data['price'])
            self.volume_history.append(market_data['volume'])
            self.funding_rate_history.append(market_data['funding_rate'])
            self.orderbook_history.append(orderbook)
            self.trade_flow_history.extend(recent_trades)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching enhanced market data: {e}")
            raise
    
    def analyze_orderbook_imbalance(self, orderbook: Dict) -> Dict:
        """Analyze orderbook for imbalance and liquidity"""
        try:
            bids = orderbook['bids'][:10]  # Top 10 levels
            asks = orderbook['asks'][:10]
            
            bid_volume = sum(level[1] for level in bids)
            ask_volume = sum(level[1] for level in asks)
            total_volume = bid_volume + ask_volume
            
            # Calculate imbalance ratio (-1 to 1, positive = more bids)
            imbalance_ratio = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate weighted mid price
            if bids and asks:
                weighted_mid = (bids[0][0] * ask_volume + asks[0][0] * bid_volume) / total_volume
            else:
                weighted_mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
            
            # Calculate liquidity depth
            price_range = asks[-1][0] - bids[-1][0] if len(bids) >= 10 and len(asks) >= 10 else 0
            liquidity_density = total_volume / price_range if price_range > 0 else 0
            
            return {
                'imbalance_ratio': imbalance_ratio,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'weighted_mid': weighted_mid,
                'liquidity_density': liquidity_density,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing orderbook: {e}")
            return {}
    
    def analyze_trade_flow(self) -> Dict:
        """Analyze recent trade flow for directional bias"""
        try:
            if len(self.trade_flow_history) < 50:
                return {'flow_bias': 0.0, 'volume_weighted_price': 0.0, 'aggressor_ratio': 0.5}
            
            recent_trades = list(self.trade_flow_history)[-100:]
            
            buy_volume = sum(trade['amount'] for trade in recent_trades if trade['side'] == 'buy')
            sell_volume = sum(trade['amount'] for trade in recent_trades if trade['side'] == 'sell')
            total_volume = buy_volume + sell_volume
            
            # Flow bias: positive = more buying pressure
            flow_bias = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # Volume weighted average price
            vwap = sum(trade['price'] * trade['amount'] for trade in recent_trades) / total_volume if total_volume > 0 else 0
            
            # Aggressor ratio (market orders vs limit orders)
            market_orders = sum(1 for trade in recent_trades if trade.get('takerOrMaker') == 'taker')
            aggressor_ratio = market_orders / len(recent_trades) if recent_trades else 0.5
            
            return {
                'flow_bias': flow_bias,
                'volume_weighted_price': vwap,
                'aggressor_ratio': aggressor_ratio,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trade flow: {e}")
            return {'flow_bias': 0.0, 'volume_weighted_price': 0.0, 'aggressor_ratio': 0.5}
    
    def detect_volatility_regime(self) -> str:
        """Detect current volatility regime using multiple timeframes"""
        try:
            if len(self.price_history) < 50:
                return 'normal'
            
            prices = np.array(list(self.price_history))
            returns = np.diff(np.log(prices)) * 100
            
            # Calculate volatility metrics
            current_vol = np.std(returns[-24:]) * np.sqrt(24) if len(returns) >= 24 else 0
            long_term_vol = np.std(returns[-168:]) * np.sqrt(24) if len(returns) >= 168 else current_vol
            
            # Volatility regime classification
            vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1
            
            if vol_ratio < 0.7:
                regime = 'low'
            elif vol_ratio > 2.0:
                regime = 'extreme'
            elif vol_ratio > 1.3:
                regime = 'high'
            else:
                regime = 'normal'
            
            # Additional check for regime stability
            recent_vols = [np.std(returns[i-12:i]) for i in range(max(12, len(returns)-48), len(returns), 12)]
            vol_stability = np.std(recent_vols) / np.mean(recent_vols) if recent_vols and np.mean(recent_vols) > 0 else 0
            
            if vol_stability > 0.5:  # High volatility instability
                regime = 'extreme'
            
            logger.info(f"Volatility regime: {regime} (ratio: {vol_ratio:.2f}, stability: {vol_stability:.2f})")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return 'normal'
    
    def detect_market_regime(self) -> str:
        """Detect market regime using price action and flow"""
        try:
            if len(self.price_history) < 50:
                return 'neutral'
            
            prices = np.array(list(self.price_history))
            
            # Trend detection using multiple timeframes
            short_trend = (prices[-1] - prices[-12]) / prices[-12] if len(prices) >= 12 else 0
            medium_trend = (prices[-1] - prices[-48]) / prices[-48] if len(prices) >= 48 else 0
            
            # Price momentum
            momentum_12 = np.mean(np.diff(prices[-12:])) if len(prices) >= 12 else 0
            momentum_48 = np.mean(np.diff(prices[-48:])) if len(prices) >= 48 else 0
            
            # Range detection
            recent_high = np.max(prices[-48:]) if len(prices) >= 48 else prices[-1]
            recent_low = np.min(prices[-48:]) if len(prices) >= 48 else prices[-1]
            range_position = (prices[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # Trade flow bias
            flow_data = self.analyze_trade_flow()
            flow_bias = flow_data.get('flow_bias', 0)
            
            # Regime classification
            if short_trend > 0.01 and medium_trend > 0.005 and flow_bias > 0.1:
                regime = 'bullish'
            elif short_trend < -0.01 and medium_trend < -0.005 and flow_bias < -0.1:
                regime = 'bearish'
            elif abs(short_trend) < 0.005 and abs(medium_trend) < 0.003:
                regime = 'choppy'
            else:
                regime = 'neutral'
            
            logger.info(f"Market regime: {regime} (short: {short_trend:.4f}, flow: {flow_bias:.3f})")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'neutral'
    
    def calculate_optimal_grid_spacing(self, market_data: Dict) -> float:
        """Calculate optimal grid spacing based on market conditions"""
        try:
            base_spread = market_data['spread_bps'] / 10000  # Convert to decimal
            
            # Volatility adjustment
            volatility_multiplier = {
                'low': 0.6,
                'normal': 1.0,
                'high': 1.4,
                'extreme': 2.0
            }.get(self.volatility_regime, 1.0)
            
            # Market regime adjustment
            regime_multiplier = {
                'choppy': 0.7,    # Tighter grids in choppy markets
                'neutral': 1.0,
                'bullish': 1.2,   # Wider grids in trending markets
                'bearish': 1.2
            }.get(self.market_regime, 1.0)
            
            # Funding rate adjustment
            funding_rate = abs(market_data.get('funding_rate', 0))
            funding_multiplier = 1.0 + min(funding_rate * 100, 0.3)  # Cap at 30% adjustment
            
            # Orderbook liquidity adjustment
            if self.orderbook_history:
                ob_analysis = self.analyze_orderbook_imbalance(self.orderbook_history[-1])
                liquidity_multiplier = max(0.8, min(1.5, 1.0 / (ob_analysis.get('liquidity_density', 1) + 1)))
            else:
                liquidity_multiplier = 1.0
            
            # Calculate optimal spacing
            optimal_spacing = max(
                base_spread * 2,  # At least 2x the spread
                0.002 * volatility_multiplier * regime_multiplier * funding_multiplier * liquidity_multiplier
            )
            
            # Ensure minimum spacing for safety
            optimal_spacing = max(optimal_spacing, 0.0015)  # 0.15% minimum
            
            logger.info(f"Optimal grid spacing: {optimal_spacing:.4f} ({optimal_spacing*100:.2f}%)")
            return optimal_spacing
            
        except Exception as e:
            logger.error(f"Error calculating optimal grid spacing: {e}")
            return 0.003  # Fallback
    
    def calculate_dynamic_grid_levels(self, volatility_regime: str, market_regime: str) -> int:
        """Calculate dynamic number of grid levels based on market conditions"""
        base_levels = self.base_grid_levels
        
        # Volatility adjustment
        vol_adjustment = {
            'low': -2,      # Fewer levels in low vol
            'normal': 0,
            'high': 2,      # More levels in high vol
            'extreme': 4
        }.get(volatility_regime, 0)
        
        # Market regime adjustment
        regime_adjustment = {
            'choppy': 3,    # More levels in choppy markets
            'neutral': 0,
            'bullish': -1,  # Fewer levels in trending markets
            'bearish': -1
        }.get(market_regime, 0)
        
        dynamic_levels = max(4, min(15, base_levels + vol_adjustment + regime_adjustment))
        
        logger.info(f"Dynamic grid levels: {dynamic_levels} (base: {base_levels})")
        return dynamic_levels
    
    def calculate_position_size_kelly(self, price: float, side: str, win_rate: float = 0.6) -> float:
        """Calculate position size using Kelly Criterion approximation"""
        try:
            # Estimate win rate and average win/loss from recent trades
            if len(self.filled_orders) >= 10:
                recent_trades = self.filled_orders[-20:]
                wins = [t for t in recent_trades if t.get('profit', 0) > 0]
                losses = [t for t in recent_trades if t.get('profit', 0) < 0]
                
                if wins and losses:
                    win_rate = len(wins) / len(recent_trades)
                    avg_win = np.mean([t['profit'] for t in wins])
                    avg_loss = abs(np.mean([t['profit'] for t in losses]))
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
                else:
                    win_loss_ratio = 1.5  # Conservative default
            else:
                win_loss_ratio = 1.5  # Conservative default
            
            # Kelly fraction
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0.1, min(0.3, kelly_fraction))  # Cap between 10% and 30%
            
            # Base position size
            capital_per_level = self.capital / (self.base_grid_levels * 2)
            leveraged_capital = capital_per_level * self.leverage * kelly_fraction
            position_size = leveraged_capital / price
            
            # Volatility adjustment
            vol_adjustment = {
                'low': 1.2,
                'normal': 1.0,
                'high': 0.8,
                'extreme': 0.6
            }.get(self.volatility_regime, 1.0)
            
            position_size *= vol_adjustment
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return self.calculate_position_size_fallback(price)
    
    def calculate_position_size_fallback(self, price: float) -> float:
        """Fallback position size calculation"""
        capital_per_level = self.capital / (self.base_grid_levels * 2)
        leveraged_capital = capital_per_level * self.leverage * 0.8  # Conservative 80%
        return leveraged_capital / price
    
    def should_place_maker_order(self, side: str, market_data: Dict) -> bool:
        """Decide whether to place maker vs taker order based on market conditions"""
        try:
            # Always prefer maker orders for grid trading
            spread_bps = market_data.get('spread_bps', 50)
            
            # In very tight spreads, consider taking
            if spread_bps < 5:  # Less than 0.5 bps
                flow_data = self.analyze_trade_flow()
                # If flow is strongly against us, consider taking
                if (side == 'buy' and flow_data.get('flow_bias', 0) < -0.3) or \
                   (side == 'sell' and flow_data.get('flow_bias', 0) > 0.3):
                    return False
            
            return True  # Default to maker orders
            
        except Exception as e:
            logger.error(f"Error deciding order type: {e}")
            return True
    
    def optimize_funding_rate_exposure(self, market_data: Dict) -> Dict:
        """Optimize position to profit from funding rates"""
        try:
            funding_rate = market_data.get('funding_rate', 0)
            
            # Historical funding rate analysis
            if len(self.funding_rate_history) >= 8:  # 8 hours of history
                avg_funding = np.mean(list(self.funding_rate_history))
                funding_trend = np.polyfit(range(len(self.funding_rate_history)), 
                                         list(self.funding_rate_history), 1)[0]
            else:
                avg_funding = funding_rate
                funding_trend = 0
            
            # Funding rate thresholds (annualized)
            high_positive_funding = 0.01  # 1% per year
            high_negative_funding = -0.01
            
            strategy = {
                'bias': 'neutral',
                'position_adjustment': 1.0,
                'grid_skew': 0.0
            }
            
            # If funding is consistently high and positive (longs pay shorts)
            if avg_funding > high_positive_funding:
                strategy = {
                    'bias': 'short',
                    'position_adjustment': 1.2,  # Increase short exposure
                    'grid_skew': 0.1  # Favor sell orders
                }
                logger.info(f"High positive funding detected: {avg_funding:.6f}, favoring shorts")
            
            # If funding is consistently negative (shorts pay longs)
            elif avg_funding < high_negative_funding:
                strategy = {
                    'bias': 'long',
                    'position_adjustment': 1.2,  # Increase long exposure
                    'grid_skew': -0.1  # Favor buy orders
                }
                logger.info(f"High negative funding detected: {avg_funding:.6f}, favoring longs")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error optimizing funding exposure: {e}")
            return {'bias': 'neutral', 'position_adjustment': 1.0, 'grid_skew': 0.0}
    
    def setup_enhanced_grid(self) -> List[Dict]:
        """Setup enhanced grid with all optimizations"""
        try:
            # Get comprehensive market data
            market_data = asyncio.run(self.get_enhanced_market_data())
            
            # Update regimes
            self.volatility_regime = self.detect_volatility_regime()
            self.market_regime = self.detect_market_regime()
            
            # Calculate optimal parameters
            optimal_spacing = self.calculate_optimal_grid_spacing(market_data)
            dynamic_levels = self.calculate_dynamic_grid_levels(self.volatility_regime, self.market_regime)
            
            # Funding rate optimization
            funding_strategy = self.optimize_funding_rate_exposure(market_data)
            
            base_price = market_data['price']
            
            logger.info(f"Enhanced Grid Setup:")
            logger.info(f"Base Price: ${base_price:.4f}")
            logger.info(f"Optimal Spacing: {optimal_spacing:.4f} ({optimal_spacing*100:.2f}%)")
            logger.info(f"Dynamic Levels: {dynamic_levels}")
            logger.info(f"Volatility Regime: {self.volatility_regime}")
            logger.info(f"Market Regime: {self.market_regime}")
            logger.info(f"Funding Strategy: {funding_strategy['bias']}")
            
            # Create optimized orders
            orders = []
            
            for i in range(1, dynamic_levels + 1):
                # Calculate prices with optimal spacing
                buy_price = base_price * (1 - optimal_spacing * i)
                sell_price = base_price * (1 + optimal_spacing * i)
                
                # Calculate position sizes using Kelly criterion
                buy_size = self.calculate_position_size_kelly(buy_price, 'buy')
                sell_size = self.calculate_position_size_kelly(sell_price, 'sell')
                
                # Apply funding rate bias
                if funding_strategy['bias'] == 'long':
                    buy_size *= funding_strategy['position_adjustment']
                    sell_size *= (2 - funding_strategy['position_adjustment'])
                elif funding_strategy['bias'] == 'short':
                    sell_size *= funding_strategy['position_adjustment']
                    buy_size *= (2 - funding_strategy['position_adjustment'])
                
                # Ensure minimum size
                buy_size = max(buy_size, 0.001)
                sell_size = max(sell_size, 0.001)
                
                orders.extend([
                    {
                        'side': 'buy',
                        'price': buy_price,
                        'size': buy_size,
                        'level': i,
                        'type': 'enhanced_grid',
                        'priority': dynamic_levels - i + 1  # Higher priority for closer levels
                    },
                    {
                        'side': 'sell',
                        'price': sell_price,
                        'size': sell_size,
                        'level': i,
                        'type': 'enhanced_grid',
                        'priority': dynamic_levels - i + 1
                    }
                ])
            
            return orders
            
        except Exception as e:
            logger.error(f"Error setting up enhanced grid: {e}")
            raise
    
    def setup_leverage_and_margin(self) -> None:
        """Setup leverage and margin mode"""
        try:
            # Set leverage
            try:
                self.bybit.set_leverage(self.leverage, self.symbol)
                logger.info(f"Leverage set to {self.leverage}x")
            except Exception as e:
                if "leverage not modified" in str(e):
                    logger.warning(f"Leverage already set: {e}")
                else:
                    raise
            
            # Set margin mode to cross margin
            self.bybit.set_margin_mode('cross', self.symbol)
            logger.info("Margin mode set to cross")
            
            # Set position mode to hedge mode
            try:
                self.bybit.set_position_mode(True, self.symbol)
                logger.info("Position mode set to hedge")
            except Exception as e:
                logger.warning(f"Could not set hedge mode: {e}")
                
        except Exception as e:
            logger.error(f"Error setting up leverage/margin: {e}")
            raise
    
    def place_enhanced_grid_orders(self) -> None:
        """Place enhanced grid orders with optimal execution"""
        try:
            orders = self.setup_enhanced_grid()
            
            # Sort orders by priority (closer levels first)
            orders.sort(key=lambda x: x['priority'], reverse=True)
            
            placed_orders = 0
            failed_orders = 0
            
            for order in orders:
                try:
                    # Determine position index
                    position_idx = 1 if order['side'] == 'buy' else 2
                    
                    # Place maker order
                    result = self.bybit.create_limit_order(
                        symbol=self.symbol,
                        side=order['side'],
                        amount=order['size'],
                        price=order['price'],
                        params={
                            'timeInForce': 'GTC',
                            'postOnly': True,  # Ensure maker order
                            'positionIdx': position_idx
                        }
                    )
                    
                    self.active_orders[result['id']] = {
                        'order_id': result['id'],
                        'side': order['side'],
                        'price': order['price'],
                        'size': order['size'],
                        'level': order['level'],
                        'type': order['type'],
                        'priority': order['priority'],
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"Enhanced order placed: {order['side']} {order['size']:.4f} @ ${order['price']:.4f} (L{order['level']})")
                    placed_orders += 1
                    time.sleep(0.05)  # Reduced delay for faster execution
                    
                except Exception as e:
                    logger.error(f"Error placing order: {e}")
                    failed_orders += 1
                    
                    # If too many failures, stop placing orders
                    if failed_orders > 3:
                        logger.warning("Too many order failures, stopping placement")
                        break
            
            logger.info(f"Enhanced grid placement complete: {placed_orders} orders placed, {failed_orders} failed")
            
        except Exception as e:
            logger.error(f"Error placing enhanced grid orders: {e}")
            raise
    
    def monitor_and_replace_orders(self) -> None:
        """Enhanced order monitoring and replacement"""
        try:
            # Get current market data
            market_data = asyncio.run(self.get_enhanced_market_data())
            
            # Check for filled orders
            filled_order_ids = []
            open_orders = self.bybit.fetch_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            for order_id, order_info in self.active_orders.items():
                if order_id not in open_order_ids:
                    try:
                        # Get filled order details
                        closed_orders = self.bybit.fetch_closed_orders(self.symbol, limit=5)
                        filled_order = next((o for o in closed_orders if o['id'] == order_id), None)
                        
                        if filled_order and filled_order['status'] == 'closed':
                            filled_order_ids.append(order_id)
                            
                            # Update order info with actual fill data
                            order_info['fill_price'] = filled_order['average']
                            order_info['fill_amount'] = filled_order['filled']
                            order_info['fees'] = filled_order.get('fee', {}).get('cost', 0)
                            
                            self.filled_orders.append(order_info)
                            self.trades_count += 1
                            
                            # Calculate profit
                            profit = self.calculate_enhanced_profit(order_info, market_data)
                            self.total_profit += profit
                            
                            logger.info(f"Order filled: {order_info['side']} {order_info['size']:.4f} @ ${order_info.get('fill_price', order_info['price']):.4f} | Profit: ${profit:.4f}")
                            
                            # Place intelligent replacement
                            self.place_intelligent_replacement(order_info, market_data)
                            
                    except Exception as e:
                        logger.error(f"Error processing filled order {order_id}: {e}")
            
            # Remove filled orders
            for order_id in filled_order_ids:
                del self.active_orders[order_id]
                
        except Exception as e:
            logger.error(f"Error monitoring orders: {e}")
    
    def calculate_enhanced_profit(self, order_info: Dict, market_data: Dict) -> float:
        """Calculate profit with all costs included"""
        try:
            fill_price = order_info.get('fill_price', order_info['price'])
            size = order_info.get('fill_amount', order_info['size'])
            fees = order_info.get('fees', 0)
            
            # Base profit from grid spacing
            notional_value = size * fill_price
            expected_profit = notional_value * 0.003  # Expected grid profit
            
            # Subtract actual fees paid
            net_profit = expected_profit - abs(fees)
            
            # Add maker rebate if applicable (estimated)
            if order_info.get('fees', 0) < 0:  # Negative fees = rebate
                self.maker_rebates += abs(fees)
            
            return net_profit
            
        except Exception as e:
            logger.error(f"Error calculating enhanced profit: {e}")
            return 0.0
    
    def place_intelligent_replacement(self, filled_order: Dict, market_data: Dict) -> None:
        """Place intelligent replacement order based on market conditions"""
        try:
            # Get current market analysis
            ob_analysis = self.analyze_orderbook_imbalance(self.orderbook_history[-1]) if self.orderbook_history else {}
            flow_analysis = self.analyze_trade_flow()
            
            if filled_order['side'] == 'buy':
                # Buy order filled, place sell order
                new_side = 'sell'
                position_idx = 2
                
                # Intelligent pricing based on market conditions
                base_price = filled_order.get('fill_price', filled_order['price'])
                
                # Adjust based on orderbook imbalance
                imbalance = ob_analysis.get('imbalance_ratio', 0)
                if imbalance > 0.2:  # Strong bid support
                    price_adjustment = 1.002  # Slightly more aggressive
                else:
                    price_adjustment = 1.003  # Standard grid spacing
                
                new_price = base_price * price_adjustment
                
            else:
                # Sell order filled, place buy order
                new_side = 'buy'
                position_idx = 1
                
                base_price = filled_order.get('fill_price', filled_order['price'])
                
                # Adjust based on orderbook imbalance
                imbalance = ob_analysis.get('imbalance_ratio', 0)
                if imbalance < -0.2:  # Strong ask pressure
                    price_adjustment = 0.998  # Slightly more aggressive
                else:
                    price_adjustment = 0.997  # Standard grid spacing
                
                new_price = base_price * price_adjustment
            
            # Calculate intelligent position size
            new_size = self.calculate_position_size_kelly(new_price, new_side)
            
            # Check if we should reduce position
            account_info = self.get_account_balance()
            current_position = account_info.get('position_size', 0)
            reduce_only = False
            
            if abs(current_position) > 0.001:
                reduce_only = (current_position > 0 and new_side == 'sell') or \
                             (current_position < 0 and new_side == 'buy')
            
            # Place replacement order
            result = self.bybit.create_limit_order(
                symbol=self.symbol,
                side=new_side,
                amount=new_size,
                price=new_price,
                params={
                    'timeInForce': 'GTC',
                    'postOnly': True,
                    'positionIdx': position_idx,
                    'reduceOnly': reduce_only
                }
            )
            
            self.active_orders[result['id']] = {
                'order_id': result['id'],
                'side': new_side,
                'price': new_price,
                'size': new_size,
                'level': filled_order['level'],
                'type': 'intelligent_replacement',
                'reduce_only': reduce_only,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Intelligent replacement: {new_side} {new_size:.4f} @ ${new_price:.4f} {'(reduceOnly)' if reduce_only else ''}")
            
        except Exception as e:
            logger.error(f"Error placing intelligent replacement: {e}")
    
    def get_account_balance(self) -> Dict:
        """Get account balance and position info"""
        try:
            balance = self.bybit.fetch_balance()
            positions = self.bybit.fetch_positions([self.symbol])
            
            usdt_balance = balance['USDT']['free']
            position_info = positions[0] if positions else {}
            
            self.current_position = position_info.get('contracts', 0)
            
            return {
                'balance': usdt_balance,
                'position_size': self.current_position,
                'liquidation_price': position_info.get('liquidationPrice'),
                'unrealized_pnl': position_info.get('unrealizedPnl', 0),
                'margin_ratio': position_info.get('marginRatio', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'balance': 0.0, 'position_size': 0, 'liquidation_price': None, 'unrealized_pnl': 0, 'margin_ratio': 0}
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily PnL"""
        try:
            if len(self.daily_pnl_history) < 7:
                return 0.0
            
            daily_returns = list(self.daily_pnl_history)
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming 365 trading days)
            sharpe = (avg_return / std_return) * np.sqrt(365)
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, current_equity: float) -> float:
        """Calculate maximum drawdown"""
        try:
            self.peak_equity = max(self.peak_equity, current_equity)
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            return current_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def risk_management_check(self) -> Dict:
        """Comprehensive risk management check"""
        try:
            account_info = self.get_account_balance()
            current_equity = account_info['balance'] + account_info.get('unrealized_pnl', 0)
            
            # Calculate risk metrics
            current_drawdown = self.calculate_max_drawdown(current_equity)
            margin_ratio = account_info.get('margin_ratio', 0)
            sharpe_ratio = self.calculate_sharpe_ratio()
            
            # Risk flags
            risk_flags = []
            
            # Drawdown check
            if current_drawdown > 0.15:  # 15% drawdown
                risk_flags.append('HIGH_DRAWDOWN')
            
            # Margin ratio check
            if margin_ratio > 0.8:  # 80% margin usage
                risk_flags.append('HIGH_MARGIN_USAGE')
            
            # Position size check
            position_value = abs(self.current_position) * account_info.get('position_size', 0)
            if position_value > self.capital * self.leverage * 0.9:
                risk_flags.append('OVERSIZED_POSITION')
            
            # Liquidation distance check
            liq_price = account_info.get('liquidation_price')
            if liq_price:
                current_price = list(self.price_history)[-1] if self.price_history else 0
                if current_price > 0:
                    liq_distance = abs(current_price - liq_price) / current_price
                    if liq_distance < 0.05:  # Less than 5% from liquidation
                        risk_flags.append('LIQUIDATION_RISK')
            
            return {
                'current_drawdown': current_drawdown,
                'max_drawdown': self.max_drawdown,
                'margin_ratio': margin_ratio,
                'sharpe_ratio': sharpe_ratio,
                'risk_flags': risk_flags,
                'equity': current_equity
            }
            
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
            return {'risk_flags': ['ERROR']}
    
    def get_enhanced_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status"""
        try:
            # Get basic account info
            account_info = self.get_account_balance()
            market_data = asyncio.run(self.get_enhanced_market_data()) if self.price_history else {}
            
            current_price = market_data.get('price', 0)
            funding_rate = market_data.get('funding_rate', 0)
            
            # Calculate performance metrics
            current_equity = account_info['balance'] + account_info.get('unrealized_pnl', 0)
            total_return = (current_equity - self.capital) / self.capital * 100
            
            # Risk metrics
            risk_metrics = self.risk_management_check()
            
            # Trading metrics
            if self.trades_count > 0:
                avg_profit_per_trade = self.total_profit / self.trades_count
                win_rate = len([o for o in self.filled_orders if o.get('profit', 0) > 0]) / len(self.filled_orders) if self.filled_orders else 0
            else:
                avg_profit_per_trade = 0
                win_rate = 0
            
            # Funding rate analysis
            funding_collected = self.funding_fees_collected
            estimated_daily_funding = funding_rate * abs(self.current_position) * current_price * 3  # 3 times per day
            
            return {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'funding_rate': funding_rate,
                'volatility_regime': self.volatility_regime,
                'market_regime': self.market_regime,
                'active_orders': len(self.active_orders),
                'total_trades': self.trades_count,
                'current_position': self.current_position,
                'account_balance': account_info['balance'],
                'unrealized_pnl': account_info.get('unrealized_pnl', 0),
                'realized_profit': self.total_profit,
                'total_equity': current_equity,
                'total_return_pct': total_return,
                'funding_collected': funding_collected,
                'estimated_daily_funding': estimated_daily_funding,
                'maker_rebates': self.maker_rebates,
                'avg_profit_per_trade': avg_profit_per_trade,
                'win_rate': win_rate,
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'current_drawdown': risk_metrics.get('current_drawdown', 0),
                'margin_ratio': risk_metrics.get('margin_ratio', 0),
                'liquidation_price': account_info.get('liquidation_price'),
                'risk_flags': risk_metrics.get('risk_flags', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced portfolio status: {e}")
            return {}
    
    def cancel_all_orders(self) -> None:
        """Cancel all active orders"""
        cancelled_count = 0
        for order_id in list(self.active_orders.keys()):
            try:
                self.bybit.cancel_order(order_id, self.symbol)
                del self.active_orders[order_id]
                cancelled_count += 1
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
        
        logger.info(f"Cancelled {cancelled_count} orders")
    
    def emergency_shutdown(self) -> None:
        """Emergency shutdown with position closing"""
        try:
            logger.warning("EMERGENCY SHUTDOWN INITIATED")
            
            # Cancel all orders
            self.cancel_all_orders()
            
            # Close position if significant
            if abs(self.current_position) > 0.001:
                side = 'sell' if self.current_position > 0 else 'buy'
                try:
                    self.bybit.create_market_order(
                        symbol=self.symbol,
                        side=side,
                        amount=abs(self.current_position)
                    )
                    logger.info(f"Emergency position closed: {side} {abs(self.current_position):.4f}")
                except Exception as e:
                    logger.error(f"Error closing position: {e}")
                    
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
    
    def run_enhanced_bot(self) -> None:
        """Main enhanced trading loop"""
        logger.info("Starting Enhanced Futures Grid Trading Bot...")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Setup
            self.setup_leverage_and_margin()
            
            # Check account
            account_info = self.get_account_balance()
            if account_info['balance'] < self.capital:
                logger.error(f"Insufficient balance: ${account_info['balance']:.2f} < ${self.capital:.2f}")
                return
            
            # Initial grid placement
            self.place_enhanced_grid_orders()
            
            # Main loop
            loop_count = 0
            while self.is_running:
                try:
                    loop_count += 1
                    
                    # Monitor and replace orders
                    self.monitor_and_replace_orders()
                    
                    # Get portfolio status
                    status = self.get_enhanced_portfolio_status()
                    
                    # Risk management
                    risk_flags = status.get('risk_flags', [])
                    
                    # Log status every 10 loops (5 minutes)
                    if loop_count % 10 == 0:
                        logger.info("=" * 80)
                        logger.info(f"Enhanced Bot Status - Loop {loop_count}")
                        logger.info(f"Price: ${status.get('current_price', 0):.4f} | Regime: {status.get('market_regime', 'unknown')}/{status.get('volatility_regime', 'unknown')}")
                        logger.info(f"Active Orders: {status.get('active_orders', 0)} | Trades: {status.get('total_trades', 0)}")
                        logger.info(f"Position: {status.get('current_position', 0):.4f} | Balance: ${status.get('account_balance', 0):.2f}")
                        logger.info(f"Realized P&L: ${status.get('realized_profit', 0):.4f} | Unrealized: ${status.get('unrealized_pnl', 0):.4f}")
                        logger.info(f"Total Return: {status.get('total_return_pct', 0):.2f}% | Sharpe: {status.get('sharpe_ratio', 0):.2f}")
                        logger.info(f"Win Rate: {status.get('win_rate', 0):.1%} | Avg/Trade: ${status.get('avg_profit_per_trade', 0):.4f}")
                        logger.info(f"Funding Rate: {status.get('funding_rate', 0):.6f} | Est. Daily: ${status.get('estimated_daily_funding', 0):.2f}")
                        logger.info(f"Max DD: {status.get('max_drawdown', 0):.1%} | Current DD: {status.get('current_drawdown', 0):.1%}")
                        if risk_flags:
                            logger.warning(f"Risk Flags: {', '.join(risk_flags)}")
                        logger.info("=" * 80)
                    
                    # Handle risk flags
                    if 'LIQUIDATION_RISK' in risk_flags or 'HIGH_DRAWDOWN' in risk_flags:
                        logger.error("CRITICAL RISK DETECTED - EMERGENCY SHUTDOWN")
                        self.emergency_shutdown()
                        break
                    
                    if 'HIGH_MARGIN_USAGE' in risk_flags:
                        logger.warning("High margin usage - reducing grid size")
                        # Could implement grid size reduction here
                    
                    # Adaptive grid refresh every hour
                    if loop_count % 120 == 0:  # Every 60 minutes
                        logger.info("Performing adaptive grid refresh...")
                        self.cancel_all_orders()
                        time.sleep(5)
                        self.place_enhanced_grid_orders()
                    
                    # Update daily PnL tracking
                    if loop_count % 2880 == 0:  # Every 24 hours
                        daily_pnl = status.get('realized_profit', 0) + status.get('unrealized_pnl', 0)
                        self.daily_pnl_history.append(daily_pnl)
                    
                    # Sleep
                    time.sleep(30)  # 30 second intervals
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt - stopping bot...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait 1 minute on error
                    
        except Exception as e:
            logger.error(f"Fatal error in enhanced bot: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup and final reporting"""
        logger.info("Enhanced bot cleanup...")
        self.is_running = False
        
        # Cancel remaining orders
        self.cancel_all_orders()
        
        # Final status
        final_status = self.get_enhanced_portfolio_status()
        
        logger.info("=" * 80)
        logger.info("FINAL ENHANCED BOT REPORT")
        logger.info("=" * 80)
        logger.info(f"Runtime: {datetime.now() - self.start_time if self.start_time else 'Unknown'}")
        logger.info(f"Total Trades: {final_status.get('total_trades', 0)}")
        logger.info(f"Final Position: {final_status.get('current_position', 0):.4f}")
        logger.info(f"Total Return: {final_status.get('total_return_pct', 0):.2f}%")
        logger.info(f"Realized Profit: ${final_status.get('realized_profit', 0):.4f}")
        logger.info(f"Unrealized P&L: ${final_status.get('unrealized_pnl', 0):.4f}")
        logger.info(f"Funding Collected: ${final_status.get('funding_collected', 0):.4f}")
        logger.info(f"Maker Rebates: ${final_status.get('maker_rebates', 0):.4f}")
        logger.info(f"Win Rate: {final_status.get('win_rate', 0):.1%}")
        logger.info(f"Sharpe Ratio: {final_status.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {final_status.get('max_drawdown', 0):.1%}")
        logger.info("=" * 80)


def main():
    """Main function for enhanced futures grid trading"""
    # Enhanced configuration
    CONFIG = {
        'symbol': 'DOGE/USDT:USDT',
        'api_key': 'aPeo9so1VKnQt2Gm9x',
        'secret': 'TrfGjSSfgUBEJg4D4EErLGXBPo6HjcwQ5kuu',
        'capital': 6.3,
        'leverage': 75,
        'base_grid_levels': 8,
        'use_testnet': False,
        'max_position_ratio': 0.9
    }
    
    # Create enhanced trader
    enhanced_trader = EnhancedFuturesGridTrader(
        symbol=CONFIG['symbol'],
        api_key=CONFIG['api_key'],
        secret=CONFIG['secret'],
        capital=CONFIG['capital'],
        leverage=CONFIG['leverage'],
        base_grid_levels=CONFIG['base_grid_levels'],
        use_testnet=CONFIG['use_testnet'],
        max_position_ratio=CONFIG['max_position_ratio']
    )
    
    # Run enhanced bot
    enhanced_trader.run_enhanced_bot()


if __name__ == "__main__":
    main()