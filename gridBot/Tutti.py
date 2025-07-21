import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter
import asyncio
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
import heapq
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # trending, ranging, volatile, calm
    confidence: float
    characteristics: Dict
    optimal_strategy: str

@dataclass
class ProfitOpportunity:
    """Identified profit opportunity"""
    strategy: str
    expected_profit: float
    confidence: float
    entry_price: float
    exit_price: float
    size: float
    risk_reward: float
    time_horizon: int  # seconds
    fee_adjusted_profit: float
    is_maker: bool  # Whether to use maker orders

@dataclass
class HedgePosition:
    """Hedge position info"""
    side: str  # long or short
    size: float
    entry_price: float
    hedge_ratio: float
    parent_position: str  # ID of main position
    pnl: float

class FeeOptimizedAdaptiveBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'DOGE/USDT:USDT', testnet: bool = True):
        """
        Fee-Optimized Multi-Strategy Adaptive AI Bot with Hedge Mode
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
        
        # === FEE OPTIMIZATION TECHNIQUES ===
        
        # FEE STRUCTURE (Bybit)
        self.maker_fee = -0.0001  # -0.01% REBATE for makers
        self.taker_fee = 0.0006   # 0.06% fee for takers
        
        # 1. MAKER-ONLY STRATEGY
        self.force_maker_orders = True
        self.maker_spread_adjustment = 0.0001  # 0.01% minimum spread
        self.post_only_retry_attempts = 3
        self.maker_order_timeout = 30  # seconds
        
        # 2. FEE-AWARE PROFIT CALCULATION
        self.min_profit_after_fees = 0.0015  # 0.15% minimum profit after fees
        self.fee_adjusted_targets = True
        self.compound_maker_rebates = True
        
        # 3. LIQUIDITY PROVIDING OPTIMIZATION
        self.liquidity_provider_mode = True
        self.depth_level_targeting = 3  # Place orders at 3rd level of book
        self.spread_capture_ratio = 0.7  # Capture 70% of spread
        
        # 4. BATCH ORDER OPTIMIZATION
        self.batch_orders = True
        self.batch_size = 5  # Group 5 orders together
        self.order_queue = deque()
        
        # 5. NET POSITION MANAGEMENT (Hedge Mode)
        self.hedge_mode = True
        self.hedge_positions = {}
        self.main_positions = {}
        self.position_pairs = {}  # Track main-hedge pairs
        
        # 6. DYNAMIC HEDGE RATIOS
        self.dynamic_hedge_ratio = True
        self.min_hedge_ratio = 0.2  # 20% minimum hedge
        self.max_hedge_ratio = 0.8  # 80% maximum hedge
        self.hedge_adjustment_threshold = 0.1  # 10% change triggers adjustment
        
        # 7. SPREAD TRADING OPTIMIZATION
        self.spread_trading = True
        self.min_spread_profit = 0.0008  # 0.08% after fees
        self.spread_positions = {}
        
        # 8. VOLUME-WEIGHTED ENTRY/EXIT
        self.volume_weighted_execution = True
        self.vwap_period = 20  # bars
        self.execution_slices = 3  # Split orders into 3 parts
        
        # 9. FEE ARBITRAGE
        self.fee_arbitrage = True
        self.maker_taker_spread = self.taker_fee - self.maker_fee  # 0.07%
        self.min_fee_arb_profit = 0.0005  # 0.05% minimum
        
        # 10. REBATE MAXIMIZATION
        self.rebate_maximization = True
        self.daily_rebate_target = 0.01  # 1% of capital in rebates
        self.rebate_tracking = defaultdict(float)
        
        # === ENHANCED AI TECHNIQUES ===
        
        # Previous techniques (enhanced for fees)
        self.neural_pattern_engine = True
        self.pattern_memory_size = 1000
        self.pattern_success_threshold = 0.65
        self.learned_patterns = deque(maxlen=self.pattern_memory_size)
        self.pattern_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0, 'net_profit': 0})
        
        self.adaptive_strategy_switching = True
        self.regime_detection_window = 100
        self.strategy_performance_tracking = defaultdict(lambda: {'profit': 0, 'trades': 0, 'fees_paid': 0, 'rebates_earned': 0})
        self.current_regime = None
        self.regime_confidence_threshold = 0.7
        
        # Additional fee-aware parameters
        self.fee_aware_position_sizing = True
        self.fee_impact_multiplier = 1.5  # Increase size for maker orders
        self.high_frequency_mode = False  # Disabled to reduce fees
        
        # Core parameters
        self.total_investment = 0
        self.active_strategies = {}
        self.position_tracker = {}
        self.performance_history = deque(maxlen=1000)
        
        # Enhanced strategy weights (fee-optimized)
        self.strategy_weights = {
            'spread_capture': 0.25,      # High weight for maker rebates
            'liquidity_provision': 0.20,  # Earn rebates
            'pattern_maker': 0.15,        # Pattern trading with maker orders
            'hedge_arbitrage': 0.15,      # Hedge mode arbitrage
            'mean_reversion': 0.10,       # Good for ranging markets
            'fee_arbitrage': 0.10,        # Pure fee arbitrage
            'momentum': 0.05              # Lower weight due to taker fees
        }
        
        # Risk management
        self.max_strategies_concurrent = 7  # More strategies in hedge mode
        self.strategy_correlation_limit = 0.6
        self.global_stop_loss = 0.20  # 20% portfolio stop loss
        
        # Performance tracking
        self.total_fees_paid = 0
        self.total_rebates_earned = 0
        self.net_profit = 0
        self.gross_profit = 0
        
        # Hedge mode specific
        self.position_mode_status = None
        self.check_and_set_hedge_mode()
    
    def check_and_set_hedge_mode(self):
        """Enable hedge mode for independent long/short positions"""
        try:
            # First check current position mode
            try:
                position_mode = self.exchange.fetch_position_mode()
                self.position_mode_status = position_mode
                logger.info(f"Current position mode: {position_mode}")
            except:
                pass
            
            # Set to hedge mode
            self.exchange.set_position_mode(hedged=True, symbol=self.symbol)
            logger.info("✅ Hedge mode enabled - can hold both long and short positions")
            
        except Exception as e:
            logger.warning(f"Could not set hedge mode, may already be enabled: {e}")
    
    def calculate_fee_adjusted_profit(self, entry: float, exit: float, size: float, 
                                    is_maker_entry: bool, is_maker_exit: bool) -> Dict:
        """Calculate actual profit after fees"""
        gross_profit = (exit - entry) / entry * size * self.leverage
        
        # Calculate fees
        entry_fee = self.maker_fee * size if is_maker_entry else self.taker_fee * size
        exit_fee = self.maker_fee * size if is_maker_exit else self.taker_fee * size
        
        total_fees = entry_fee + exit_fee
        net_profit = gross_profit - total_fees
        
        # Include rebates (negative fees)
        rebates = 0
        if is_maker_entry and self.maker_fee < 0:
            rebates += abs(self.maker_fee) * size
        if is_maker_exit and self.maker_fee < 0:
            rebates += abs(self.maker_fee) * size
        
        return {
            'gross_profit': gross_profit,
            'total_fees': total_fees,
            'rebates': rebates,
            'net_profit': net_profit,
            'fee_impact': total_fees / gross_profit if gross_profit > 0 else 1.0
        }
    
    def place_maker_order_with_retry(self, side: str, price: float, size: float, 
                                   strategy_id: str, max_attempts: int = 3) -> Optional[Dict]:
        """Place maker order with retry logic to ensure fee rebate"""
        try:
            attempts = 0
            order = None
            
            while attempts < max_attempts:
                try:
                    # Adjust price to ensure maker order
                    orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
                    
                    if side == 'buy':
                        # Place below best bid to ensure maker
                        best_bid = orderbook['bids'][0][0]
                        adjusted_price = min(price, best_bid - self.maker_spread_adjustment)
                        
                        order = self.exchange.create_limit_buy_order(
                            symbol=self.symbol,
                            amount=size,
                            price=adjusted_price,
                            params={
                                'postOnly': True,  # Ensure maker order
                                'timeInForce': 'PostOnly',
                                'clientOrderId': f'{strategy_id}_{int(time.time()*1000)}'
                            }
                        )
                    else:
                        # Place above best ask to ensure maker
                        best_ask = orderbook['asks'][0][0]
                        adjusted_price = max(price, best_ask + self.maker_spread_adjustment)
                        
                        order = self.exchange.create_limit_sell_order(
                            symbol=self.symbol,
                            amount=size,
                            price=adjusted_price,
                            params={
                                'postOnly': True,
                                'timeInForce': 'PostOnly',
                                'clientOrderId': f'{strategy_id}_{int(time.time()*1000)}'
                            }
                        )
                    
                    logger.info(f"✅ Maker order placed: {side} {size} @ {adjusted_price:.4f} (rebate: ${abs(self.maker_fee) * size * adjusted_price:.4f})")
                    return order
                    
                except Exception as e:
                    if 'would not be maker' in str(e).lower() or 'post only' in str(e).lower():
                        attempts += 1
                        logger.warning(f"Order would be taker, retrying... ({attempts}/{max_attempts})")
                        time.sleep(0.1)
                    else:
                        raise e
            
            # If maker order fails, decide whether to skip or place taker
            if self.force_maker_orders:
                logger.warning(f"Could not place maker order after {max_attempts} attempts, skipping...")
                return None
            else:
                # Place as taker if necessary
                logger.warning(f"Placing as taker order (fee: ${self.taker_fee * size * price:.4f})")
                if side == 'buy':
                    order = self.exchange.create_market_buy_order(
                        symbol=self.symbol,
                        amount=size
                    )
                else:
                    order = self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=size
                    )
                return order
                
        except Exception as e:
            logger.error(f"Error placing maker order: {e}")
            return None
    
    def identify_spread_capture_opportunities(self) -> List[ProfitOpportunity]:
        """Identify opportunities to capture spread with maker rebates"""
        try:
            opportunities = []
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=20)
            
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            spread = best_ask - best_bid
            spread_percentage = spread / best_bid
            
            # Only trade if spread is wide enough
            min_required_spread = (self.min_profit_after_fees + abs(self.maker_fee) * 2) / self.spread_capture_ratio
            
            if spread_percentage > min_required_spread:
                # Calculate optimal entry points
                bid_depth = sum(vol for _, vol in orderbook['bids'][:3])
                ask_depth = sum(vol for _, vol in orderbook['asks'][:3])
                
                # Weighted midpoint
                weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / (bid_depth + ask_depth)
                
                # Buy side opportunity
                buy_price = best_bid + spread * 0.1  # 10% into spread
                sell_price = weighted_mid + spread * 0.1
                
                # Calculate fee-adjusted profit
                fee_calc = self.calculate_fee_adjusted_profit(
                    buy_price, sell_price, self.total_investment * 0.1,
                    is_maker_entry=True, is_maker_exit=True
                )
                
                if fee_calc['net_profit'] > 0:
                    opportunity = ProfitOpportunity(
                        strategy='spread_capture_long',
                        expected_profit=fee_calc['gross_profit'],
                        confidence=0.8,
                        entry_price=buy_price,
                        exit_price=sell_price,
                        size=self.total_investment * 0.1,
                        risk_reward=3.0,  # Spread trades have good R:R
                        time_horizon=300,  # 5 minutes
                        fee_adjusted_profit=fee_calc['net_profit'],
                        is_maker=True
                    )
                    opportunities.append(opportunity)
                
                # Short side opportunity (for hedge mode)
                if self.hedge_mode:
                    sell_price_entry = best_ask - spread * 0.1
                    buy_price_exit = weighted_mid - spread * 0.1
                    
                    fee_calc_short = self.calculate_fee_adjusted_profit(
                        sell_price_entry, buy_price_exit, self.total_investment * 0.1,
                        is_maker_entry=True, is_maker_exit=True
                    )
                    
                    if fee_calc_short['net_profit'] > 0:
                        opportunity = ProfitOpportunity(
                            strategy='spread_capture_short',
                            expected_profit=fee_calc_short['gross_profit'],
                            confidence=0.8,
                            entry_price=sell_price_entry,
                            exit_price=buy_price_exit,
                            size=self.total_investment * 0.1,
                            risk_reward=3.0,
                            time_horizon=300,
                            fee_adjusted_profit=fee_calc_short['net_profit'],
                            is_maker=True
                        )
                        opportunities.append(opportunity)
                
                if opportunities:
                    logger.info(f"💰 Spread capture opportunity: {spread_percentage*100:.3f}% spread (net profit: ${fee_calc['net_profit']:.4f})")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying spread opportunities: {e}")
            return []
    
    def create_hedge_positions(self, main_position: Dict) -> Optional[Dict]:
        """Create optimal hedge position for risk management and profit"""
        try:
            if not self.hedge_mode:
                return None
            
            # Calculate dynamic hedge ratio based on market conditions
            volatility = self.calculate_current_volatility()
            regime = self.detect_market_regime()
            
            # Base hedge ratio
            hedge_ratio = 0.5  # 50% default
            
            # Adjust based on volatility
            if volatility > 0.02:  # High volatility
                hedge_ratio = min(hedge_ratio * 1.5, self.max_hedge_ratio)
            elif volatility < 0.01:  # Low volatility
                hedge_ratio = max(hedge_ratio * 0.7, self.min_hedge_ratio)
            
            # Adjust based on regime
            if regime.regime_type == 'trending':
                hedge_ratio *= 0.8  # Less hedging in trends
            elif regime.regime_type == 'volatile':
                hedge_ratio *= 1.2  # More hedging in volatile markets
            
            # Calculate hedge size
            hedge_size = main_position['size'] * hedge_ratio
            
            # Determine hedge entry price
            current_price = self.get_current_price()
            
            if main_position['side'] == 'long':
                # Short hedge for long position
                hedge_price = current_price * 1.001  # Slightly above for maker order
                hedge_side = 'sell'
            else:
                # Long hedge for short position  
                hedge_price = current_price * 0.999  # Slightly below for maker order
                hedge_side = 'buy'
            
            # Place hedge order as maker
            hedge_order = self.place_maker_order_with_retry(
                side=hedge_side,
                price=hedge_price,
                size=hedge_size / current_price * self.leverage,
                strategy_id=f"hedge_{main_position['id']}"
            )
            
            if hedge_order:
                hedge_position = HedgePosition(
                    side='short' if hedge_side == 'sell' else 'long',
                    size=hedge_size,
                    entry_price=hedge_price,
                    hedge_ratio=hedge_ratio,
                    parent_position=main_position['id'],
                    pnl=0
                )
                
                self.hedge_positions[hedge_order['id']] = hedge_position
                self.position_pairs[main_position['id']] = hedge_order['id']
                
                logger.info(f"🛡️ Hedge position created: {hedge_ratio*100:.0f}% hedge @ ${hedge_price:.4f}")
                return hedge_order
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating hedge position: {e}")
            return None
    
    def optimize_hedge_ratios(self):
        """Dynamically adjust hedge ratios based on market conditions"""
        try:
            for main_id, hedge_id in self.position_pairs.items():
                if main_id in self.active_strategies and hedge_id in self.hedge_positions:
                    main_pos = self.active_strategies[main_id]
                    hedge_pos = self.hedge_positions[hedge_id]
                    
                    # Calculate current P&L
                    current_price = self.get_current_price()
                    
                    if main_pos['side'] == 'long':
                        main_pnl = (current_price - main_pos['entry_price']) / main_pos['entry_price']
                        hedge_pnl = (hedge_pos.entry_price - current_price) / hedge_pos.entry_price
                    else:
                        main_pnl = (main_pos['entry_price'] - current_price) / main_pos['entry_price']
                        hedge_pnl = (current_price - hedge_pos.entry_price) / hedge_pos.entry_price
                    
                    total_pnl = main_pnl + hedge_pnl * hedge_pos.hedge_ratio
                    
                    # Adjust hedge ratio if needed
                    if abs(total_pnl) < 0.001:  # Position is balanced
                        continue
                    
                    # Calculate optimal adjustment
                    if total_pnl > 0.005:  # Winning too much - reduce hedge
                        new_ratio = hedge_pos.hedge_ratio * 0.9
                    elif total_pnl < -0.003:  # Losing - increase hedge
                        new_ratio = hedge_pos.hedge_ratio * 1.1
                    else:
                        continue
                    
                    # Apply bounds
                    new_ratio = max(self.min_hedge_ratio, min(self.max_hedge_ratio, new_ratio))
                    
                    if abs(new_ratio - hedge_pos.hedge_ratio) > self.hedge_adjustment_threshold:
                        # Adjust hedge position
                        self.adjust_hedge_position(hedge_id, new_ratio)
                        
        except Exception as e:
            logger.error(f"Error optimizing hedge ratios: {e}")
    
    def adjust_hedge_position(self, hedge_id: str, new_ratio: float):
        """Adjust hedge position size"""
        try:
            hedge_pos = self.hedge_positions.get(hedge_id)
            if not hedge_pos:
                return
            
            main_pos = self.active_strategies.get(hedge_pos.parent_position)
            if not main_pos:
                return
            
            # Calculate new size
            new_size = main_pos['size'] * new_ratio
            size_change = new_size - hedge_pos.size
            
            current_price = self.get_current_price()
            contracts_change = abs(size_change) / current_price * self.leverage
            
            if size_change > 0:
                # Increase hedge
                order = self.place_maker_order_with_retry(
                    side='buy' if hedge_pos.side == 'long' else 'sell',
                    price=current_price * (1.001 if hedge_pos.side == 'short' else 0.999),
                    size=contracts_change,
                    strategy_id=f"hedge_adjust_{hedge_id}"
                )
            else:
                # Reduce hedge
                order = self.place_maker_order_with_retry(
                    side='sell' if hedge_pos.side == 'long' else 'buy',
                    price=current_price * (0.999 if hedge_pos.side == 'long' else 1.001),
                    size=contracts_change,
                    strategy_id=f"hedge_reduce_{hedge_id}"
                )
            
            if order:
                hedge_pos.size = new_size
                hedge_pos.hedge_ratio = new_ratio
                logger.info(f"⚖️ Hedge adjusted to {new_ratio*100:.0f}%")
                
        except Exception as e:
            logger.error(f"Error adjusting hedge position: {e}")
    
    def identify_fee_arbitrage_opportunities(self) -> List[ProfitOpportunity]:
        """Identify pure fee arbitrage opportunities"""
        try:
            opportunities = []
            
            # Get order book
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=10)
            
            # Look for situations where maker-taker spread can be captured
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            spread = best_ask - best_bid
            
            # Check if we can place maker orders on both sides
            # and capture fee differential when they fill
            if spread / best_bid > self.maker_taker_spread * 2:
                # Place buy maker order
                buy_price = best_bid + spread * 0.2
                # When filled, immediately place sell maker order
                sell_price = best_ask - spread * 0.2
                
                # Both orders earn rebates
                total_rebate = abs(self.maker_fee) * 2 * self.total_investment * 0.05
                
                # Even if prices don't move, we earn rebates
                fee_profit = total_rebate
                
                opportunity = ProfitOpportunity(
                    strategy='fee_arbitrage',
                    expected_profit=fee_profit,
                    confidence=0.9,  # High confidence in fee structure
                    entry_price=buy_price,
                    exit_price=sell_price,
                    size=self.total_investment * 0.05,
                    risk_reward=float('inf'),  # No risk if both are makers
                    time_horizon=600,  # 10 minutes
                    fee_adjusted_profit=fee_profit,
                    is_maker=True
                )
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding fee arbitrage: {e}")
            return []
    
    def execute_volume_weighted_entry(self, opportunity: ProfitOpportunity) -> List[Dict]:
        """Execute orders using VWAP to minimize market impact"""
        try:
            orders = []
            
            # Calculate VWAP levels
            df = self.fetch_ohlcv_data('1m', self.vwap_period)
            if df.empty:
                return orders
            
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            current_price = self.get_current_price()
            
            # Split order into slices
            slice_size = opportunity.size / self.execution_slices
            
            for i in range(self.execution_slices):
                # Calculate slice price based on VWAP
                if 'long' in opportunity.strategy:
                    # Buy below VWAP if possible
                    slice_price = min(opportunity.entry_price, vwap * (1 - 0.0001 * i))
                else:
                    # Sell above VWAP if possible
                    slice_price = max(opportunity.entry_price, vwap * (1 + 0.0001 * i))
                
                # Place maker order for slice
                contracts = self.calculate_position_size_for_amount(slice_price, slice_size)
                
                order = self.place_maker_order_with_retry(
                    side='buy' if 'long' in opportunity.strategy else 'sell',
                    price=slice_price,
                    size=contracts,
                    strategy_id=f"{opportunity.strategy}_slice_{i}"
                )
                
                if order:
                    orders.append(order)
                    # Small delay between slices
                    time.sleep(2)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error in volume weighted execution: {e}")
            return []
    
    def calculate_current_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            df = self.fetch_ohlcv_data('5m', 50)
            if df.empty:
                return 0.01
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.01
    
    def execute_batch_orders(self):
        """Execute queued orders in batches to optimize fees"""
        try:
            if len(self.order_queue) >= self.batch_size:
                batch = []
                for _ in range(self.batch_size):
                    if self.order_queue:
                        batch.append(self.order_queue.popleft())
                
                # Group by side
                buy_orders = [o for o in batch if o['side'] == 'buy']
                sell_orders = [o for o in batch if o['side'] == 'sell']
                
                # Execute buys
                if buy_orders:
                    # Find optimal price for all
                    avg_price = sum(o['price'] for o in buy_orders) / len(buy_orders)
                    total_size = sum(o['size'] for o in buy_orders)
                    
                    order = self.place_maker_order_with_retry(
                        side='buy',
                        price=avg_price,
                        size=total_size,
                        strategy_id='batch_buy'
                    )
                
                # Execute sells
                if sell_orders:
                    avg_price = sum(o['price'] for o in sell_orders) / len(sell_orders)
                    total_size = sum(o['size'] for o in sell_orders)
                    
                    order = self.place_maker_order_with_retry(
                        side='sell',
                        price=avg_price,
                        size=total_size,
                        strategy_id='batch_sell'
                    )
                    
        except Exception as e:
            logger.error(f"Error executing batch orders: {e}")
    
    def track_fee_performance(self, order_id: str, order_info: Dict):
        """Track fees and rebates for performance analysis"""
        try:
            # Get order details
            order = self.exchange.fetch_order(order_id, self.symbol)
            
            if order['status'] == 'closed':
                fee = order.get('fee', {})
                fee_cost = fee.get('cost', 0)
                fee_currency = fee.get('currency', 'USDT')
                
                # Check if maker or taker
                is_maker = order.get('takerOrMaker', 'taker') == 'maker'
                
                if is_maker and self.maker_fee < 0:
                    # Earned rebate
                    self.total_rebates_earned += abs(fee_cost)
                    self.rebate_tracking[datetime.now().date()] += abs(fee_cost)
                    logger.info(f"💰 Earned rebate: ${abs(fee_cost):.4f}")
                else:
                    # Paid fee
                    self.total_fees_paid += fee_cost
                    logger.info(f"💸 Paid fee: ${fee_cost:.4f}")
                
                # Update strategy tracking
                strategy = order_info.get('strategy', 'unknown')
                if is_maker:
                    self.strategy_performance_tracking[strategy]['rebates_earned'] += abs(fee_cost)
                else:
                    self.strategy_performance_tracking[strategy]['fees_paid'] += fee_cost
                    
        except Exception as e:
            logger.error(f"Error tracking fee performance: {e}")
    
    def display_fee_optimized_dashboard(self):
        """Display enhanced dashboard with fee metrics"""
        try:
            current_price = self.get_current_price()
            account_info = self.get_account_info()
            
            logger.info("\n" + "=" * 80)
            logger.info("💎 FEE-OPTIMIZED ADAPTIVE AI BOT DASHBOARD")
            logger.info("=" * 80)
            
            # Market state
            if self.current_regime:
                logger.info(f"📊 Market Regime: {self.current_regime.regime_type.upper()} ({self.current_regime.confidence:.2f})")
            
            logger.info(f"💱 Current Price: ${current_price:.4f}")
            logger.info(f"🔄 Position Mode: HEDGE MODE {'✅' if self.hedge_mode else '❌'}")
            
            # Account info
            if account_info:
                logger.info(f"💰 USDT Balance: ${account_info.get('usdt_balance', 0):.2f}")
            
            # Fee Performance
            logger.info(f"\n💸 FEE PERFORMANCE:")
            logger.info(f"  • Total Fees Paid: ${self.total_fees_paid:.4f}")
            logger.info(f"  • Total Rebates Earned: ${self.total_rebates_earned:.4f}")
            logger.info(f"  • Net Fee Impact: ${self.total_rebates_earned - self.total_fees_paid:.4f}")
            logger.info(f"  • Fee Efficiency: {(self.total_rebates_earned / (self.total_fees_paid + 0.01)) * 100:.1f}%")
            
            # Today's rebates
            today_rebates = self.rebate_tracking.get(datetime.now().date(), 0)
            logger.info(f"  • Today's Rebates: ${today_rebates:.4f} ({(today_rebates/self.total_investment)*100:.2f}% of capital)")
            
            # Profit Analysis
            self.gross_profit = sum(s['profit'] for s in self.strategy_performance_tracking.values())
            self.net_profit = self.gross_profit + self.total_rebates_earned - self.total_fees_paid
            
            logger.info(f"\n💵 PROFIT ANALYSIS:")
            logger.info(f"  • Gross Profit: ${self.gross_profit:.4f}")
            logger.info(f"  • Net Profit (after fees): ${self.net_profit:.4f}")
            logger.info(f"  • Profit Retention: {(self.net_profit/self.gross_profit)*100:.1f}%" if self.gross_profit > 0 else "N/A")
            
            # Active positions (Hedge Mode)
            main_positions = [p for p in self.active_strategies.values() if 'hedge' not in p.get('strategy_id', '')]
            hedge_positions = list(self.hedge_positions.values())
            
            logger.info(f"\n📈 ACTIVE POSITIONS:")
            logger.info(f"  • Main Positions: {len(main_positions)}")
            logger.info(f"  • Hedge Positions: {len(hedge_positions)}")
            logger.info(f"  • Total Positions: {len(main_positions) + len(hedge_positions)}")
            
            # Position details
            if main_positions:
                logger.info(f"\n  Main Positions:")
                for pos in main_positions[:5]:  # Show top 5
                    logger.info(f"    - {pos.get('strategy_id', 'unknown')}: ${pos['size']:.2f} ({pos['side']})")
            
            if hedge_positions:
                logger.info(f"\n  Hedge Positions:")
                for hedge in hedge_positions[:5]:
                    logger.info(f"    - Hedge {hedge.hedge_ratio*100:.0f}%: ${hedge.size:.2f} ({hedge.side})")
            
            # Strategy performance (fee-adjusted)
            logger.info(f"\n🏆 STRATEGY PERFORMANCE (Fee-Adjusted):")
            for strategy, perf in sorted(self.strategy_performance_tracking.items(), 
                                       key=lambda x: x[1]['profit'] - x[1]['fees_paid'] + x[1]['rebates_earned'], 
                                       reverse=True)[:5]:
                if perf['trades'] > 0:
                    net_profit = perf['profit'] - perf['fees_paid'] + perf['rebates_earned']
                    fee_impact = ((perf['fees_paid'] - perf['rebates_earned']) / (perf['profit'] + 0.01)) * 100
                    logger.info(f"  • {strategy}: Net ${net_profit:.4f} ({perf['trades']} trades, fee impact: {fee_impact:.1f}%)")
            
            # Maker order success rate
            total_orders = sum(s['trades'] for s in self.strategy_performance_tracking.values())
            maker_orders = int(total_orders * 0.8)  # Estimate based on strategy
            logger.info(f"\n📊 ORDER EXECUTION:")
            logger.info(f"  • Total Orders: {total_orders}")
            logger.info(f"  • Maker Orders: ~{maker_orders} ({(maker_orders/total_orders)*100:.1f}%)")
            logger.info(f"  • Avg Rebate per Order: ${self.total_rebates_earned/maker_orders:.6f}" if maker_orders > 0 else "N/A")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying dashboard: {e}")
    
    def execute_fee_optimized_opportunities(self, opportunities: List[ProfitOpportunity]):
        """Execute opportunities with fee optimization"""
        try:
            if not opportunities:
                return
            
            # Sort by fee-adjusted profit
            opportunities.sort(key=lambda x: x.fee_adjusted_profit * x.confidence, reverse=True)
            
            # Filter out opportunities that don't meet minimum profit after fees
            profitable_opps = [o for o in opportunities if o.fee_adjusted_profit > self.min_profit_after_fees * o.size]
            
            if not profitable_opps:
                logger.info("No opportunities meet minimum profit threshold after fees")
                return
            
            executed = 0
            total_expected_rebates = 0
            
            for opp in profitable_opps[:self.max_strategies_concurrent]:
                # Check if we should batch this order
                if self.batch_orders and opp.size < self.total_investment * 0.05:
                    self.order_queue.append({
                        'side': 'buy' if 'long' in opp.strategy else 'sell',
                        'price': opp.entry_price,
                        'size': opp.size / opp.entry_price * self.leverage,
                        'strategy': opp.strategy
                    })
                    logger.info(f"📦 Queued for batch execution: {opp.strategy}")
                else:
                    # Execute immediately
                    if self.volume_weighted_execution and opp.size > self.total_investment * 0.1:
                        # Use VWAP execution for large orders
                        orders = self.execute_volume_weighted_entry(opp)
                        success = len(orders) > 0
                    else:
                        # Regular maker order
                        success = self.execute_fee_optimized_trade(opp)
                    
                    if success:
                        executed += 1
                        if opp.is_maker:
                            total_expected_rebates += abs(self.maker_fee) * opp.size * 2  # Entry + exit
                        
                        # Create hedge if in hedge mode
                        if self.hedge_mode and opp.strategy not in ['spread_capture', 'fee_arbitrage']:
                            main_position = {
                                'id': opp.strategy,
                                'side': 'long' if 'long' in opp.strategy else 'short',
                                'size': opp.size,
                                'entry_price': opp.entry_price
                            }
                            self.create_hedge_positions(main_position)
            
            # Execute any pending batch orders
            if len(self.order_queue) >= self.batch_size:
                self.execute_batch_orders()
            
            if executed > 0:
                logger.info(f"✅ Executed {executed} fee-optimized opportunities")
                logger.info(f"💰 Expected rebates: ${total_expected_rebates:.4f}")
                
        except Exception as e:
            logger.error(f"Error executing opportunities: {e}")
    
    def execute_fee_optimized_trade(self, opportunity: ProfitOpportunity) -> bool:
        """Execute a single trade with fee optimization"""
        try:
            contracts = self.calculate_position_size_for_amount(opportunity.entry_price, opportunity.size)
            
            # Always try maker orders first
            if opportunity.is_maker:
                order = self.place_maker_order_with_retry(
                    side='buy' if 'long' in opportunity.strategy else 'sell',
                    price=opportunity.entry_price,
                    size=contracts,
                    strategy_id=opportunity.strategy,
                    max_attempts=self.post_only_retry_attempts
                )
            else:
                # For strategies that require immediate execution
                logger.warning(f"Using taker order for {opportunity.strategy} (fee impact: ${self.taker_fee * opportunity.size:.4f})")
                if 'long' in opportunity.strategy:
                    order = self.exchange.create_market_buy_order(
                        symbol=self.symbol,
                        amount=contracts
                    )
                else:
                    order = self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=contracts
                    )
            
            if order:
                # Track active strategy
                self.active_strategies[opportunity.strategy] = {
                    'order_id': order['id'],
                    'entry_price': opportunity.entry_price,
                    'exit_price': opportunity.exit_price,
                    'size': opportunity.size,
                    'contracts': contracts,
                    'side': 'long' if 'long' in opportunity.strategy else 'short',
                    'start_time': time.time(),
                    'time_horizon': opportunity.time_horizon,
                    'expected_profit': opportunity.expected_profit,
                    'fee_adjusted_profit': opportunity.fee_adjusted_profit,
                    'is_maker': opportunity.is_maker,
                    'status': 'pending',
                    'strategy_id': opportunity.strategy
                }
                
                # Track fees
                self.track_fee_performance(order['id'], {'strategy': opportunity.strategy})
                
                # Place exit order
                self.place_fee_optimized_exit_orders(opportunity, order['id'])
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def place_fee_optimized_exit_orders(self, opportunity: ProfitOpportunity, entry_order_id: str):
        """Place exit orders optimized for maker rebates"""
        try:
            strategy_info = self.active_strategies[opportunity.strategy]
            
            # Schedule exit order placement after entry fills
            def place_exits():
                time.sleep(5)  # Wait for entry to fill
                
                try:
                    # Check if entry filled
                    order = self.exchange.fetch_order(entry_order_id, self.symbol)
                    
                    if order['status'] == 'closed':
                        # Always try to exit as maker for rebate
                        if 'long' in opportunity.strategy:
                            # Place limit sell order for exit
                            exit_order = self.place_maker_order_with_retry(
                                side='sell',
                                price=opportunity.exit_price,
                                size=strategy_info['contracts'],
                                strategy_id=f"{opportunity.strategy}_exit"
                            )
                            
                            # Stop loss (can be taker if necessary)
                            stop_price = opportunity.entry_price * 0.995
                            sl_order = self.exchange.create_stop_limit_sell_order(
                                symbol=self.symbol,
                                amount=strategy_info['contracts'],
                                price=stop_price * 0.999,  # Slightly below to ensure fill
                                stopPrice=stop_price,
                                params={'reduceOnly': True}
                            )
                        else:
                            # Place limit buy order for exit
                            exit_order = self.place_maker_order_with_retry(
                                side='buy',
                                price=opportunity.exit_price,
                                size=strategy_info['contracts'],
                                strategy_id=f"{opportunity.strategy}_exit"
                            )
                            
                            # Stop loss
                            stop_price = opportunity.entry_price * 1.005
                            sl_order = self.exchange.create_stop_limit_buy_order(
                                symbol=self.symbol,
                                amount=strategy_info['contracts'],
                                price=stop_price * 1.001,
                                stopPrice=stop_price,
                                params={'reduceOnly': True}
                            )
                        
                        if exit_order:
                            strategy_info['tp_order_id'] = exit_order['id']
                        strategy_info['sl_order_id'] = sl_order['id']
                        strategy_info['status'] = 'active'
                        
                except Exception as e:
                    logger.error(f"Error placing exit orders: {e}")
            
            threading.Thread(target=place_exits, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error setting up exit orders: {e}")
    
    def monitor_and_optimize_positions(self):
        """Monitor positions and optimize for fees"""
        try:
            current_time = time.time()
            
            for strategy_name, strategy_info in list(self.active_strategies.items()):
                if strategy_info['status'] != 'active':
                    continue
                
                # Check if we can improve the exit price for better rebate
                if 'tp_order_id' in strategy_info:
                    try:
                        tp_order = self.exchange.fetch_order(strategy_info['tp_order_id'], self.symbol)
                        
                        if tp_order['status'] == 'open':
                            # Check if we can get a better price as maker
                            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
                            current_price = self.get_current_price()
                            
                            if strategy_info['side'] == 'long':
                                best_ask = orderbook['asks'][0][0]
                                # If we can place a better maker order
                                if best_ask - self.maker_spread_adjustment > tp_order['price']:
                                    # Cancel and replace
                                    self.exchange.cancel_order(strategy_info['tp_order_id'], self.symbol)
                                    new_exit = self.place_maker_order_with_retry(
                                        side='sell',
                                        price=best_ask - self.maker_spread_adjustment,
                                        size=strategy_info['contracts'],
                                        strategy_id=f"{strategy_name}_exit_improved"
                                    )
                                    if new_exit:
                                        strategy_info['tp_order_id'] = new_exit['id']
                                        logger.info(f"📈 Improved exit price for {strategy_name}: ${new_exit['price']:.4f}")
                        
                        elif tp_order['status'] == 'closed':
                            # Take profit hit
                            self.handle_closed_position(strategy_name, strategy_info, True)
                            
                    except Exception as e:
                        logger.error(f"Error checking TP order: {e}")
                
                # Check stop loss
                if 'sl_order_id' in strategy_info:
                    try:
                        sl_order = self.exchange.fetch_order(strategy_info['sl_order_id'], self.symbol)
                        if sl_order['status'] == 'closed':
                            # Stop loss hit
                            self.handle_closed_position(strategy_name, strategy_info, False)
                    except:
                        pass
                
                # Time-based exit optimization
                if current_time - strategy_info['start_time'] > strategy_info['time_horizon'] * 0.8:
                    # Close to time limit - try to exit as maker
                    current_price = self.get_current_price()
                    
                    if strategy_info['side'] == 'long':
                        exit_price = current_price * 1.0001  # Slightly above for maker
                    else:
                        exit_price = current_price * 0.9999  # Slightly below for maker
                    
                    # Cancel existing exit order
                    if 'tp_order_id' in strategy_info:
                        try:
                            self.exchange.cancel_order(strategy_info['tp_order_id'], self.symbol)
                        except:
                            pass
                    
                    # Place new maker exit
                    exit_order = self.place_maker_order_with_retry(
                        side='sell' if strategy_info['side'] == 'long' else 'buy',
                        price=exit_price,
                        size=strategy_info['contracts'],
                        strategy_id=f"{strategy_name}_time_exit"
                    )
                    
                    if exit_order:
                        strategy_info['tp_order_id'] = exit_order['id']
                        logger.info(f"⏰ Time-based exit optimization for {strategy_name}")
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def handle_closed_position(self, strategy_name: str, strategy_info: Dict, is_profit: bool):
        """Handle closed position and track performance"""
        try:
            # Calculate actual profit/loss
            current_price = self.get_current_price()
            
            if is_profit:
                exit_price = strategy_info.get('exit_price', current_price)
            else:
                exit_price = current_price
            
            # Calculate fee-adjusted profit
            fee_calc = self.calculate_fee_adjusted_profit(
                strategy_info['entry_price'],
                exit_price,
                strategy_info['size'],
                strategy_info.get('is_maker', False),
                strategy_info.get('is_maker', False)
            )
            
            # Update strategy performance
            base_strategy = strategy_name.split('_')[0]
            self.update_strategy_performance(base_strategy, fee_calc['net_profit'], is_profit)
            self.strategy_profits[strategy_name] = fee_calc['net_profit']
            
            # Track fee performance
            self.total_fees_paid += max(0, fee_calc['total_fees'])
            self.total_rebates_earned += fee_calc['rebates']
            
            logger.info(f"{'✅' if is_profit else '🛑'} Position closed: {strategy_name}")
            logger.info(f"  • Gross P/L: ${fee_calc['gross_profit']:.4f}")
            logger.info(f"  • Fees: ${fee_calc['total_fees']:.4f}")
            logger.info(f"  • Rebates: ${fee_calc['rebates']:.4f}")
            logger.info(f"  • Net P/L: ${fee_calc['net_profit']:.4f}")
            
            # Close hedge position if exists
            if self.hedge_mode and strategy_name in self.position_pairs:
                hedge_id = self.position_pairs[strategy_name]
                self.close_hedge_position(hedge_id)
            
            # Remove from active strategies
            del self.active_strategies[strategy_name]
            
        except Exception as e:
            logger.error(f"Error handling closed position: {e}")
    
    def close_hedge_position(self, hedge_id: str):
        """Close hedge position"""
        try:
            if hedge_id not in self.hedge_positions:
                return
            
            hedge_pos = self.hedge_positions[hedge_id]
            current_price = self.get_current_price()
            
            # Close at market or as maker if possible
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
            
            if hedge_pos.side == 'long':
                # Sell to close
                exit_price = orderbook['bids'][0][0] + self.maker_spread_adjustment
                order = self.place_maker_order_with_retry(
                    side='sell',
                    price=exit_price,
                    size=hedge_pos.size / current_price * self.leverage,
                    strategy_id=f"hedge_close_{hedge_id}"
                )
            else:
                # Buy to close
                exit_price = orderbook['asks'][0][0] - self.maker_spread_adjustment
                order = self.place_maker_order_with_retry(
                    side='buy',
                    price=exit_price,
                    size=hedge_pos.size / current_price * self.leverage,
                    strategy_id=f"hedge_close_{hedge_id}"
                )
            
            if order:
                # Calculate hedge P/L
                if hedge_pos.side == 'long':
                    hedge_pnl = (exit_price - hedge_pos.entry_price) / hedge_pos.entry_price * hedge_pos.size
                else:
                    hedge_pnl = (hedge_pos.entry_price - exit_price) / hedge_pos.entry_price * hedge_pos.size
                
                logger.info(f"🛡️ Hedge closed: ${hedge_pnl:.4f} P/L")
                
                del self.hedge_positions[hedge_id]
                
        except Exception as e:
            logger.error(f"Error closing hedge position: {e}")
    
    def run_fee_optimized_bot(self, check_interval: int = 5):
        """Main bot loop with fee optimization"""
        try:
            logger.info("💎 Starting Fee-Optimized Adaptive AI Bot...")
            logger.info("💰 Maker Rebate: {:.2f}%".format(abs(self.maker_fee) * 100))
            logger.info("💸 Taker Fee: {:.2f}%".format(self.taker_fee * 100))
            logger.info("🔄 Hedge Mode: ENABLED")
            logger.info("📊 Fee Optimization: ACTIVE")
            
            # Set leverage
            self.set_leverage()
            
            # Load previous state if exists
            self.load_bot_state()
            
            # Initialize
            last_opportunity_scan = time.time()
            last_performance_update = time.time()
            last_hedge_optimization = time.time()
            last_rebate_check = time.time()
            
            opportunity_scan_interval = 20  # Scan every 20 seconds
            performance_update_interval = 60  # Update every minute
            hedge_optimization_interval = 120  # Optimize hedges every 2 minutes
            rebate_check_interval = 300  # Check rebates every 5 minutes
            
            iteration = 0
            
            while True:
                try:
                    iteration += 1
                    current_time = time.time()
                    
                    # 1. Monitor active positions
                    self.monitor_and_optimize_positions()
                    
                    # 2. Scan for new opportunities
                    if current_time - last_opportunity_scan > opportunity_scan_interval:
                        logger.info(f"\n🔍 Scanning for fee-optimized opportunities... (iteration {iteration})")
                        
                        # Detect market regime
                        self.current_regime = self.detect_market_regime()
                        
                        # Collect opportunities
                        all_opportunities = []
                        
                        # Fee-optimized strategies
                        all_opportunities.extend(self.identify_spread_capture_opportunities())
                        all_opportunities.extend(self.identify_fee_arbitrage_opportunities())
                        
                        # Enhanced pattern recognition (maker orders)
                        pattern_opps = self.learn_and_recognize_patterns()
                        for opp in pattern_opps:
                            # Adjust for maker orders
                            opp.is_maker = True
                            fee_calc = self.calculate_fee_adjusted_profit(
                                opp.entry_price, opp.exit_price, opp.size, True, True
                            )
                            opp.fee_adjusted_profit = fee_calc['net_profit']
                        all_opportunities.extend(pattern_opps)
                        
                        # Other strategies (adjusted for fees)
                        other_opps = []
                        other_opps.extend(self.analyze_sentiment_and_flow_opportunities())
                        other_opps.extend(self.detect_psychological_levels())
                        other_opps.extend(self.multi_timeframe_confluence_analysis())
                        
                        # Calculate fee impact for each
                        for opp in other_opps:
                            opp.is_maker = True  # Try to execute as maker
                            fee_calc = self.calculate_fee_adjusted_profit(
                                opp.entry_price, opp.exit_price, opp.size, True, True
                            )
                            opp.fee_adjusted_profit = fee_calc['net_profit']
                        
                        all_opportunities.extend(other_opps)
                        
                        # Execute best opportunities
                        if all_opportunities:
                            logger.info(f"📋 Found {len(all_opportunities)} opportunities")
                            self.execute_fee_optimized_opportunities(all_opportunities)
                        
                        last_opportunity_scan = current_time
                    
                    # 3. Optimize hedge positions
                    if current_time - last_hedge_optimization > hedge_optimization_interval:
                        if self.hedge_mode and self.dynamic_hedge_ratio:
                            self.optimize_hedge_ratios()
                        last_hedge_optimization = current_time
                    
                    # 4. Check rebate targets
                    if current_time - last_rebate_check > rebate_check_interval:
                        today_rebates = self.rebate_tracking.get(datetime.now().date(), 0)
                        target_rebates = self.total_investment * self.daily_rebate_target
                        
                        if today_rebates < target_rebates * 0.5:  # Less than 50% of target
                            logger.info(f"⚠️ Below rebate target: ${today_rebates:.4f} / ${target_rebates:.4f}")
                            # Could increase spread capture activity here
                        
                        last_rebate_check = current_time
                    
                    # 5. Update performance metrics
                    if current_time - last_performance_update > performance_update_interval:
                        self.display_fee_optimized_dashboard()
                        last_performance_update = current_time
                    
                    # 6. Risk management
                    if iteration % 12 == 0:  # Every minute
                        self.check_risk_management()
                    
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval * 2)
                    
        except Exception as e:
            logger.error(f"Fatal error in bot: {e}")
            raise
        finally:
            self.cleanup()
    
    # Override parent methods for fee optimization
    def analyze_sentiment_and_flow_opportunities(self) -> List[ProfitOpportunity]:
        """Sentiment analysis with fee optimization"""
        opportunities = []
        sentiment = self.analyze_sentiment_and_flow()
        
        if abs(sentiment['sentiment']) > 0.3:
            current_price = self.get_current_price()
            
            # Adjust entry/exit for maker orders
            if sentiment['sentiment'] > 0:
                entry = current_price * 0.9998  # Below market for maker buy
                exit = current_price * 1.0025   # Higher target to cover fees
            else:
                entry = current_price * 1.0002  # Above market for maker sell
                exit = current_price * 0.9975   # Lower target to cover fees
            
            size = self.total_investment * 0.1
            
            # Calculate fee-adjusted profit
            fee_calc = self.calculate_fee_adjusted_profit(entry, exit, size, True, True)
            
            if fee_calc['net_profit'] > self.min_profit_after_fees * size:
                opp = ProfitOpportunity(
                    strategy='sentiment_long' if sentiment['sentiment'] > 0 else 'sentiment_short',
                    expected_profit=fee_calc['gross_profit'],
                    confidence=min(abs(sentiment['sentiment']), 0.9),
                    entry_price=entry,
                    exit_price=exit,
                    size=size,
                    risk_reward=abs(exit - entry) / (entry * 0.003),
                    time_horizon=1800,
                    fee_adjusted_profit=fee_calc['net_profit'],
                    is_maker=True
                )
                opportunities.append(opp)
        
        return opportunities
    
    # Include all parent class methods
    def detect_market_regime(self) -> MarketRegime:
        """Market regime detection (inherited)"""
        return super().detect_market_regime() if hasattr(super(), 'detect_market_regime') else MarketRegime('unknown', 0.0, {}, 'grid')
    
    def learn_and_recognize_patterns(self) -> List[ProfitOpportunity]:
        """Pattern recognition (inherited and enhanced)"""
        # Use parent implementation but adjust for fees
        return super().learn_and_recognize_patterns() if hasattr(super(), 'learn_and_recognize_patterns') else []
    
    def analyze_sentiment_and_flow(self) -> Dict:
        """Sentiment analysis (inherited)"""
        return super().analyze_sentiment_and_flow() if hasattr(super(), 'analyze_sentiment_and_flow') else {'sentiment': 0}
    
    def detect_psychological_levels(self) -> List[ProfitOpportunity]:
        """Psychological levels (inherited)"""
        return super().detect_psychological_levels() if hasattr(super(), 'detect_psychological_levels') else []
    
    def multi_timeframe_confluence_analysis(self) -> List[ProfitOpportunity]:
        """Multi-timeframe analysis (inherited)"""
        return super().multi_timeframe_confluence_analysis() if hasattr(super(), 'multi_timeframe_confluence_analysis') else []
    
    def update_strategy_performance(self, strategy: str, profit: float, win: bool):
        """Update strategy performance with fee tracking"""
        if strategy not in self.strategy_performance_tracking:
            self.strategy_performance_tracking[strategy] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0,
                'fees_paid': 0,
                'rebates_earned': 0
            }
        
        perf = self.strategy_performance_tracking[strategy]
        perf['trades'] += 1
        perf['profit'] += profit
        
        if win:
            perf['wins'] += 1
        else:
            perf['losses'] += 1
    
    def check_risk_management(self):
        """Risk management with hedge mode considerations"""
        try:
            # Calculate net exposure (main positions - hedge positions)
            main_exposure = sum(info['size'] for info in self.active_strategies.values() 
                              if 'hedge' not in info.get('strategy_id', ''))
            hedge_exposure = sum(hedge.size for hedge in self.hedge_positions.values())
            net_exposure = main_exposure - hedge_exposure
            
            # Check if we're over-leveraged
            max_exposure = self.total_investment * 2.0  # 200% max net exposure
            
            if abs(net_exposure) > max_exposure:
                logger.warning(f"⚠️ Net exposure limit exceeded: ${abs(net_exposure):.2f} > ${max_exposure:.2f}")
                self.reduce_net_exposure()
            
            # Check drawdown
            if hasattr(self, 'peak_balance'):
                current_balance = self.get_account_info().get('usdt_balance', self.total_investment)
                drawdown = (self.peak_balance - current_balance) / self.peak_balance
                
                if drawdown > self.global_stop_loss:
                    logger.warning(f"⚠️ Maximum drawdown reached: {drawdown*100:.1f}%")
                    self.emergency_close_all()
                    return
            
            # Update peak balance
            current_balance = self.get_account_info().get('usdt_balance', self.total_investment)
            if not hasattr(self, 'peak_balance'):
                self.peak_balance = current_balance
            else:
                self.peak_balance = max(self.peak_balance, current_balance)
                
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
    
    def reduce_net_exposure(self):
        """Reduce net exposure by adjusting positions"""
        try:
            # First try to increase hedges
            for main_id, strategy_info in self.active_strategies.items():
                if main_id not in self.position_pairs and 'hedge' not in strategy_info.get('strategy_id', ''):
                    # Create hedge for unhedged position
                    self.create_hedge_positions(strategy_info)
                    logger.info(f"🛡️ Added hedge for {main_id} to reduce exposure")
                    return
            
            # If still over-exposed, close smallest positions
            positions = sorted(self.active_strategies.items(), key=lambda x: x[1]['size'])
            
            for strategy_id, _ in positions[:2]:  # Close 2 smallest
                self.close_strategy_position(strategy_id)
                
        except Exception as e:
            logger.error(f"Error reducing exposure: {e}")
    
    def close_strategy_position(self, strategy_name: str):
        """Close position with fee optimization"""
        try:
            strategy_info = self.active_strategies.get(strategy_name)
            if not strategy_info:
                return
            
            # Cancel exit orders
            for order_field in ['tp_order_id', 'sl_order_id']:
                if order_field in strategy_info:
                    try:
                        self.exchange.cancel_order(strategy_info[order_field], self.symbol)
                    except:
                        pass
            
            # Try to close as maker for rebate
            current_price = self.get_current_price()
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
            
            if strategy_info['side'] == 'long':
                # Sell to close - place between bid and mid for quick maker fill
                best_bid = orderbook['bids'][0][0]
                close_price = best_bid + (current_price - best_bid) * 0.3
                
                close_order = self.place_maker_order_with_retry(
                    side='sell',
                    price=close_price,
                    size=strategy_info['contracts'],
                    strategy_id=f"{strategy_name}_close",
                    max_attempts=2  # Less attempts for closing
                )
            else:
                # Buy to close
                best_ask = orderbook['asks'][0][0]
                close_price = best_ask - (best_ask - current_price) * 0.3
                
                close_order = self.place_maker_order_with_retry(
                    side='buy',
                    price=close_price,
                    size=strategy_info['contracts'],
                    strategy_id=f"{strategy_name}_close",
                    max_attempts=2
                )
            
            if not close_order:
                # Fall back to market order if maker fails
                logger.warning(f"Using market order to close {strategy_name}")
                if strategy_info['side'] == 'long':
                    self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=strategy_info['contracts'],
                        params={'reduceOnly': True}
                    )
                else:
                    self.exchange.create_market_buy_order(
                        symbol=self.symbol,
                        amount=strategy_info['contracts'],
                        params={'reduceOnly': True}
                    )
            
            # Handle closed position
            self.handle_closed_position(strategy_name, strategy_info, False)
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def emergency_close_all(self):
        """Emergency close all positions"""
        try:
            logger.warning("🚨 EMERGENCY CLOSE ALL POSITIONS")
            
            # Close all main positions
            for strategy_name in list(self.active_strategies.keys()):
                self.close_strategy_position(strategy_name)
            
            # Close all hedge positions
            for hedge_id in list(self.hedge_positions.keys()):
                self.close_hedge_position(hedge_id)
            
            # Cancel all orders
            self.exchange.cancel_all_orders(self.symbol)
            
            logger.info("✅ All positions closed")
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
    
    def cleanup(self):
        """Cleanup on exit"""
        try:
            # Close all positions
            for strategy_name in list(self.active_strategies.keys()):
                self.close_strategy_position(strategy_name)
            
            # Save state
            self.save_bot_state()
            
            # Final performance report
            logger.info("\n" + "=" * 80)
            logger.info("📊 FINAL PERFORMANCE REPORT")
            logger.info("=" * 80)
            logger.info(f"Gross Profit: ${self.gross_profit:.4f}")
            logger.info(f"Total Fees Paid: ${self.total_fees_paid:.4f}")
            logger.info(f"Total Rebates Earned: ${self.total_rebates_earned:.4f}")
            logger.info(f"Net Profit: ${self.net_profit:.4f}")
            logger.info(f"Fee Efficiency: {(self.total_rebates_earned / (self.total_fees_paid + 0.01)) * 100:.1f}%")
            logger.info("=" * 80)
            
            logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    def save_bot_state(self):
        """Save bot state including fee metrics"""
        try:
            state = {
                'learned_patterns': list(self.learned_patterns)[-100:],
                'pattern_performance': dict(self.pattern_performance),
                'strategy_performance': dict(self.strategy_performance_tracking),
                'strategy_weights': self.strategy_weights,
                'total_fees_paid': self.total_fees_paid,
                'total_rebates_earned': self.total_rebates_earned,
                'rebate_tracking': dict(self.rebate_tracking),
                'timestamp': time.time()
            }
            
            with open('fee_optimized_bot_state.json', 'w') as f:
                json.dump(state, f, default=str)  # default=str for datetime serialization
            
            logger.info("💾 Bot state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_bot_state(self):
        """Load previous bot state"""
        try:
            with open('fee_optimized_bot_state.json', 'r') as f:
                state = json.load(f)
            
            self.learned_patterns = deque(state.get('learned_patterns', []), maxlen=self.pattern_memory_size)
            self.pattern_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0}, 
                                                 state.get('pattern_performance', {}))
            self.strategy_performance_tracking = defaultdict(
                lambda: {'profit': 0, 'trades': 0, 'fees_paid': 0, 'rebates_earned': 0},
                state.get('strategy_performance', {})
            )
            self.strategy_weights = state.get('strategy_weights', self.strategy_weights)
            self.total_fees_paid = state.get('total_fees_paid', 0)
            self.total_rebates_earned = state.get('total_rebates_earned', 0)
            
            # Convert rebate tracking dates back from strings
            rebate_tracking = state.get('rebate_tracking', {})
            self.rebate_tracking = defaultdict(float)
            for date_str, amount in rebate_tracking.items():
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    self.rebate_tracking[date] = amount
                except:
                    pass
            
            logger.info("💾 Bot state loaded from previous session")
            logger.info(f"   Previous fees paid: ${self.total_fees_paid:.4f}")
            logger.info(f"   Previous rebates earned: ${self.total_rebates_earned:.4f}")
            
        except FileNotFoundError:
            logger.info("No previous state found - starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    # Helper methods
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
    
    # Strategy profit tracking
    strategy_profits = defaultdict(float)


# USAGE EXAMPLE
if __name__ == "__main__":
    # Configuration
    API_KEY = "XP6EVrF9NhU5em6EVU"
    API_SECRET = "gqIUmxKZnIY7oXNYvazIvpyqO42EONoNnkLu"
    
    # Investment amount
    TOTAL_INVESTMENT = 99888  # USDT
    
    # Initialize fee-optimized bot
    fee_bot = FeeOptimizedAdaptiveBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='DOGE/USDT:USDT',
        testnet=True  # Set to False for live trading
    )
    
    # Configure bot parameters
    fee_bot.total_investment = TOTAL_INVESTMENT
    
    # Fee optimization settings
    fee_bot.force_maker_orders = True  # Always try to be maker
    fee_bot.min_profit_after_fees = 0.0015  # 0.15% minimum after fees
    fee_bot.hedge_mode = True  # Enable hedge mode
    fee_bot.dynamic_hedge_ratio = True  # Dynamic hedging
    
    # Strategy settings
    fee_bot.spread_trading = True
    fee_bot.fee_arbitrage = True
    fee_bot.liquidity_provider_mode = True
    fee_bot.volume_weighted_execution = True
    fee_bot.batch_orders = True
    
    # Risk parameters
    fee_bot.global_stop_loss = 0.20  # 20% maximum drawdown
    fee_bot.max_strategies_concurrent = 7
    fee_bot.min_hedge_ratio = 0.2
    fee_bot.max_hedge_ratio = 0.8
    
    # Performance targets
    fee_bot.daily_rebate_target = 0.01  # 1% of capital in daily rebates
    fee_bot.min_spread_profit = 0.0008  # 0.08% minimum spread profit
    
    try:
        logger.info("💎 Starting Fee-Optimized Adaptive AI Bot!")
        logger.info("🔄 Hedge Mode: ENABLED")
        logger.info("💰 Maker Rebates: ACTIVE")
        logger.info("📊 Fee Arbitrage: MONITORING")
        logger.info("🛡️ Dynamic Hedging: ENABLED")
        logger.info("⚡ 10+ Fee Optimization Techniques: ACTIVE")
        
        # Run bot
        fee_bot.run_fee_optimized_bot(check_interval=5)
        
    except KeyboardInterrupt:
        fee_bot.cleanup()
        logger.info("🛑 Bot stopped safely")