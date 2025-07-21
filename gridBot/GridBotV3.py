import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BybitGridBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = '1000PEPE/USDT:USDT', testnet: bool = True):
        """
        Initialize Bybit Grid Trading Bot with enhanced profitability features
        
        Args:
            api_key: Your Bybit API key
            api_secret: Your Bybit API secret
            symbol: Trading pair (default: BTC/USDT perpetual)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.symbol = symbol
        self.leverage = 50
        
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
            
        # Enhanced grid parameters
        self.grid_levels = 0
        self.grid_spacing = 0
        self.upper_price = 0
        self.lower_price = 0
        self.total_investment = 0
        self.order_amount = 0
        self.price_decimals = 2
        
        # Dynamic grid adjustment parameters
        self.dynamic_grid_enabled = True
        self.grid_shift_threshold = 0.15  # Shift grid when price moves 15% from center
        self.grid_expansion_factor = 1.2  # Expand grid by 20% in trending markets
        self.grid_contraction_factor = 0.8  # Contract grid by 20% in ranging markets
        
        # Enhanced volatility parameters
        self.atr_multiplier = 2.0
        self.min_grid_levels = 15
        self.max_grid_levels = 60
        self.volatility_window = deque(maxlen=24)  # Store 24 hours of volatility data
        
        # Multi-timeframe trend detection
        self.trend_mode = 'neutral'
        self.trend_threshold = 0.015  # 1.5% momentum threshold
        self.trend_buy_ratio = 0.6  # 50% for trend, 50% for grid
        self.trailing_stop_percent = 0.015  # 1.5% trailing stop
        self.trend_position = None
        self.highest_price_since_trend = 0
        self.trend_entry_price = 0
        
        # Scalping parameters for quick profits
        self.scalp_enabled = True
        self.scalp_threshold = 0.003  # 0.3% quick profit
        self.scalp_orders = {}
        
        # Order management
        self.buy_orders = {}
        self.sell_orders = {}
        self.pending_orders = deque(maxlen=100)  # Track recent order flow
        
        # FIXED: Performance tracking with proper initialization
        self.grid_profits = 0
        self.trend_profits = 0
        self.scalp_profits = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.profit_factor = 0
        self.realized_pnl = 0  # Track actual realized PnL
        self.unrealized_pnl = 0  # Track unrealized PnL
        
        # FIXED: Risk management with proper balance tracking
        self.max_drawdown = 0.1  # 10% max drawdown
        self.daily_profit_target = 0.40 # 40% daily target
        self.stop_loss_enabled = True
        self.initial_balance = 0
        self.highest_balance = 0
        self.starting_balance = 0  # Balance when bot started
        self.session_start_time = time.time()
        
        # Market microstructure analysis
        self.order_flow_imbalance = deque(maxlen=100)
        self.bid_ask_spread_history = deque(maxlen=100)
        self.volume_profile = {}
        
        # Advanced features
        self.mean_reversion_enabled = True
        self.momentum_acceleration_enabled = True
        self.smart_order_routing = True
        self.anti_manipulation_enabled = True
        
        # Fee optimization
        self.maker_fee = 0.0001  # 0.01%
        self.taker_fee = 0.0006  # 0.06%
        self.always_post_only = True  # Always try to be maker
        
        # Initialize position mode
        self.check_position_mode()
        
    def get_account_info(self) -> Dict:
        """FIXED: Get accurate account balance and position info"""
        try:
            # Fetch balance
            balance = self.exchange.fetch_balance()
            
            # Get USDT balance (includes unrealized PnL)
            usdt_info = balance.get('USDT', {})
            free_balance = usdt_info.get('free', 0)
            used_balance = usdt_info.get('used', 0)
            total_balance = usdt_info.get('total', free_balance + used_balance)
            
            # Fetch positions for accurate PnL
            positions = self.exchange.fetch_positions([self.symbol])
            
            # Calculate unrealized PnL from positions
            unrealized_pnl = 0
            position_info = []
            
            for pos in positions:
                if pos['contracts'] != 0:  # Active position
                    # Get position details
                    contracts = pos['contracts']
                    side = pos['side']
                    entry_price = pos['info'].get('avgPrice', 0)
                    mark_price = pos['markPrice']
                    
                    # Calculate unrealized PnL
                    if side == 'long':
                        pos_pnl = (mark_price - float(entry_price)) * contracts
                    else:  # short
                        pos_pnl = (float(entry_price) - mark_price) * abs(contracts)
                    
                    unrealized_pnl += pos_pnl
                    
                    position_info.append({
                        'side': side,
                        'contracts': contracts,
                        'entry_price': float(entry_price),
                        'mark_price': mark_price,
                        'unrealized_pnl': pos_pnl
                    })
            
            # Calculate wallet balance (excluding unrealized PnL)
            wallet_balance = total_balance - unrealized_pnl
            
            info = {
                'usdt_balance': total_balance,  # Total balance including unrealized PnL
                'wallet_balance': wallet_balance,  # Balance without unrealized PnL
                'free_balance': free_balance,
                'used_balance': used_balance,
                'unrealized_pnl': unrealized_pnl,
                'positions': position_info
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {
                'usdt_balance': 0,
                'wallet_balance': 0,
                'free_balance': 0,
                'used_balance': 0,
                'unrealized_pnl': 0,
                'positions': []
            }
    
    def manage_risk(self) -> bool:
        """FIXED: Accurate risk management with proper drawdown calculation"""
        try:
            # Get current account info
            account_info = self.get_account_info()
            wallet_balance = account_info.get('wallet_balance', 0)  # Use wallet balance for accurate tracking
            unrealized_pnl = account_info.get('unrealized_pnl', 0)
            
            # Initialize starting balance if not set
            if self.starting_balance == 0:
                self.starting_balance = wallet_balance
                self.initial_balance = wallet_balance
                self.highest_balance = wallet_balance
                logger.info(f"💰 Starting balance initialized: ${self.starting_balance:.2f}")
                
            # Update highest balance (using wallet balance, not total balance)
            if wallet_balance > self.highest_balance:
                self.highest_balance = wallet_balance
                logger.info(f"📈 New highest balance: ${self.highest_balance:.2f}")
                
            # FIXED: Calculate drawdown from highest point
            if self.highest_balance > 0:
                current_drawdown = (self.highest_balance - wallet_balance) / self.highest_balance
            else:
                current_drawdown = 0
            
            # Calculate profit from starting balance
            profit_from_start = (wallet_balance - self.starting_balance) / self.starting_balance if self.starting_balance > 0 else 0
            
            # Log current status
            if abs(current_drawdown) > 0.01:  # Only log if drawdown > 1%
                logger.info(f"📊 Risk Status: Balance: ${wallet_balance:.2f}, Drawdown: {current_drawdown*100:.2f}%, Profit: {profit_from_start*100:.2f}%")
            
            # Check max drawdown
            if current_drawdown > self.max_drawdown:
                logger.warning(f"⚠️ HIGH DRAWDOWN: {current_drawdown*100:.2f}% (Max allowed: {self.max_drawdown*100:.0f}%)")
                logger.warning(f"   Current: ${wallet_balance:.2f}, Peak: ${self.highest_balance:.2f}")
                # Can implement emergency measures here if needed
                # return False  # Uncomment to stop trading on high drawdown
                
            # Check daily profit target
            session_hours = (time.time() - self.session_start_time) / 3600
            if session_hours >= 24:  # Reset daily tracking
                self.session_start_time = time.time()
                self.initial_balance = wallet_balance
                
            daily_profit = (wallet_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
            
            if daily_profit >= self.daily_profit_target:
                logger.info(f"🎯 Daily profit target reached: {daily_profit*100:.2f}%")
                # Reduce position sizes after hitting target
                self.order_amount *= 0.5
                
            # Position size check
            if account_info.get('positions'):
                total_position_value = 0
                for pos in account_info['positions']:
                    if pos.get('contracts', 0) != 0:
                        total_position_value += abs(pos['contracts'] * pos.get('mark_price', 0))
                        
                # Check if overexposed
                if wallet_balance > 0:
                    exposure_ratio = total_position_value / wallet_balance
                    if exposure_ratio > 33:  # More than 10x account size
                        logger.warning(f"High exposure detected: {exposure_ratio:.2f}x")
                        return False  # Don't place new orders
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return True  # Continue trading but log error
    
    def calculate_performance_metrics(self) -> Dict:
        """FIXED: Calculate accurate performance metrics"""
        try:
            # Get current account info
            account_info = self.get_account_info()
            wallet_balance = account_info.get('wallet_balance', 0)
            total_balance = account_info.get('usdt_balance', 0)
            unrealized_pnl = account_info.get('unrealized_pnl', 0)
            
            # Calculate total profits (realized only)
            total_profits = self.grid_profits + self.trend_profits + self.scalp_profits
            
            # Calculate profit factor
            if self.total_trades > 0:
                win_rate = self.winning_trades / self.total_trades
                avg_win = total_profits / self.winning_trades if self.winning_trades > 0 else 0
                losing_trades = self.total_trades - self.winning_trades
                avg_loss = abs(total_profits / losing_trades) if losing_trades > 0 else 0
                
                if avg_loss > 0:
                    self.profit_factor = avg_win / avg_loss
                else:
                    self.profit_factor = float('inf') if avg_win > 0 else 0
            else:
                win_rate = 0
                self.profit_factor = 0
                
            # FIXED: Calculate returns from starting balance
            if self.starting_balance > 0:
                # Total return includes both realized and unrealized
                total_return = ((wallet_balance - self.starting_balance) / self.starting_balance) * 100
                
                # Calculate drawdown from highest point
                if self.highest_balance > wallet_balance:
                    current_drawdown = ((self.highest_balance - wallet_balance) / self.highest_balance) * 100
                else:
                    current_drawdown = 0
            else:
                total_return = 0
                current_drawdown = 0
                
            return {
                'total_profits': round(total_profits, 2),
                'grid_profits': round(self.grid_profits, 2),
                'trend_profits': round(self.trend_profits, 2),
                'scalp_profits': round(self.scalp_profits, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': round(win_rate * 100, 1),
                'profit_factor': round(self.profit_factor, 2),
                'total_return': round(total_return, 2),
                'current_drawdown': round(current_drawdown, 2),
                'wallet_balance': round(wallet_balance, 2),
                'total_balance': round(total_balance, 2),
                'starting_balance': round(self.starting_balance, 2),
                'highest_balance': round(self.highest_balance, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_profits': 0,
                'grid_profits': 0,
                'trend_profits': 0,
                'scalp_profits': 0,
                'unrealized_pnl': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'current_drawdown': 0,
                'wallet_balance': 0,
                'total_balance': 0,
                'starting_balance': 0,
                'highest_balance': 0
            }
    
    def check_filled_orders(self):
        """FIXED: Enhanced order checking with accurate profit tracking"""
        try:
            if self.trend_mode != 'neutral':
                return
                
            # Fetch recent closed orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=50)
            
            for order in closed_orders:
                order_id = order['id']
                
                # Check buy orders
                if order_id in self.buy_orders and self.buy_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.buy_orders[order_id]['status'] = 'filled'
                        
                        # Place sell order one grid level up
                        new_sell_price = filled_price + self.grid_spacing
                        
                        # Add small randomization to avoid order clustering
                        new_sell_price *= (1 + np.random.uniform(-0.0001, 0.0001))
                        
                        if new_sell_price <= self.upper_price:
                            sell_amount = self.calculate_position_size(new_sell_price, self.order_amount)
                            sell_order = self.place_smart_order('sell', new_sell_price, sell_amount)
                            
                            if sell_order:
                                self.sell_orders[sell_order['id']] = {
                                    'price': new_sell_price,
                                    'amount': sell_amount,
                                    'status': 'open',
                                    'placed_at': time.time(),
                                    'pair_buy_price': filled_price  # Track paired buy price
                                }
                                
                        # Calculate expected profit (for display only)
                        fee_cost = filled_price * filled_amount * self.maker_fee * 2  # Buy and sell fees
                        expected_profit = (self.grid_spacing * filled_amount) - fee_cost
                        
                        logger.info(f"✅ Buy filled @ ${filled_price:.{self.price_decimals}f}, sell placed @ ${new_sell_price:.{self.price_decimals}f}")
                        logger.info(f"   Expected profit: ${expected_profit:.2f}")
                        
                # Check sell orders
                elif order_id in self.sell_orders and self.sell_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.sell_orders[order_id]['status'] = 'filled'
                        
                        # Calculate actual profit if this was a paired trade
                        if 'pair_buy_price' in self.sell_orders[order_id]:
                            buy_price = self.sell_orders[order_id]['pair_buy_price']
                            
                            # FIXED: Accurate profit calculation
                            # For perpetual contracts: profit = (exit_price - entry_price) * contracts
                            gross_profit = (filled_price - buy_price) * filled_amount
                            
                            # Calculate fees
                            buy_fee = buy_price * filled_amount * self.maker_fee
                            sell_fee = filled_price * filled_amount * self.maker_fee
                            total_fees = buy_fee + sell_fee
                            
                            # Net profit
                            net_profit = gross_profit - total_fees
                            
                            # Add to total grid profits
                            self.grid_profits += net_profit
                            self.total_trades += 1
                            if net_profit > 0:
                                self.winning_trades += 1
                                
                            logger.info(f"💰 Grid profit realized: ${net_profit:.2f}")
                            logger.info(f"   Price diff: ${filled_price - buy_price:.{self.price_decimals}f}, Amount: {filled_amount:.4f}")
                            logger.info(f"   Gross: ${gross_profit:.2f}, Fees: ${total_fees:.2f}")
                            
                        # Place buy order one grid level down
                        new_buy_price = filled_price - self.grid_spacing
                        new_buy_price *= (1 + np.random.uniform(-0.0001, 0.0001))
                        
                        if new_buy_price >= self.lower_price:
                            buy_amount = self.calculate_position_size(new_buy_price, self.order_amount)
                            buy_order = self.place_smart_order('buy', new_buy_price, buy_amount)
                            
                            if buy_order:
                                self.buy_orders[buy_order['id']] = {
                                    'price': new_buy_price,
                                    'amount': buy_amount,
                                    'status': 'open',
                                    'placed_at': time.time()
                                }
                                
                        logger.info(f"✅ Sell filled @ ${filled_price:.{self.price_decimals}f}, buy placed @ ${new_buy_price:.{self.price_decimals}f}")
                        
            # Check scalp orders separately
            self.check_scalp_orders()
            
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
    
    def run(self, check_interval: int = 10, recalibration_interval: int = 3600):
        """FIXED: Run the bot with accurate balance tracking"""
        try:
            # Set leverage
            self.set_leverage()
            
            # Get initial balance before any trading
            account_info = self.get_account_info()
            self.starting_balance = account_info.get('wallet_balance', 0)
            self.initial_balance = self.starting_balance
            self.highest_balance = self.starting_balance
            
            logger.info(f"💰 Starting wallet balance: ${self.starting_balance:.2f}")
            logger.info(f"💳 Total balance (with unrealized): ${account_info.get('usdt_balance', 0):.2f}")
            
            # Initial risk check
            if not self.manage_risk():
                logger.error("Risk check failed. Bot will not start.")
                return
            
            # Place initial grid orders
            self.place_grid_orders()
            
            logger.info(f"🤖 Enhanced Grid Bot Started")
            logger.info(f"  - Check interval: {check_interval}s")
            logger.info(f"  - Recalibration: every {recalibration_interval/3600:.1f} hours")
            logger.info(f"  - Grid allocation: {(1-self.trend_buy_ratio)*100:.0f}%")
            logger.info(f"  - Trend allocation: {self.trend_buy_ratio*100:.0f}%")
            logger.info(f"  - Scalping: {'Enabled' if self.scalp_enabled else 'Disabled'}")
            logger.info(f"  - Risk limits: {self.max_drawdown*100:.0f}% max drawdown, {self.daily_profit_target*100:.1f}% daily target")
            
            last_recalibration = time.time()
            last_trend_check = time.time()
            last_performance_log = time.time()
            last_optimization = time.time()
            
            trend_check_interval = 30  # Check trends every 30 seconds
            performance_log_interval = 300  # Log performance every 5 minutes
            optimization_interval = 1800  # Optimize every 30 minutes
            
            while True:
                try:
                    # Risk management check
                    if not self.manage_risk():
                        logger.error("Risk limit reached. Stopping bot.")
                        break
                        
                    # Check for strong trends
                    if time.time() - last_trend_check > trend_check_interval:
                        self.check_and_manage_trend()
                        last_trend_check = time.time()
                        
                    # Normal grid operations (only if not in trend mode)
                    if self.trend_mode == 'neutral':
                        # Check filled orders
                        self.check_filled_orders()
                        
                        # Check grid boundaries
                        self.check_grid_boundaries()
                        
                        # Refresh scalp orders periodically
                        if int(time.time()) % 120 == 0:  # Every 2 minutes
                            self.place_scalp_orders()
                            
                    # Performance optimization
                    if time.time() - last_optimization > optimization_interval:
                        self.optimize_grid_performance()
                        last_optimization = time.time()
                        
                    # Log performance metrics
                    if time.time() - last_performance_log > performance_log_interval:
                        metrics = self.calculate_performance_metrics()
                        
                        logger.info(f"📊 PERFORMANCE UPDATE:")
                        logger.info(f"  💰 Realized P&L: ${metrics.get('total_profits', 0):.2f}")
                        logger.info(f"  📊 Unrealized P&L: ${metrics.get('unrealized_pnl', 0):.2f}")
                        logger.info(f"  📈 Total Return: {metrics.get('total_return', 0):.2f}%")
                        logger.info(f"  🎯 Win Rate: {metrics.get('win_rate', 0):.1f}%")
                        logger.info(f"  📊 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                        logger.info(f"  💳 Wallet Balance: ${metrics.get('wallet_balance', 0):.2f}")
                        logger.info(f"  💰 Total Balance: ${metrics.get('total_balance', 0):.2f}")
                        
                        if metrics.get('current_drawdown', 0) > 1:  # Only show if > 1%
                            logger.warning(f"  ⚠️ Drawdown: {metrics.get('current_drawdown', 0):.1f}%")
                            
                        # Breakdown by strategy
                        logger.info(f"  Strategy Breakdown:")
                        logger.info(f"    - Grid: ${metrics.get('grid_profits', 0):.2f}")
                        logger.info(f"    - Trend: ${metrics.get('trend_profits', 0):.2f}")
                        logger.info(f"    - Scalp: ${metrics.get('scalp_profits', 0):.2f}")
                        
                        # Current mode
                        if self.trend_mode != 'neutral':
                            logger.info(f"  ⚡ MODE: TREND FOLLOWING")
                        else:
                            open_buys = sum(1 for o in self.buy_orders.values() if o['status'] == 'open')
                            open_sells = sum(1 for o in self.sell_orders.values() if o['status'] == 'open')
                            logger.info(f"  📊 MODE: GRID TRADING ({open_buys} buys, {open_sells} sells)")
                            
                        last_performance_log = time.time()
                        
                    # Periodic grid recalibration (only in neutral mode)
                    if self.trend_mode == 'neutral' and time.time() - last_recalibration > recalibration_interval:
                        logger.info("🔄 Performing periodic grid recalibration...")
                        
                        # Cancel current orders
                        self.exchange.cancel_all_orders(self.symbol)
                        self.buy_orders.clear()
                        self.sell_orders.clear()
                        
                        # Reconfigure grid
                        self.auto_configure_grid(self.total_investment)
                        
                        # Place new orders
                        self.place_grid_orders()
                        
                        last_recalibration = time.time()
                        logger.info("✅ Grid recalibration complete")
                        
                    # Sleep before next iteration
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval * 2)  # Longer sleep on error
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            # Final cleanup
            self.stop()
            
            # Final performance report
            logger.info("=" * 50)
            logger.info("FINAL PERFORMANCE REPORT")
            logger.info("=" * 50)
            
            metrics = self.calculate_performance_metrics()
            
            logger.info(f"Starting Balance: ${metrics.get('starting_balance', 0):.2f}")
            logger.info(f"Final Wallet Balance: ${metrics.get('wallet_balance', 0):.2f}")
            logger.info(f"Total Balance (incl. unrealized): ${metrics.get('total_balance', 0):.2f}")
            logger.info(f"Highest Balance: ${metrics.get('highest_balance', 0):.2f}")
            logger.info(f"")
            logger.info(f"Realized Profits: ${metrics.get('total_profits', 0):.2f}")
            logger.info(f"Unrealized P&L: ${metrics.get('unrealized_pnl', 0):.2f}")
            logger.info(f"Total Return: {metrics.get('total_return', 0):.2f}%")
            logger.info(f"Max Drawdown: {metrics.get('current_drawdown', 0):.2f}%")
            logger.info(f"")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            logger.info("\nStrategy Performance:")
            logger.info(f"  Grid Trading: ${metrics.get('grid_profits', 0):.2f}")
            logger.info(f"  Trend Following: ${metrics.get('trend_profits', 0):.2f}")
            logger.info(f"  Scalping: ${metrics.get('scalp_profits', 0):.2f}")
            
            logger.info("=" * 50)

    def check_position_mode(self):
        """Check and set appropriate position mode for trading"""
        try:
            self.exchange.set_position_mode(hedged=False, symbol=self.symbol)
            logger.info("Position mode set to one-way mode")
        except Exception as e:
            error_message = str(e).lower()
            if "position mode not modified" in error_message or "not modified" in error_message:
                logger.info("Position mode already set correctly")
            else:
                logger.info(f"Position mode check completed: {e}")
            pass
            
    def analyze_order_book(self) -> Dict:
        """
        Analyze order book for better entry/exit points
        """
        try:
            order_book = self.exchange.fetch_order_book(self.symbol, limit=20)
            
            # Calculate bid/ask imbalance
            total_bid_volume = sum([bid[1] for bid in order_book['bids'][:10]])
            total_ask_volume = sum([ask[1] for ask in order_book['asks'][:10]])
            
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Calculate spread
            spread = order_book['asks'][0][0] - order_book['bids'][0][0]
            spread_percent = (spread / order_book['bids'][0][0]) * 100
            
            # Detect walls
            bid_wall = max(order_book['bids'][:5], key=lambda x: x[1])[0] if order_book['bids'] else 0
            ask_wall = min(order_book['asks'][:5], key=lambda x: x[1])[0] if order_book['asks'] else 0
            
            # Store for analysis
            self.order_flow_imbalance.append(imbalance)
            self.bid_ask_spread_history.append(spread_percent)
            
            return {
                'imbalance': imbalance,
                'spread_percent': spread_percent,
                'bid_wall': bid_wall,
                'ask_wall': ask_wall,
                'bid_pressure': total_bid_volume > total_ask_volume * 1.2,
                'ask_pressure': total_ask_volume > total_bid_volume * 1.2
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book: {e}")
            return {
                'imbalance': 0,
                'spread_percent': 0.1,
                'bid_wall': 0,
                'ask_wall': 0,
                'bid_pressure': False,
                'ask_pressure': False
            }
            
    def calculate_dynamic_grid_params(self) -> Dict:
        """
        Calculate dynamic grid parameters based on multiple factors
        """
        try:
            # Fetch multi-timeframe data
            df_5m = self.fetch_ohlcv_data('5m', 288)  # 24 hours
            df_15m = self.fetch_ohlcv_data('15m', 96)  # 24 hours
            df_1h = self.fetch_ohlcv_data('1h', 168)  # 1 week
            
            # Calculate volatility metrics
            volatility_5m = df_5m['close'].pct_change().std() * np.sqrt(288)
            volatility_15m = df_15m['close'].pct_change().std() * np.sqrt(96)
            volatility_1h = df_1h['close'].pct_change().std() * np.sqrt(24)
            
            # Weighted volatility
            current_volatility = (volatility_5m * 0.5 + volatility_15m * 0.3 + volatility_1h * 0.2)
            self.volatility_window.append(current_volatility)
            
            # Calculate trend strength
            sma_20 = df_1h['close'].rolling(20).mean().iloc[-1]
            sma_50 = df_1h['close'].rolling(50).mean().iloc[-1]
            current_price = df_1h['close'].iloc[-1]
            
            trend_strength = abs(current_price - sma_50) / sma_50
            
            # Determine market regime
            if current_price > sma_20 > sma_50:
                market_regime = 'bullish'
            elif current_price < sma_20 < sma_50:
                market_regime = 'bearish'
            else:
                market_regime = 'ranging'
                
            # Calculate support/resistance levels
            support_levels = self.find_support_resistance(df_1h, 'support')
            resistance_levels = self.find_support_resistance(df_1h, 'resistance')
            
            # Adjust grid parameters based on analysis
            base_range = current_price * current_volatility * self.atr_multiplier
            
            # Dynamic adjustments
            if market_regime == 'ranging':
                # Tighter grid in ranging market
                range_multiplier = 0.8
                optimal_levels = 40
            elif market_regime in ['bullish', 'bearish']:
                # Wider grid in trending market
                range_multiplier = 1.2
                optimal_levels = 25
            else:
                range_multiplier = 1.0
                optimal_levels = 30
                
            # Adjust based on volatility trend
            if len(self.volatility_window) > 10:
                recent_vol = np.mean(list(self.volatility_window)[-5:])
                older_vol = np.mean(list(self.volatility_window)[-20:-10])
                
                if recent_vol > older_vol * 1.2:  # Increasing volatility
                    range_multiplier *= 1.1
                    optimal_levels = int(optimal_levels * 0.8)
                elif recent_vol < older_vol * 0.8:  # Decreasing volatility
                    range_multiplier *= 0.9
                    optimal_levels = int(optimal_levels * 1.2)
                    
            # Final calculations
            adjusted_range = base_range * range_multiplier
            upper_price = current_price + (adjusted_range / 2)
            lower_price = current_price - (adjusted_range / 2)
            
            # Snap to support/resistance if close
            if support_levels:
                closest_support = min(support_levels, key=lambda x: abs(x - lower_price))
                if abs(closest_support - lower_price) / lower_price < 0.01:  # Within 1%
                    lower_price = closest_support
                    
            if resistance_levels:
                closest_resistance = min(resistance_levels, key=lambda x: abs(x - upper_price))
                if abs(closest_resistance - upper_price) / upper_price < 0.01:  # Within 1%
                    upper_price = closest_resistance
                    
            # Ensure profitable spacing
            min_spacing_percent = 0.12  # 0.12% minimum for profit after fees
            min_spacing = current_price * (min_spacing_percent / 100)
            
            grid_spacing = (upper_price - lower_price) / (optimal_levels - 1)
            if grid_spacing < min_spacing:
                optimal_levels = int((upper_price - lower_price) / min_spacing) + 1
                grid_spacing = (upper_price - lower_price) / (optimal_levels - 1)
                
            # Ensure levels are within bounds
            optimal_levels = max(self.min_grid_levels, min(self.max_grid_levels, optimal_levels))
            
            return {
                'upper_price': upper_price,
                'lower_price': lower_price,
                'grid_levels': optimal_levels,
                'grid_spacing': grid_spacing,
                'market_regime': market_regime,
                'volatility': current_volatility,
                'trend_strength': trend_strength,
                'support_levels': support_levels[:3],  # Top 3 support levels
                'resistance_levels': resistance_levels[:3]  # Top 3 resistance levels
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic grid params: {e}")
            # Return safe defaults
            current_price = self.get_current_price()
            return {
                'upper_price': current_price * 1.02,
                'lower_price': current_price * 0.98,
                'grid_levels': 25,
                'grid_spacing': current_price * 0.0008,
                'market_regime': 'unknown',
                'volatility': 0.02,
                'trend_strength': 0,
                'support_levels': [],
                'resistance_levels': []
            }
            
    def find_support_resistance(self, df: pd.DataFrame, level_type: str = 'support') -> List[float]:
        """
        Find support and resistance levels using price action
        """
        try:
            # Find local minima and maxima
            window = 5
            if level_type == 'support':
                levels = df['low'].iloc[np.where(
                    (df['low'].shift(window) > df['low']) & 
                    (df['low'].shift(-window) > df['low'])
                )[0]].values
            else:
                levels = df['high'].iloc[np.where(
                    (df['high'].shift(window) < df['high']) & 
                    (df['high'].shift(-window) < df['high'])
                )[0]].values
                
            # Cluster nearby levels
            if len(levels) > 0:
                clustered = []
                sorted_levels = sorted(levels)
                
                current_cluster = [sorted_levels[0]]
                for level in sorted_levels[1:]:
                    if level / current_cluster[-1] - 1 < 0.005:  # Within 0.5%
                        current_cluster.append(level)
                    else:
                        clustered.append(np.mean(current_cluster))
                        current_cluster = [level]
                        
                if current_cluster:
                    clustered.append(np.mean(current_cluster))
                    
                # Sort by relevance (frequency)
                level_counts = {}
                for level in clustered:
                    count = sum(1 for l in levels if abs(l - level) / level < 0.005)
                    level_counts[level] = count
                    
                sorted_levels = sorted(level_counts.keys(), key=lambda x: level_counts[x], reverse=True)
                return sorted_levels[:5]  # Return top 5 levels
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return []
            
    def place_smart_order(self, order_type: str, price: float, amount: float) -> Optional[Dict]:
        """
        Place orders with smart routing and anti-manipulation features
        """
        try:
            # Analyze order book
            book_analysis = self.analyze_order_book()
            
            # Adjust price based on order book
            if self.smart_order_routing:
                if order_type == 'buy':
                    # If strong ask pressure, place order slightly lower
                    if book_analysis['ask_pressure']:
                        price = price * 0.9995
                    # If bid wall nearby, place just above it
                    elif book_analysis['bid_wall'] and abs(book_analysis['bid_wall'] - price) / price < 0.001:
                        price = book_analysis['bid_wall'] + (price * 0.0001)
                else:  # sell
                    # If strong bid pressure, place order slightly higher
                    if book_analysis['bid_pressure']:
                        price = price * 1.0005
                    # If ask wall nearby, place just below it
                    elif book_analysis['ask_wall'] and abs(book_analysis['ask_wall'] - price) / price < 0.001:
                        price = book_analysis['ask_wall'] - (price * 0.0001)
                        
            # Round price
            price = round(price, self.price_decimals)
            
            # Check for manipulation
            if self.anti_manipulation_enabled:
                # Detect potential wash trading or spoofing
                if len(self.order_flow_imbalance) > 10:
                    recent_imbalances = list(self.order_flow_imbalance)[-10:]
                    volatility_imbalance = np.std(recent_imbalances)
                    
                    # High volatility in order flow might indicate manipulation
                    if volatility_imbalance > 0.3:
                        logger.warning("Potential market manipulation detected, delaying order")
                        time.sleep(np.random.uniform(1, 3))  # Random delay
                        
            # Place order with appropriate parameters
            params = {
                'timeInForce': 'PostOnly' if self.always_post_only else 'GTC',
                'reduceOnly': False
            }
            
            if order_type == 'buy':
                order = self.exchange.create_limit_buy_order(
                    symbol=self.symbol,
                    amount=amount,
                    price=price,
                    params=params
                )
                logger.info(f"Smart buy order placed: {amount:.4f} @ ${price:.{self.price_decimals}f}")
            else:
                order = self.exchange.create_limit_sell_order(
                    symbol=self.symbol,
                    amount=amount,
                    price=price,
                    params=params
                )
                logger.info(f"Smart sell order placed: {amount:.4f} @ ${price:.{self.price_decimals}f}")
                
            return order
            
        except Exception as e:
            # If PostOnly fails, try without it
            if "post only" in str(e).lower():
                try:
                    params = {'timeInForce': 'GTC', 'reduceOnly': False}
                    if order_type == 'buy':
                        order = self.exchange.create_limit_buy_order(
                            symbol=self.symbol,
                            amount=amount,
                            price=price,
                            params=params
                        )
                    else:
                        order = self.exchange.create_limit_sell_order(
                            symbol=self.symbol,
                            amount=amount,
                            price=price,
                            params=params
                        )
                    logger.info(f"Order placed without PostOnly: {amount:.4f} @ ${price:.{self.price_decimals}f}")
                    return order
                except Exception as e2:
                    logger.error(f"Error placing order: {e2}")
                    return None
            else:
                logger.error(f"Error placing smart order: {e}")
                return None
                
    def place_scalp_orders(self):
        """
        Place quick scalp orders for small, fast profits
        """
        if not self.scalp_enabled:
            return
            
        try:
            current_price = self.get_current_price()
            book_analysis = self.analyze_order_book()
            
            # Only scalp in low spread conditions
            if book_analysis['spread_percent'] > 0.05:  # Skip if spread > 0.05%
                return
                
            # Calculate scalp levels
            scalp_buy_price = current_price * (1 - self.scalp_threshold)
            scalp_sell_price = current_price * (1 + self.scalp_threshold)
            
            # Use smaller amounts for scalping
            scalp_amount = self.order_amount * 0.5
            position_size = self.calculate_position_size(current_price, scalp_amount)
            
            # Place scalp orders
            buy_order = self.place_smart_order('buy', scalp_buy_price, position_size)
            if buy_order:
                self.scalp_orders[buy_order['id']] = {
                    'type': 'buy',
                    'price': scalp_buy_price,
                    'target': scalp_sell_price,
                    'amount': position_size
                }
                
            sell_order = self.place_smart_order('sell', scalp_sell_price, position_size)
            if sell_order:
                self.scalp_orders[sell_order['id']] = {
                    'type': 'sell',
                    'price': scalp_sell_price,
                    'target': scalp_buy_price,
                    'amount': position_size
                }
                
        except Exception as e:
            logger.error(f"Error placing scalp orders: {e}")
            
    def check_scalp_orders(self):
        """
        Check and manage scalp orders for quick profits
        """
        try:
            if not self.scalp_orders:
                return
                
            # Get recent orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=20)
            
            for order in closed_orders:
                order_id = order['id']
                
                if order_id in self.scalp_orders and order['status'] == 'closed':
                    scalp_info = self.scalp_orders[order_id]
                    
                    # Place immediate counter order
                    if scalp_info['type'] == 'buy':
                        # Place immediate sell at target
                        counter_order = self.place_smart_order(
                            'sell',
                            scalp_info['target'],
                            scalp_info['amount']
                        )
                        
                        if counter_order:
                            # Track potential profit
                            potential_profit = (scalp_info['target'] - scalp_info['price']) * scalp_info['amount'] * self.leverage
                            logger.info(f"Scalp buy filled, sell placed. Potential profit: ${potential_profit:.2f}")
                            
                    # Remove from tracking
                    del self.scalp_orders[order_id]
                    
        except Exception as e:
            logger.error(f"Error checking scalp orders: {e}")
            
    def detect_momentum_acceleration(self) -> Dict:
        """
        Detect momentum acceleration for enhanced trend following
        """
        try:
            df = self.fetch_ohlcv_data('5m', 60)  # Last 5 hours
            
            # Calculate momentum at different intervals
            momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            momentum_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            momentum_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            # Calculate acceleration
            recent_momentum = momentum_5
            older_momentum = (df['close'].iloc[-10] - df['close'].iloc[-15]) / df['close'].iloc[-15]
            acceleration = recent_momentum - older_momentum
            
            # Volume analysis
            recent_volume = df['volume'].iloc[-5:].mean()
            older_volume = df['volume'].iloc[-20:-10].mean()
            volume_increase = recent_volume / older_volume if older_volume > 0 else 1
            
            # Determine if momentum is accelerating
            is_accelerating_up = (
                momentum_5 > 0 and
                momentum_10 > 0 and
                momentum_5 > momentum_10 and
                acceleration > 0.001 and
                volume_increase > 1.2
            )
            
            is_accelerating_down = (
                momentum_5 < 0 and
                momentum_10 < 0 and
                momentum_5 < momentum_10 and
                acceleration < -0.001 and
                volume_increase > 1.2
            )
            
            return {
                'is_accelerating_up': is_accelerating_up,
                'is_accelerating_down': is_accelerating_down,
                'acceleration': acceleration,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'volume_increase': volume_increase
            }
            
        except Exception as e:
            logger.error(f"Error detecting momentum acceleration: {e}")
            return {
                'is_accelerating_up': False,
                'is_accelerating_down': False,
                'acceleration': 0,
                'momentum_5': 0,
                'momentum_10': 0,
                'momentum_20': 0,
                'volume_increase': 1
            }
            
    def calculate_position_size(self, price: float, usdt_amount: float) -> float:
        """
        Calculate position size with proper risk management
        """
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
            
            if position_size < min_size:
                position_size = min_size
                
        except Exception as e:
            logger.warning(f"Could not get market info: {e}")
            position_size = round(position_size, 3)
            
        return position_size
        
    def manage_risk(self) -> bool:
        """
        Comprehensive risk management system
        
        Returns:
            bool: True if trading should continue, False if should stop
        """
        try:
            # Get current account info
            account_info = self.get_account_info()
            current_balance = account_info.get('usdt_balance', 0)
            
            if self.initial_balance == 0:
                self.initial_balance = current_balance
                self.highest_balance = current_balance
                
            # Update highest balance
            if current_balance > self.highest_balance:
                self.highest_balance = current_balance
                
            # FIXED: Improved drawdown calculation
            # Only calculate drawdown if we've actually had a higher balance than current
            # and if the difference is significant (more than 0.5%)
            if self.highest_balance > current_balance and \
               ((self.highest_balance - current_balance) / self.highest_balance) > 0.005:
                current_drawdown = (self.highest_balance - current_balance) / self.highest_balance
            else:
                current_drawdown = 0  # No significant drawdown
            
            # Check max drawdown but only log a warning instead of emergency stop
            if current_drawdown > self.max_drawdown:
                logger.warning(f"MAX DRAWDOWN REACHED: {current_drawdown*100:.2f}% (Current: ${current_balance:.2f}, Peak: ${self.highest_balance:.2f})")
                # Emergency stop removed as requested
                # self.emergency_stop()
                # return False
                
            # Check daily profit target
            daily_profit = (current_balance - self.initial_balance) / self.initial_balance
            if daily_profit >= self.daily_profit_target:
                logger.info(f"🎯 Daily profit target reached: {daily_profit*100:.2f}%")
                # Reduce position sizes or pause trading
                self.order_amount *= 0.5  # Reduce risk after hitting target
                
            # Position size check
            if account_info.get('positions'):
                total_position_value = 0
                for pos in account_info['positions']:
                    if pos['contracts'] != 0:
                        total_position_value += abs(pos['contracts'] * pos['markPrice'])
                        
                # Check if overexposed
                exposure_ratio = total_position_value / current_balance
                if exposure_ratio > 10:  # More than 2x account size
                    logger.warning(f"High exposure detected: {exposure_ratio:.2f}x")
                    # Don't place new orders until exposure reduces
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return True  # Continue trading but log error
            
    def emergency_stop(self):
        """
        Emergency stop - close all positions and cancel all orders
        """
        try:
            logger.error("EMERGENCY STOP ACTIVATED")
            
            # Cancel all orders
            self.exchange.cancel_all_orders(self.symbol)
            
            # Close all positions
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['contracts'] != 0:
                    if pos['side'] == 'long':
                        self.exchange.create_market_sell_order(
                            self.symbol,
                            abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                    else:
                        self.exchange.create_market_buy_order(
                            self.symbol,
                            abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                        
            logger.error("All positions closed and orders cancelled")
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            
    def optimize_grid_performance(self):
        """
        Continuously optimize grid performance based on results
        """
        try:
            # Calculate performance metrics
            if self.total_trades > 0:
                win_rate = self.winning_trades / self.total_trades
                
                # Adjust grid spacing based on win rate
                if win_rate < 0.4:  # Low win rate
                    # Increase spacing to improve probability
                    self.grid_spacing *= 1.1
                    logger.info(f"Increasing grid spacing due to low win rate: {win_rate*100:.1f}%")
                elif win_rate > 0.7:  # High win rate
                    # Decrease spacing to capture more profits
                    self.grid_spacing *= 0.95
                    logger.info(f"Decreasing grid spacing due to high win rate: {win_rate*100:.1f}%")
                    
            # Adjust based on profit factor
            if self.total_trades > 10:
                if self.profit_factor < 1.2:
                    # Need to improve edge
                    self.trend_threshold *= 0.9  # Be more selective with trends
                    self.scalp_threshold *= 1.1  # Wider scalp targets
                elif self.profit_factor > 2:
                    # Can be more aggressive
                    self.order_amount *= 1.1  # Increase position sizes
                    
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
            
    def fetch_ohlcv_data(self, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for analysis
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
            
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
        
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            raise
            
    def detect_strong_trend(self) -> Dict:
        """
        Enhanced trend detection with multiple confirmations
        """
        try:
            # Fetch multi-timeframe data
            df_5m = self.fetch_ohlcv_data('5m', 100)
            df_15m = self.fetch_ohlcv_data('15m', 100)
            df_1h = self.fetch_ohlcv_data('1h', 50)
            
            # Calculate EMAs for trend
            df_15m['ema_9'] = df_15m['close'].ewm(span=9, adjust=False).mean()
            df_15m['ema_21'] = df_15m['close'].ewm(span=21, adjust=False).mean()
            df_15m['ema_55'] = df_15m['close'].ewm(span=55, adjust=False).mean()
            
            # Calculate momentum
            momentum_5 = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-5]) / df_15m['close'].iloc[-5]
            momentum_10 = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-10]) / df_15m['close'].iloc[-10]
            momentum_20 = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-20]) / df_15m['close'].iloc[-20]
            
            # RSI calculation
            delta = df_1h['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD
            exp1 = df_15m['close'].ewm(span=12, adjust=False).mean()
            exp2 = df_15m['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            
            # Volume confirmation
            volume_sma = df_15m['volume'].rolling(20).mean()
            current_volume_ratio = df_15m['volume'].iloc[-1] / volume_sma.iloc[-1]
            
            # Price action
            current_price = df_15m['close'].iloc[-1]
            ema_9 = df_15m['ema_9'].iloc[-1]
            ema_21 = df_15m['ema_21'].iloc[-1]
            ema_55 = df_15m['ema_55'].iloc[-1]
            
            # Enhanced trend criteria
            strong_uptrend = (
                current_price > ema_9 > ema_21 > ema_55 and
                momentum_5 > self.trend_threshold and
                momentum_10 > self.trend_threshold * 0.7 and
                current_rsi > 55 and current_rsi < 75 and
                macd_histogram.iloc[-1] > 0 and
                macd_histogram.iloc[-1] > macd_histogram.iloc[-3] and
                current_volume_ratio > 1.1  # Volume confirmation
            )
            
            strong_downtrend = (
                current_price < ema_9 < ema_21 < ema_55 and
                momentum_5 < -self.trend_threshold and
                momentum_10 < -self.trend_threshold * 0.7 and
                current_rsi < 45 and current_rsi > 25 and
                macd_histogram.iloc[-1] < 0 and
                macd_histogram.iloc[-1] < macd_histogram.iloc[-3] and
                current_volume_ratio > 1.1
            )
            
            # Check momentum acceleration
            momentum_accel = self.detect_momentum_acceleration()
            
            # Enhance trend detection with acceleration
            if momentum_accel['is_accelerating_up'] and not strong_uptrend:
                # Early trend detection
                if momentum_5 > self.trend_threshold * 0.5 and current_price > ema_21:
                    strong_uptrend = True
                    
            # Calculate trend strength
            if strong_uptrend:
                trend_strength = min(1.0, (momentum_5 + momentum_10) / (2 * self.trend_threshold * 2))
            elif strong_downtrend:
                trend_strength = min(1.0, abs(momentum_5 + momentum_10) / (2 * self.trend_threshold * 2))
            else:
                trend_strength = 0
                
            return {
                'is_strong_uptrend': strong_uptrend,
                'is_strong_downtrend': strong_downtrend,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'momentum_20': momentum_20,
                'rsi': current_rsi,
                'trend_strength': trend_strength,
                'current_price': current_price,
                'volume_ratio': current_volume_ratio,
                'is_accelerating': momentum_accel['is_accelerating_up'] or momentum_accel['is_accelerating_down']
            }
            
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            return {
                'is_strong_uptrend': False,
                'is_strong_downtrend': False,
                'momentum_5': 0,
                'momentum_10': 0,
                'momentum_20': 0,
                'rsi': 50,
                'trend_strength': 0,
                'current_price': self.get_current_price(),
                'volume_ratio': 1,
                'is_accelerating': False
            }
            
    def enter_trend_position(self, trend_direction: str, trend_info: Dict):
        """
        Enter trend position with improved sizing and timing
        """
        try:
            current_price = self.get_current_price()
            
            # Dynamic position sizing based on trend strength
            base_ratio = self.trend_buy_ratio
            if trend_info['trend_strength'] > 0.7:
                position_ratio = base_ratio * 1.2  # Increase size for strong trends
            elif trend_info['is_accelerating']:
                position_ratio = base_ratio * 1.1  # Slight increase for accelerating trends
            else:
                position_ratio = base_ratio
                
            # Cap maximum allocation
            position_ratio = min(position_ratio, 0.7)  # Max 70% for trend
            
            trend_investment = self.total_investment * position_ratio
            position_size = self.calculate_position_size(current_price, trend_investment)
            
            if trend_direction == 'up':
                # Use limit order slightly above market for better fill
                entry_price = current_price * 1.0001
                order = self.place_smart_order('buy', entry_price, position_size)
                
                if order:
                    self.trend_position = {
                        'direction': 'long',
                        'entry_price': entry_price,
                        'size': position_size,
                        'order_id': order['id'],
                        'initial_stop': current_price * (1 - self.trailing_stop_percent * 2),  # Wider initial stop
                        'trend_strength': trend_info['trend_strength']
                    }
                    
                    self.trend_entry_price = entry_price
                    self.highest_price_since_trend = current_price
                    
                    logger.info(f"🚀 TREND LONG entered: {position_size:.4f} @ ${entry_price:.2f}")
                    logger.info(f"  - Trend strength: {trend_info['trend_strength']*100:.0f}%")
                    logger.info(f"  - Investment: ${trend_investment:.2f} ({position_ratio*100:.0f}% of total)")
                    logger.info(f"  - Momentum 5m: {trend_info['momentum_5']*100:.2f}%")
                    
        except Exception as e:
            logger.error(f"Error entering trend position: {e}")
            
    def manage_trend_position(self):
        """
        Enhanced trend position management with dynamic trailing stops
        """
        if not self.trend_position:
            return
            
        try:
            current_price = self.get_current_price()
            
            if self.trend_position['direction'] == 'long':
                # Update highest price
                if current_price > self.highest_price_since_trend:
                    self.highest_price_since_trend = current_price
                    
                    # Tighten stop as profit increases
                    profit_percent = (current_price - self.trend_entry_price) / self.trend_entry_price
                    
                    if profit_percent > 0.03:  # 3% profit
                        trailing_percent = 0.01  # Tight 1% trailing stop
                    elif profit_percent > 0.02:  # 2% profit
                        trailing_percent = 0.012  # 1.2% trailing stop
                    elif profit_percent > 0.01:  # 1% profit
                        trailing_percent = 0.015  # 1.5% trailing stop
                    else:
                        trailing_percent = self.trailing_stop_percent
                        
                    logger.info(f"📈 New trend high: ${current_price:.2f} (+{profit_percent*100:.2f}%)")
                else:
                    trailing_percent = self.trailing_stop_percent
                    
                # Calculate dynamic trailing stop
                trailing_stop_price = max(
                    self.highest_price_since_trend * (1 - trailing_percent),
                    self.trend_position['initial_stop']  # Never go below initial stop
                )
                
                # Check trend momentum
                trend_info = self.detect_strong_trend()
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                if current_price <= trailing_stop_price:
                    should_exit = True
                    exit_reason = "Trailing stop hit"
                elif trend_info['momentum_5'] < -0.005:  # Momentum reversal
                    should_exit = True
                    exit_reason = "Momentum reversal"
                elif trend_info['rsi'] > 80:  # Overbought
                    should_exit = True
                    exit_reason = "RSI overbought"
                    
                if should_exit:
                    # Exit position
                    order = self.exchange.create_market_sell_order(
                        symbol=self.symbol,
                        amount=self.trend_position['size'],
                        params={'reduceOnly': True}
                    )
                    
                    # Calculate profit
                    exit_price = current_price
                    profit_percent = ((exit_price - self.trend_entry_price) / self.trend_entry_price) * 100
                    profit_usdt = (exit_price - self.trend_entry_price) * self.trend_position['size'] * self.leverage
                    
                    self.trend_profits += profit_usdt
                    self.total_trades += 1
                    if profit_usdt > 0:
                        self.winning_trades += 1
                        
                    logger.info(f"📉 Trend exit @ ${exit_price:.2f} - {exit_reason}")
                    logger.info(f"  - Profit: {profit_percent:.2f}% (${profit_usdt:.2f})")
                    logger.info(f"  - Duration: {(time.time() - self.trend_position.get('entry_time', time.time()))/60:.1f} minutes")
                    
                    # Reset trend position
                    self.trend_position = None
                    self.trend_mode = 'neutral'
                    self.highest_price_since_trend = 0
                    
                    # Resume grid trading
                    self.resume_grid_trading()
                else:
                    # Log position status
                    distance_to_stop = ((current_price - trailing_stop_price) / current_price) * 100
                    profit_pct = ((current_price - self.trend_entry_price) / self.trend_entry_price) * 100
                    logger.info(f"Trend position: +{profit_pct:.2f}%, Stop: ${trailing_stop_price:.2f} (-{distance_to_stop:.2f}%)")
                    
        except Exception as e:
            logger.error(f"Error managing trend position: {e}")
            
    def auto_configure_grid(self, total_investment: float):
        """
        Enhanced auto-configuration with dynamic parameters
        """
        logger.info("🔧 Analyzing market for optimal grid configuration...")
        
        # Get dynamic parameters
        params = self.calculate_dynamic_grid_params()
        
        self.upper_price = params['upper_price']
        self.lower_price = params['lower_price']
        self.grid_levels = params['grid_levels']
        self.grid_spacing = params['grid_spacing']
        self.total_investment = total_investment
        
        # Adjust order amount based on market regime
        if params['market_regime'] == 'ranging':
            # More capital for grid in ranging markets
            grid_allocation = 0.7
        elif params['market_regime'] in ['bullish', 'bearish']:
            # Less for grid, more for trend in trending markets
            grid_allocation = 0.5
        else:
            grid_allocation = 0.6
            
        grid_investment = total_investment * grid_allocation
        self.order_amount = grid_investment / self.grid_levels
        
        # Set price decimals
        current_price = params.get('current_price', self.get_current_price())
        if current_price < 0.01:
            self.price_decimals = 6
        elif current_price < 0.1:
            self.price_decimals = 5
        elif current_price < 1:
            self.price_decimals = 4
        elif current_price < 10:
            self.price_decimals = 3
        else:
            self.price_decimals = 2
            
        logger.info(f"📊 Grid Configuration Complete:")
        logger.info(f"  - Market Regime: {params['market_regime']}")
        logger.info(f"  - Price Range: ${self.lower_price:.{self.price_decimals}f} - ${self.upper_price:.{self.price_decimals}f}")
        logger.info(f"  - Grid Levels: {self.grid_levels}")
        logger.info(f"  - Grid Spacing: ${self.grid_spacing:.{self.price_decimals}f} ({self.grid_spacing/current_price*100:.3f}%)")
        logger.info(f"  - Grid Investment: ${grid_investment:.2f} ({grid_allocation*100:.0f}%)")
        logger.info(f"  - Order Size: ${self.order_amount:.2f} per level")
        
        if params.get('support_levels'):
            logger.info(f"  - Key Support: {params['support_levels']}")
        if params.get('resistance_levels'):
            logger.info(f"  - Key Resistance: {params['resistance_levels']}")
            
    def place_grid_orders(self):
        """
        Place grid orders with smart order routing
        """
        try:
            current_price = self.get_current_price()
            grid_levels = self.calculate_grid_levels()
            
            logger.info(f"📍 Current price: ${current_price:.{self.price_decimals}f}")
            logger.info(f"📊 Placing {len(grid_levels)} grid orders...")
            
            placed_buys = 0
            placed_sells = 0
            
            for price in grid_levels:
                if price < current_price * 0.9995:  # Buy orders below current price
                    amount = self.calculate_position_size(price, self.order_amount)
                    order = self.place_smart_order('buy', price, amount)
                    
                    if order:
                        self.buy_orders[order['id']] = {
                            'price': price,
                            'amount': amount,
                            'status': 'open',
                            'placed_at': time.time()
                        }
                        placed_buys += 1
                        
                elif price > current_price * 1.0005:  # Sell orders above current price
                    amount = self.calculate_position_size(price, self.order_amount)
                    order = self.place_smart_order('sell', price, amount)
                    
                    if order:
                        self.sell_orders[order['id']] = {
                            'price': price,
                            'amount': amount,
                            'status': 'open',
                            'placed_at': time.time()
                        }
                        placed_sells += 1
                        
            logger.info(f"✅ Grid orders placed: {placed_buys} buys, {placed_sells} sells")
            
            # Place initial scalp orders
            self.place_scalp_orders()
            
        except Exception as e:
            logger.error(f"Error placing grid orders: {e}")
            raise
            
    def calculate_grid_levels(self) -> List[float]:
        """Calculate all grid price levels"""
        levels = []
        for i in range(self.grid_levels):
            price = self.lower_price + (i * self.grid_spacing)
            levels.append(round(price, self.price_decimals))
        return levels
        
    def check_filled_orders(self):
        """
        Enhanced order checking with profit tracking
        """
        try:
            if self.trend_mode != 'neutral':
                return
                
            # Fetch recent closed orders
            closed_orders = self.exchange.fetch_closed_orders(self.symbol, limit=50)
            
            for order in closed_orders:
                order_id = order['id']
                
                # Check buy orders
                if order_id in self.buy_orders and self.buy_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.buy_orders[order_id]['status'] = 'filled'
                        
                        # Place sell order one grid level up
                        new_sell_price = filled_price + self.grid_spacing
                        
                        # Add small randomization to avoid order clustering
                        new_sell_price *= (1 + np.random.uniform(-0.0001, 0.0001))
                        
                        if new_sell_price <= self.upper_price:
                            sell_amount = self.calculate_position_size(new_sell_price, self.order_amount)
                            sell_order = self.place_smart_order('sell', new_sell_price, sell_amount)
                            
                            if sell_order:
                                self.sell_orders[sell_order['id']] = {
                                    'price': new_sell_price,
                                    'amount': sell_amount,
                                    'status': 'open',
                                    'placed_at': time.time(),
                                    'pair_buy_price': filled_price  # Track paired buy price
                                }
                                
                        # Track profit
                        fee_cost = filled_price * filled_amount * self.maker_fee * 2  # Buy and sell fees
                        expected_profit = (self.grid_spacing * filled_amount * self.leverage) - fee_cost
                        
                        logger.info(f"✅ Buy filled @ ${filled_price:.{self.price_decimals}f}, sell placed @ ${new_sell_price:.{self.price_decimals}f}")
                        logger.info(f"   Expected profit: ${expected_profit:.2f}")
                        
                # Check sell orders
                elif order_id in self.sell_orders and self.sell_orders[order_id]['status'] == 'open':
                    if order['status'] == 'closed' and order['filled'] > 0:
                        filled_price = order['price']
                        filled_amount = order['filled']
                        self.sell_orders[order_id]['status'] = 'filled'
                        
                        # Calculate actual profit if this was a paired trade
                        if 'pair_buy_price' in self.sell_orders[order_id]:
                            buy_price = self.sell_orders[order_id]['pair_buy_price']
                            # Improved profit calculation with leverage
                            profit = (filled_price - buy_price) * filled_amount * self.leverage
                            
                            # More accurate fee calculation
                            fee_cost = (filled_price * filled_amount * self.maker_fee) + \
                                      (buy_price * filled_amount * self.maker_fee)
                            
                            # Calculate net profit
                            net_profit = profit - fee_cost
                            
                            # Add to total grid profits
                            self.grid_profits += net_profit
                            self.total_trades += 1
                            if net_profit > 0:
                                self.winning_trades += 1
                                
                            logger.info(f"💰 Grid profit realized: ${net_profit:.2f} (Price diff: ${filled_price - buy_price:.{self.price_decimals}f}, Leverage: {self.leverage}x)")
                            
                        # Place buy order one grid level down
                        new_buy_price = filled_price - self.grid_spacing
                        new_buy_price *= (1 + np.random.uniform(-0.0001, 0.0001))
                        
                        if new_buy_price >= self.lower_price:
                            buy_amount = self.calculate_position_size(new_buy_price, self.order_amount)
                            buy_order = self.place_smart_order('buy', new_buy_price, buy_amount)
                            
                            if buy_order:
                                self.buy_orders[buy_order['id']] = {
                                    'price': new_buy_price,
                                    'amount': buy_amount,
                                    'status': 'open',
                                    'placed_at': time.time()
                                }
                                
                        logger.info(f"✅ Sell filled @ ${filled_price:.{self.price_decimals}f}, buy placed @ ${new_buy_price:.{self.price_decimals}f}")
                        
            # Check scalp orders separately
            self.check_scalp_orders()
            
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
            
    def check_grid_boundaries(self):
        """
        Dynamic grid boundary adjustment
        """
        try:
            current_price = self.get_current_price()
            grid_center = (self.upper_price + self.lower_price) / 2
            distance_from_center = abs(current_price - grid_center) / grid_center
            
            # Check if price has moved significantly from grid center
            if distance_from_center > self.grid_shift_threshold:
                logger.warning(f"⚠️ Price ${current_price:.{self.price_decimals}f} moved {distance_from_center*100:.1f}% from grid center")
                
                # Get market analysis
                book_analysis = self.analyze_order_book()
                trend_info = self.detect_strong_trend()
                
                # Don't adjust if strong trend detected (let trend system handle it)
                if trend_info['is_strong_uptrend'] or trend_info['is_strong_downtrend']:
                    return
                    
                logger.info("🔄 Adjusting grid boundaries...")
                
                # Cancel current orders
                self.exchange.cancel_all_orders(self.symbol)
                self.buy_orders.clear()
                self.sell_orders.clear()
                
                # Reconfigure with new center
                self.auto_configure_grid(self.total_investment)
                
                # Place new orders
                self.place_grid_orders()
                
                logger.info("✅ Grid boundaries adjusted successfully")
                
        except Exception as e:
            logger.error(f"Error checking grid boundaries: {e}")
            
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        try:
            total_profits = self.grid_profits + self.trend_profits + self.scalp_profits
            
            # Calculate profit factor
            if self.total_trades > 0:
                win_rate = self.winning_trades / self.total_trades
                avg_win = total_profits / self.winning_trades if self.winning_trades > 0 else 0
                avg_loss = abs(total_profits / (self.total_trades - self.winning_trades)) if self.total_trades > self.winning_trades else 0
                
                if avg_loss > 0:
                    self.profit_factor = avg_win / avg_loss
                else:
                    self.profit_factor = float('inf') if avg_win > 0 else 0
            else:
                win_rate = 0
                self.profit_factor = 0
                
            # Get account info
            account_info = self.get_account_info()
            current_balance = account_info.get('usdt_balance', 0)
            
            # Calculate returns
            if self.initial_balance > 0:
                total_return = (current_balance - self.initial_balance) / self.initial_balance * 100
                
                # FIXED: Only calculate drawdown if we've actually had a higher balance than current
                # and if the difference is significant (more than 0.5%)
                if self.highest_balance > current_balance and \
                   ((self.highest_balance - current_balance) / self.highest_balance) > 0.005:
                    current_drawdown = (self.highest_balance - current_balance) / self.highest_balance * 100
                else:
                    current_drawdown = 0  # No significant drawdown
            else:
                total_return = 0
                current_drawdown = 0
                
            return {
                'total_profits': round(total_profits, 2),
                'grid_profits': round(self.grid_profits, 2),
                'trend_profits': round(self.trend_profits, 2),
                'scalp_profits': round(self.scalp_profits, 2),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': round(win_rate * 100, 1),
                'profit_factor': round(self.profit_factor, 2),
                'total_return': round(total_return, 2),
                'current_drawdown': round(current_drawdown, 2),
                'current_balance': round(current_balance, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    def get_account_info(self) -> Dict:
        """Get account balance and position info"""
        try:
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions([self.symbol])
            
            info = {
                'usdt_balance': balance['USDT']['free'] if 'USDT' in balance else 0,
                'positions': positions
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
            
    def pause_grid_trading(self):
        """Pause grid trading when entering trend mode"""
        try:
            logger.info("⏸️ Pausing grid trading for trend following...")
            self.exchange.cancel_all_orders(self.symbol)
            self.buy_orders.clear()
            self.sell_orders.clear()
            self.scalp_orders.clear()
            
        except Exception as e:
            logger.error(f"Error pausing grid trading: {e}")
            
    def resume_grid_trading(self):
        """Resume grid trading after trend mode"""
        try:
            logger.info("▶️ Resuming grid trading...")
            
            # Brief delay to let market settle
            time.sleep(2)
            
            # Recalculate grid based on new price
            self.auto_configure_grid(self.total_investment)
            
            # Place new grid orders
            self.place_grid_orders()
            
            logger.info("✅ Grid trading resumed successfully")
            
        except Exception as e:
            logger.error(f"Error resuming grid trading: {e}")
            
    def check_and_manage_trend(self):
        """
        Enhanced trend detection and management
        """
        trend_info = self.detect_strong_trend()
        
        # If already in trend mode, manage the position
        if self.trend_mode != 'neutral' and self.trend_position:
            self.manage_trend_position()
            return
            
        # Check if we should enter trend mode
        if trend_info['is_strong_uptrend'] and self.trend_mode == 'neutral':
            logger.info("🚀 STRONG UPTREND DETECTED!")
            logger.info(f"  - Momentum 5m: {trend_info['momentum_5']*100:.2f}%")
            logger.info(f"  - Momentum 10m: {trend_info['momentum_10']*100:.2f}%")
            logger.info(f"  - RSI: {trend_info['rsi']:.2f}")
            logger.info(f"  - Volume ratio: {trend_info['volume_ratio']:.2f}x")
            logger.info(f"  - Trend strength: {trend_info['trend_strength']*100:.0f}%")
            
            # Check risk before entering
            if self.manage_risk():
                # Pause grid trading
                self.pause_grid_trading()
                
                # Enter trend mode
                self.trend_mode = 'strong_up'
                self.enter_trend_position('up', trend_info)
                
    def set_leverage(self):
        """Set leverage for the trading pair"""
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"✅ Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            error_message = str(e).lower()
            if "leverage not modified" in error_message or "110043" in error_message:
                logger.info(f"Leverage already set to {self.leverage}x")
            else:
                logger.error(f"Error setting leverage: {e}")
                raise
                
    def run(self, check_interval: int = 10, recalibration_interval: int = 3600):
        """
        Run the enhanced grid trading bot with multiple profit strategies
        
        Args:
            check_interval: Seconds between checking for filled orders
            recalibration_interval: Seconds between grid recalibration (default: 1 hour)
        """
        try:
            # Set leverage
            self.set_leverage()
            
            # Initial risk check
            if not self.manage_risk():
                logger.error("Risk check failed. Bot will not start.")
                return
                
            # Get initial balance
            account_info = self.get_account_info()
            self.initial_balance = account_info.get('usdt_balance', 0)
            self.highest_balance = self.initial_balance
            
            logger.info(f"💰 Starting balance: ${self.initial_balance:.2f}")
            
            # Place initial grid orders
            self.place_grid_orders()
            
            logger.info(f"🤖 Enhanced Grid Bot Started")
            logger.info(f"  - Check interval: {check_interval}s")
            logger.info(f"  - Recalibration: every {recalibration_interval/3600:.1f} hours")
            logger.info(f"  - Grid allocation: {(1-self.trend_buy_ratio)*100:.0f}%")
            logger.info(f"  - Trend allocation: {self.trend_buy_ratio*100:.0f}%")
            logger.info(f"  - Scalping: {'Enabled' if self.scalp_enabled else 'Disabled'}")
            logger.info(f"  - Risk limits: {self.max_drawdown*100:.0f}% max drawdown, {self.daily_profit_target*100:.1f}% daily target")
            
            last_recalibration = time.time()
            last_trend_check = time.time()
            last_performance_log = time.time()
            last_optimization = time.time()
            
            trend_check_interval = 30  # Check trends every 30 seconds
            performance_log_interval = 300  # Log performance every 5 minutes
            optimization_interval = 1800  # Optimize every 30 minutes
            
            while True:
                try:
                    # Risk management check
                    if not self.manage_risk():
                        logger.error("Risk limit reached. Stopping bot.")
                        break
                        
                    # Check for strong trends
                    if time.time() - last_trend_check > trend_check_interval:
                        self.check_and_manage_trend()
                        last_trend_check = time.time()
                        
                    # Normal grid operations (only if not in trend mode)
                    if self.trend_mode == 'neutral':
                        # Check filled orders
                        self.check_filled_orders()
                        
                        # Check grid boundaries
                        self.check_grid_boundaries()
                        
                        # Refresh scalp orders periodically
                        if int(time.time()) % 120 == 0:  # Every 2 minutes
                            self.place_scalp_orders()
                            
                    # Performance optimization
                    if time.time() - last_optimization > optimization_interval:
                        self.optimize_grid_performance()
                        last_optimization = time.time()
                        
                    # Log performance metrics
                    if time.time() - last_performance_log > performance_log_interval:
                        metrics = self.calculate_performance_metrics()
                        
                        logger.info(f"📊 PERFORMANCE UPDATE:")
                        logger.info(f"  💰 Total P&L: ${metrics.get('total_profits', 0):.2f}")
                        logger.info(f"  📈 Return: {metrics.get('total_return', 0):.2f}%")
                        logger.info(f"  🎯 Win Rate: {metrics.get('win_rate', 0):.1f}%")
                        logger.info(f"  📊 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                        logger.info(f"  💳 Balance: ${metrics.get('current_balance', 0):.2f}")
                        
                        if metrics.get('current_drawdown', 0) > 0:
                            logger.warning(f"  ⚠️ Drawdown: {metrics.get('current_drawdown', 0):.1f}%")
                            
                        # Breakdown by strategy
                        logger.info(f"  Strategy Breakdown:")
                        logger.info(f"    - Grid: ${metrics.get('grid_profits', 0):.2f}")
                        logger.info(f"    - Trend: ${metrics.get('trend_profits', 0):.2f}")
                        logger.info(f"    - Scalp: ${metrics.get('scalp_profits', 0):.2f}")
                        
                        # Current mode
                        if self.trend_mode != 'neutral':
                            logger.info(f"  ⚡ MODE: TREND FOLLOWING")
                        else:
                            open_buys = sum(1 for o in self.buy_orders.values() if o['status'] == 'open')
                            open_sells = sum(1 for o in self.sell_orders.values() if o['status'] == 'open')
                            logger.info(f"  📊 MODE: GRID TRADING ({open_buys} buys, {open_sells} sells)")
                            
                        last_performance_log = time.time()
                        
                    # Periodic grid recalibration (only in neutral mode)
                    if self.trend_mode == 'neutral' and time.time() - last_recalibration > recalibration_interval:
                        logger.info("🔄 Performing periodic grid recalibration...")
                        
                        # Cancel current orders
                        self.exchange.cancel_all_orders(self.symbol)
                        self.buy_orders.clear()
                        self.sell_orders.clear()
                        
                        # Reconfigure grid
                        self.auto_configure_grid(self.total_investment)
                        
                        # Place new orders
                        self.place_grid_orders()
                        
                        last_recalibration = time.time()
                        logger.info("✅ Grid recalibration complete")
                        
                    # Sleep before next iteration
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(check_interval * 2)  # Longer sleep on error
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            # Final cleanup
            self.stop()
            
            # Final performance report
            logger.info("=" * 50)
            logger.info("FINAL PERFORMANCE REPORT")
            logger.info("=" * 50)
            
            metrics = self.calculate_performance_metrics()
            
            logger.info(f"Total Profits: ${metrics.get('total_profits', 0):.2f}")
            logger.info(f"Total Return: {metrics.get('total_return', 0):.2f}%")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            logger.info(f"Final Balance: ${metrics.get('current_balance', 0):.2f}")
            
            logger.info("\nStrategy Performance:")
            logger.info(f"  Grid Trading: ${metrics.get('grid_profits', 0):.2f}")
            logger.info(f"  Trend Following: ${metrics.get('trend_profits', 0):.2f}")
            logger.info(f"  Scalping: ${metrics.get('scalp_profits', 0):.2f}")
            
            logger.info("=" * 50)
            
    def stop(self):
        """Cancel all open orders and close positions safely"""
        try:
            logger.info("🛑 Stopping bot...")
            
            # Cancel all orders
            self.exchange.cancel_all_orders(self.symbol)
            logger.info("All orders cancelled")
            
            # Close any open positions
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['contracts'] != 0:
                    logger.info(f"Closing position: {pos['contracts']} contracts")
                    if pos['side'] == 'long':
                        self.exchange.create_market_sell_order(
                            self.symbol,
                            abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                    else:
                        self.exchange.create_market_buy_order(
                            self.symbol,
                            abs(pos['contracts']),
                            params={'reduceOnly': True}
                        )
                        
            logger.info("✅ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "VDpt0WQXIjXul4OBrS"
    API_SECRET = "z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9"
    
    # Investment amount
    TOTAL_INVESTMENT = 41  # Total USDT to use
    
    # Initialize enhanced bot
    bot = BybitGridBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='1000PEPE/USDT:USDT',  # Can use any perpetual pair
        testnet=False  # Use testnet for testing
    )
    
    # Configure bot parameters (optional - will use optimized defaults)
    bot.scalp_enabled = True  # Enable scalping
    bot.trend_buy_ratio = 0.5  # 50% for trends
    bot.max_drawdown = 0.8  # 10% max drawdown
    bot.daily_profit_target = 0.1 # 2% daily target
    
    # Auto-configure grid based on market conditions
    bot.auto_configure_grid(total_investment=TOTAL_INVESTMENT)
    
    try:
        # Run the enhanced bot
        bot.run(
            check_interval=10,  # Check every 10 seconds
            recalibration_interval=3600  # Recalibrate hourly
        )
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot terminated by user")