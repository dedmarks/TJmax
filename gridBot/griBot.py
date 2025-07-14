import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
# New imports for enhancements
from arch import arch_model
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_trading_futures.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FuturesGridTrader:
    def __init__(self, symbol: str, api_key: str, secret: str, capital: float, 
                 leverage: int = 7, grid_levels: int = 10, min_grid_spacing: float = 0.005,
                 max_position_ratio: float = 0.8, use_testnet: bool = True,
                 z_score_window: int = 20, z_score_threshold: float = 2.0,
                 var_confidence: float = 0.95, var_days: int = 1):
        """
        Initialize Futures Grid Trading Bot with Leverage
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            api_key: Bybit API key
            secret: Bybit API secret
            capital: Total capital for grid trading
            leverage: Leverage multiplier (1-100)
            grid_levels: Number of grid levels each side
            min_grid_spacing: Minimum grid spacing as percentage
            max_position_ratio: Maximum position size ratio
            use_testnet: Use testnet for testing
            z_score_window: Window size for Z-score calculation
            z_score_threshold: Z-score threshold for mean reversion signals
        """
        self.bybit = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': use_testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap'  # Use perpetual futures
            }
        })
        
        self.symbol = symbol
        self.capital = capital
        self.leverage = leverage
        self.grid_levels = grid_levels
        self.min_grid_spacing = min_grid_spacing
        self.max_position_ratio = max_position_ratio
        
        # Trading state
        self.active_orders: Dict[str, Dict] = {}
        self.filled_orders: List[Dict] = []
        self.base_price: Optional[float] = None
        self.grid_spacing: Optional[float] = None
        self.total_profit: float = 0.0
        self.is_running: bool = False
        self.current_position: float = 0.0  # Track net position
        self.position_side: str = 'both'  # 'both', 'long', 'short'
        self.price_history: List[float] = []  # Add this line to store price history
        
        # Performance tracking
        self.trades_count: int = 0
        self.start_time: Optional[datetime] = None
        self.funding_fees: float = 0.0
        
        # VaR parameters
        self.var_confidence = var_confidence
        self.var_days = var_days
        self.returns_history = []
        
        # Risk management
        self.max_position_size: float = 0.0
        self.liquidation_price: Optional[float] = None
        
        logger.info(f"Futures Grid Trader initialized for {symbol}")
        logger.info(f"Capital: ${capital}, Leverage: {leverage}x, Grid Levels: {grid_levels}")
    
    def setup_leverage_and_margin(self) -> None:
        """Setup leverage and margin mode"""
        try:
            # Set leverage
            try:
                self.bybit.set_leverage(self.leverage, self.symbol)
                logger.info(f"Leverage set to {self.leverage}x")
            except Exception as e:
                error_str = str(e)
                if "leverage not modified" in error_str:
                    logger.warning(f"Leverage already set or cannot be modified: {e}")
                    # Continue execution instead of raising the error
                else:
                    raise  # Re-raise if it's a different error
            
            # Set margin mode to cross margin
            self.bybit.set_margin_mode('cross', self.symbol)
            logger.info("Margin mode set to cross")
            
            # Set position mode to hedge mode (allows both long and short)
            try:
                self.bybit.set_position_mode(True, self.symbol)  # True = hedge mode
                logger.info("Position mode set to hedge")
            except Exception as e:
                logger.warning(f"Could not set hedge mode: {e}")
                
        except Exception as e:
            logger.error(f"Error setting up leverage/margin: {e}")
            raise
    
    def get_market_data(self) -> Dict:
        """Get current market data"""
        try:
            ticker = self.bybit.fetch_ticker(self.symbol)
            funding_rate = self.bybit.fetch_funding_rate(self.symbol)
            
            # In the get_market_data method, add this code at the end before the return statement
            
            market_data = {
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'funding_rate': funding_rate['fundingRate'],
                'funding_time': funding_rate['fundingDatetime']
            }
            
            # Store price in history
            self.price_history.append(market_data['price'])
            
            # Keep history at a reasonable size
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
                
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise
    
    def forecast_volatility_garch(self, days: int = 7) -> float:
        """Forecast volatility using GARCH model"""
        try:
            # Get historical data
            ohlcv = self.bybit.fetch_ohlcv(self.symbol, '1h', limit=days*24)
            prices = np.array([candle[4] for candle in ohlcv])
            returns = 100 * np.diff(np.log(prices))
            
            # Skip if not enough data
            if len(returns) < 30:
                return self.calculate_volatility(days)
            
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Forecast volatility
            forecast = model_fit.forecast(horizon=24)
            forecasted_var = forecast.variance.iloc[-1].values[0]
            forecasted_vol = np.sqrt(forecasted_var)
            
            # Convert to daily volatility
            daily_vol = forecasted_vol * np.sqrt(24) / 100
            
            logger.info(f"GARCH forecasted volatility: {daily_vol:.4f}")
            return daily_vol
            
        except Exception as e:
            logger.error(f"Error forecasting volatility with GARCH: {e}")
            # Fallback to traditional volatility calculation
            return self.calculate_volatility(days)
    
    def calculate_volatility(self, days: int = 7) -> float:
        """Calculate daily volatility from historical data"""
        try:
            ohlcv = self.bybit.fetch_ohlcv(self.symbol, '1h', limit=days*24)
            prices = [candle[4] for candle in ohlcv]
            returns = np.diff(np.log(prices))
            daily_volatility = np.std(returns) * np.sqrt(24)
            logger.info(f"Calculated volatility: {daily_volatility:.4f}")
            return daily_volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02  # Default 2% volatility
    
    def get_account_balance(self) -> Dict:
        """Get account balance and position info"""
        try:
            balance = self.bybit.fetch_balance()
            positions = self.bybit.fetch_positions([self.symbol])
            
            usdt_balance = balance['USDT']['free']
            position_info = positions[0] if positions else {}
            
            self.current_position = position_info.get('contracts', 0)
            self.liquidation_price = position_info.get('liquidationPrice')
            
            logger.info(f"USDT Balance: ${usdt_balance:.2f}")
            logger.info(f"Current Position: {self.current_position}")
            
            return {
                'balance': usdt_balance,
                'position_size': self.current_position,
                'liquidation_price': self.liquidation_price,
                'unrealized_pnl': position_info.get('unrealizedPnl', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'balance': 0.0, 'position_size': 0, 'liquidation_price': None}
    
    def calculate_position_size(self, price: float, side: str) -> float:
        """Calculate position size based on capital and leverage"""
        try:
            # Calculate notional value per grid level
            capital_per_level = self.capital / (self.grid_levels * 2)
            
            # Apply leverage
            leveraged_capital = capital_per_level * self.leverage
            
            # Calculate position size in contracts
            position_size = leveraged_capital / price
            
            # Apply maximum position ratio
            max_size = (self.capital * self.leverage * self.max_position_ratio) / price
            position_size = min(position_size, max_size / (self.grid_levels * 2))
            
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def setup_grid(self) -> List[Dict]:
        """Setup grid parameters and create order list"""
        market_data = self.get_market_data()
        
        # Update price history for Z-score calculation
        self.price_history.append(market_data['price'])
        if len(self.price_history) > 100:  # Limit history size
            self.price_history = self.price_history[-100:]
        
        # Use GARCH for volatility forecasting
        volatility = self.forecast_volatility_garch()
        
        self.base_price = market_data['price']
        
        # Dynamic grid spacing based on GARCH volatility forecast
        self.grid_spacing = max(
            volatility * 0.3,  # 30% of forecasted volatility
            self.min_grid_spacing  # Minimum spacing
        )
        
        logger.info(f"Base Price: ${self.base_price:.2f}")
        logger.info(f"Grid Spacing: {self.grid_spacing:.4f} ({self.grid_spacing*100:.2f}%)")
        logger.info(f"Funding Rate: {market_data['funding_rate']:.6f}")
        
        # Create grid orders
        orders = self.create_grid_orders()
        
        # Adjust orders based on Z-score mean reversion
        adjusted_orders = self.adjust_grid_based_on_z_score(orders)
        
        return adjusted_orders
    
    def create_grid_orders(self) -> List[Dict]:
        """Create grid buy and sell orders for futures"""
        orders = []
        
        for i in range(1, self.grid_levels + 1):
            # Buy orders below current price (long positions)
            buy_price = self.base_price * (1 - self.grid_spacing * i)
            buy_size = self.calculate_position_size(buy_price, 'buy')
            
            # Sell orders above current price (short positions)
            sell_price = self.base_price * (1 + self.grid_spacing * i)
            sell_size = self.calculate_position_size(sell_price, 'sell')
            
            orders.append({
                'side': 'buy',
                'price': buy_price,
                'size': buy_size,
                'level': i,
                'type': 'grid'
            })
            
            orders.append({
                'side': 'sell',
                'price': sell_price,
                'size': sell_size,
                'level': i,
                'type': 'grid'
            })
        
        return orders
    
    def place_grid_orders(self) -> None:
        """Place all grid orders"""
        orders = self.setup_grid()
        placed_orders = 0
        
        for order in orders:
            try:
                # Add position index parameter based on side
                position_idx = 1 if order['side'] == 'buy' else 2  # 1 for long, 2 for short in hedge mode
                # For one-way mode, use position_idx = 0
                
                result = self.bybit.create_limit_order(
                    symbol=self.symbol,
                    side=order['side'],
                    amount=order['size'],
                    price=order['price'],
                    params={
                        'timeInForce': 'GTC',  # Good Till Cancelled
                        'postOnly': True,  # Maker only orders
                        'positionIdx': position_idx  # Add position index
                    }
                )
                
                self.active_orders[result['id']] = {
                    'order_id': result['id'],
                    'side': order['side'],
                    'price': order['price'],
                    'size': order['size'],
                    'level': order['level'],
                    'type': order['type'],
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Placed {order['side']} order: {order['size']:.4f} @ ${order['price']:.2f}")
                placed_orders += 1
                time.sleep(0.1)  # Rate limit protection
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
        
        logger.info(f"Successfully placed {placed_orders} orders")
    
    def check_filled_orders(self) -> None:
        """Check for filled orders and place replacements"""
        filled_order_ids = []
        
        try:
            # Get all open orders in one call
            open_orders = self.bybit.fetch_open_orders(self.symbol)
            open_order_ids = {order['id'] for order in open_orders}
            
            # Check which orders from our tracking are no longer open
            for order_id, order_info in self.active_orders.items():
                if order_id not in open_order_ids:
                    try:
                        # Try to get the closed order details
                        order_status = self.bybit.fetch_closed_orders(self.symbol, limit=1, params={'orderId': order_id})
                        
                        if order_status and len(order_status) > 0 and order_status[0]['status'] == 'closed':
                            filled_order_ids.append(order_id)
                            self.filled_orders.append(order_info)
                            self.trades_count += 1
                            
                            # Update position tracking
                            if order_info['side'] == 'buy':
                                self.current_position += order_info['size']
                            else:
                                self.current_position -= order_info['size']
                            
                            # Calculate profit (simplified)
                            profit = self.calculate_trade_profit(order_info)
                            self.total_profit += profit
                            
                            logger.info(f"Order filled: {order_info['side']} {order_info['size']:.4f} @ ${order_info['price']:.2f} | Profit: ${profit:.2f}")
                            
                            # Place replacement order on opposite side
                            self.place_replacement_order(order_info)
                    except Exception as e:
                        logger.error(f"Error checking closed order {order_id}: {e}")
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
        
        # Remove filled orders from active orders
        for order_id in filled_order_ids:
            del self.active_orders[order_id]
    
    def calculate_trade_profit(self, order_info: Dict) -> float:
        """Calculate profit from a single trade"""
        # For futures, profit depends on price movement and leverage
        notional_value = order_info['size'] * order_info['price']
        profit = notional_value * self.grid_spacing * self.leverage
        
        # Account for trading fees (approximately 0.1% for maker orders)
        trading_fee = notional_value * 0.001
        return profit - trading_fee
    
    def place_replacement_order(self, filled_order: Dict) -> None:
        """Place replacement order after a fill"""
        try:
            # Calculate replacement order parameters
            if filled_order['side'] == 'buy':
                # If buy order filled, place sell order above
                new_price = filled_order['price'] * (1 + self.grid_spacing)
                new_side = 'sell'
                position_idx = 2  # Short position
            else:
                # If sell order filled, place buy order below
                new_price = filled_order['price'] * (1 - self.grid_spacing)
                new_side = 'buy'
                position_idx = 1  # Long position
            
            # Calculate new position size
            new_size = self.calculate_position_size(new_price, new_side)
            
            # Determine if this order should be reduceOnly
            # In place_replacement_order method
            # Get latest position before deciding on reduceOnly
            account_info = self.get_account_balance()
            current_position = account_info['position_size']
            
            # Only set reduceOnly if we actually have a position
            reduce_only = False
            if abs(current_position) > 0.00001:  # Non-zero position
                reduce_only = (current_position > 0 and new_side == 'sell') or \
                             (current_position < 0 and new_side == 'buy')
            
            # Place new order
            result = self.bybit.create_limit_order(
                symbol=self.symbol,
                side=new_side,
                amount=new_size,
                price=new_price,
                params={
                    'timeInForce': 'GTC',
                    'postOnly': True,
                    'positionIdx': position_idx,  # Add position index
                    'reduceOnly': reduce_only  # Add reduceOnly parameter when appropriate
                }
            )
            
            self.active_orders[result['id']] = {
                'order_id': result['id'],
                'side': new_side,
                'price': new_price,
                'size': new_size,
                'level': filled_order['level'],
                'type': 'replacement',
                'reduce_only': reduce_only,  # Track if this is a reduceOnly order
                'timestamp': datetime.now()
            }
            
            logger.info(f"Replacement order: {new_side} {new_size:.4f} @ ${new_price:.2f} {'(reduceOnly)' if reduce_only else ''}")
            
        except Exception as e:
            logger.error(f"Error placing replacement order: {e}")
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL from current position"""
        try:
            if self.current_position == 0:
                return 0.0
            
            current_price = self.get_market_data()['price']
            
            # Calculate PnL based on position and price movement
            if self.current_position > 0:  # Long position
                avg_entry_price = sum(order['price'] for order in self.filled_orders 
                                    if order['side'] == 'buy') / len([o for o in self.filled_orders if o['side'] == 'buy'])
                pnl = self.current_position * (current_price - avg_entry_price)
            else:  # Short position
                avg_entry_price = sum(order['price'] for order in self.filled_orders 
                                    if order['side'] == 'sell') / len([o for o in self.filled_orders if o['side'] == 'sell'])
                pnl = abs(self.current_position) * (avg_entry_price - current_price)
            
            return pnl * self.leverage
        except Exception as e:
            logger.error(f"Error calculating unrealized PnL: {e}")
            return 0.0
    
    def get_funding_fees(self) -> float:
        """Get funding fees paid/received"""
        try:
            # This would require fetching funding history from the exchange
            # For now, return accumulated funding fees
            return self.funding_fees
        except Exception as e:
            logger.error(f"Error getting funding fees: {e}")
            return 0.0
    
    def check_liquidation_risk(self) -> bool:
        """Check if position is at risk of liquidation"""
        try:
            account_info = self.get_account_balance()
            current_price = self.get_market_data()['price']
            
            if account_info['liquidation_price']:
                distance_to_liquidation = abs(current_price - account_info['liquidation_price']) / current_price
                
                if distance_to_liquidation < 0.1:  # Less than 10% from liquidation
                    logger.warning(f"LIQUIDATION RISK: Current price ${current_price:.2f}, Liquidation price ${account_info['liquidation_price']:.2f}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking liquidation risk: {e}")
            return False
    
    def calculate_var(self) -> Dict:
        """Calculate Value at Risk (VaR) using historical method"""
        try:
            # Need sufficient price history
            if len(self.price_history) < 30:
                return {'var': 0.0, 'var_pct': 0.0, 'cvar': 0.0}
                
            # Calculate returns
            prices = np.array(self.price_history)
            returns = np.diff(np.log(prices)) * 100  # Percentage returns
            
            # Calculate VaR
            var_percentile = 100 - (self.var_confidence * 100)
            var_return = np.percentile(returns, var_percentile)  # Negative return at confidence level
            
            # Calculate CVaR (Conditional VaR or Expected Shortfall)
            # CVaR is the expected loss exceeding VaR
            cvar_returns = returns[returns <= var_return]
            cvar_return = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_return
            
            # Convert to dollar amounts
            current_value = self.capital * self.leverage
            var_amount = current_value * abs(var_return) / 100 * np.sqrt(self.var_days)
            cvar_amount = current_value * abs(cvar_return) / 100 * np.sqrt(self.var_days)
            var_pct = (var_amount / self.capital) * 100
            
            return {
                'var': var_amount,
                'var_pct': var_pct,
                'cvar': cvar_amount
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'var': 0.0, 'var_pct': 0.0, 'cvar': 0.0}
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are exceeded"""
        try:
            # Only check risk limits if we have an active position
            if abs(self.current_position) < 0.00001:  # Effectively zero position
                return False
                
            # Calculate VaR
            var_metrics = self.calculate_var()
            
            # Check if VaR exceeds threshold (e.g., 10% of capital)
            var_limit = 0.60  # 10% of capital as maximum acceptable VaR
            var_exceeded = var_metrics['var_pct'] > (var_limit * 100)
            
            if var_exceeded:
                logger.warning(f"VaR limit exceeded: {var_metrics['var_pct']:.2f}% > {var_limit*100:.2f}%")
                
            return var_exceeded
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            market_data = self.get_market_data()
            account_info = self.get_account_balance()
            current_price = market_data['price']
            
            price_change = (current_price - self.base_price) / self.base_price if self.base_price else 0
            unrealized_pnl = self.calculate_unrealized_pnl()
            funding_fees = self.get_funding_fees()
            
            # Calculate VaR
            var_metrics = self.calculate_var()
            
            return {
                'current_price': current_price,
                'base_price': self.base_price,
                'price_change_pct': price_change * 100,
                'active_orders': len(self.active_orders),
                'filled_orders': len(self.filled_orders),
                'total_trades': self.trades_count,
                'current_position': self.current_position,
                'realized_profit': self.total_profit,
                'unrealized_pnl': unrealized_pnl,
                'funding_fees': funding_fees,
                'total_pnl': self.total_profit + unrealized_pnl + funding_fees,
                'liquidation_price': account_info.get('liquidation_price'),
                'funding_rate': market_data['funding_rate'],
                'account_balance': account_info['balance'],
                'var': var_metrics['var'],
                'var_pct': var_metrics['var_pct'],
                'cvar': var_metrics['cvar']
            }
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def monitor_grid(self) -> bool:
        """Monitor grid and check if adjustment is needed"""
        status = self.get_portfolio_status()
        
        # Check liquidation risk
        liquidation_risk = self.check_liquidation_risk()
        
        # Check VaR risk limits
        var_risk_exceeded = self.check_risk_limits()
        
        logger.info("=" * 60)
        logger.info(f"Current Price: ${status.get('current_price', 0):.2f}")
        logger.info(f"Price Change: {status.get('price_change_pct', 0):.2f}%")
        logger.info(f"Active Orders: {status.get('active_orders', 0)}")
        logger.info(f"Current Position: {status.get('current_position', 0):.4f}")
        logger.info(f"Total Trades: {status.get('total_trades', 0)}")
        logger.info(f"Realized Profit: ${status.get('realized_profit', 0):.2f}")
        logger.info(f"Unrealized PnL: ${status.get('unrealized_pnl', 0):.2f}")
        logger.info(f"Funding Fees: ${status.get('funding_fees', 0):.2f}")
        logger.info(f"Total PnL: ${status.get('total_pnl', 0):.2f}")
        logger.info(f"Funding Rate: {status.get('funding_rate', 0):.6f}")
        logger.info(f"VaR ({self.var_confidence*100}%, {self.var_days}d): ${status.get('var', 0):.2f} ({status.get('var_pct', 0):.2f}%)")
        logger.info(f"CVaR (Expected Shortfall): ${status.get('cvar', 0):.2f}")
        
        # Fix for the TypeError - Check if liquidation_price is None
        liq_price = status.get('liquidation_price')
        if liq_price is not None:
            logger.info(f"Liquidation Price: ${liq_price:.2f}")
        else:
            logger.info("Liquidation Price: Not available")
            
        logger.info(f"Account Balance: ${status.get('account_balance', 0):.2f}")
        if liquidation_risk:
            logger.warning("LIQUIDATION RISK DETECTED!")
        if var_risk_exceeded:
            logger.warning("VAR RISK LIMIT EXCEEDED!")
        logger.info("=" * 60)
        
        # Check if grid needs adjustment
        if abs(status.get('price_change_pct', 0)) > 25:  # 25% price movement
            logger.warning("Price moved significantly, grid adjustment recommended")
            return True
        
        if liquidation_risk or var_risk_exceeded:
            logger.error("RISK LIMIT EXCEEDED: Action required!")
            return True
        
        return False
    
    def emergency_close_position(self) -> None:
        """Emergency close all positions"""
        try:
            logger.warning("EMERGENCY: Closing all positions")
            
            # Cancel all orders first
            self.cancel_all_orders()
            
            # Close position if any
            if abs(self.current_position) > 0:
                side = 'sell' if self.current_position > 0 else 'buy'
                self.bybit.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=abs(self.current_position)
                )
                logger.info(f"Emergency position closed: {side} {abs(self.current_position):.4f}")
                
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
    
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
    
    def save_state(self, filename: str = 'futures_grid_state.json') -> None:
        """Save current state to file"""
        state = {
            'base_price': self.base_price,
            'grid_spacing': self.grid_spacing,
            'total_profit': self.total_profit,
            'trades_count': self.trades_count,
            'current_position': self.current_position,
            'funding_fees': self.funding_fees,
            'filled_orders': self.filled_orders,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'leverage': self.leverage
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"State saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, filename: str = 'futures_grid_state.json') -> None:
        """Load state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.base_price = state.get('base_price')
            self.grid_spacing = state.get('grid_spacing')
            self.total_profit = state.get('total_profit', 0.0)
            self.trades_count = state.get('trades_count', 0)
            self.current_position = state.get('current_position', 0.0)
            self.funding_fees = state.get('funding_fees', 0.0)
            self.filled_orders = state.get('filled_orders', [])
            
            if state.get('start_time'):
                self.start_time = datetime.fromisoformat(state['start_time'])
            
            logger.info(f"State loaded from {filename}")
        except FileNotFoundError:
            logger.info("No previous state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def run_grid_bot(self) -> None:
        """Main trading loop"""
        logger.info("Starting Futures Grid Trading Bot with Leverage...")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Load previous state if exists
        self.load_state()
        
        try:
            # Setup leverage and margin
            self.setup_leverage_and_margin()
            
            # Check account balance
            account_info = self.get_account_balance()
            if account_info['balance'] < self.capital:
                logger.error(f"Insufficient balance: ${account_info['balance']:.2f} < ${self.capital:.2f}")
                return
            
            # Initial setup
            self.place_grid_orders()
            
            # New line in the __init__ method, with the other state variables
            self.price_history = []
            
            # Pre-load price history with historical data
            try:
                ohlcv = self.bybit.fetch_ohlcv(self.symbol, '1h', limit=100)
                self.price_history = [candle[4] for candle in ohlcv]  # Use closing prices
                logger.info(f"Pre-loaded {len(self.price_history)} historical price points")
            except Exception as e:
                logger.error(f"Error pre-loading price history: {e}")
            
            # Main trading loop
            while self.is_running:
                try:
                    # Check for filled orders
                    self.check_filled_orders()
                    
                    # Monitor performance
                    needs_adjustment = self.monitor_grid()
                    
                    if needs_adjustment:
                        # Check if it's liquidation risk
                        if self.check_liquidation_risk():
                            logger.error("LIQUIDATION RISK: Emergency shutdown!")
                            self.emergency_close_position()
                            break
                        else:
                            user_input = input("Grid adjustment recommended. Restart grid? (y/n): ")
                            if user_input.lower() == 'y':
                                logger.info("Restarting grid...")
                                self.cancel_all_orders()
                                time.sleep(5)
                                self.place_grid_orders()
                    
                    # Save state periodically
                    self.save_state()
                    
                    # Sleep before next iteration
                    time.sleep(30)  # Check every 30 seconds
                    
                except KeyboardInterrupt:
                    logger.info("Stopping bot...")
                    self.stop_bot()
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.cleanup()
    
    def stop_bot(self) -> None:
        """Stop the bot gracefully"""
        self.is_running = False
        logger.info("Bot stop signal received")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up...")
        self.cancel_all_orders()
        self.save_state()
        
        # Final status report
        status = self.get_portfolio_status()
        logger.info("Final Status:")
        logger.info(f"Total Trades: {status.get('total_trades', 0)}")
        logger.info(f"Final Position: {status.get('current_position', 0):.4f}")
        logger.info(f"Total Profit: ${status.get('total_pnl', 0):.2f}")
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger.info(f"Runtime: {runtime}")
    # Add these methods to the FuturesGridTrader class, before the main() function
# For example, place them after the update_returns_history method

    def calculate_z_score(self) -> float:
        """Calculate Z-score for mean reversion strategy"""
        try:
            # Add fallback if attribute is missing
            z_score_window = getattr(self, 'z_score_window', 20)  # Default to 20 if missing
            
            # Need at least z_score_window data points
            if len(self.price_history) < z_score_window:
                return 0.0
                
            # Use recent price history based on window size
            recent_prices = self.price_history[-z_score_window:]
            
            # Calculate mean and standard deviation
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            # Avoid division by zero
            if std_price == 0:
                return 0.0
                
            # Calculate Z-score (current price's deviation from mean in std units)
            current_price = self.price_history[-1]
            z_score = (current_price - mean_price) / std_price
            
            logger.info(f"Current Z-score: {z_score:.2f}")
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating Z-score: {e}")
            return 0.0
    
    def is_mean_reverting(self) -> bool:
        """Test if the price series shows mean reversion properties"""
        try:
            # Need sufficient price history
            if len(self.price_history) < 30:
                return False
                
            # Perform Augmented Dickey-Fuller test for stationarity
            # A stationary series is more likely to be mean-reverting
            prices = np.array(self.price_history)
            result = adfuller(prices)
            
            # p-value less than 0.05 suggests stationarity (reject unit root hypothesis)
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            if is_stationary:
                logger.info(f"Price series is stationary (p-value: {p_value:.4f}), likely mean-reverting")
            else:
                logger.info(f"Price series is not stationary (p-value: {p_value:.4f}), may not be mean-reverting")
                
            return is_stationary
            
        except Exception as e:
            logger.error(f"Error testing for mean reversion: {e}")
            return False
    
    def adjust_grid_based_on_z_score(self, orders: List[Dict]) -> List[Dict]:
        """Adjust grid orders based on Z-score mean reversion signals"""
        try:
            # Calculate current Z-score
            z_score = self.calculate_z_score()
            
            # If Z-score is near zero or we don't have enough data, no adjustment needed
            if abs(z_score) < 0.5 or len(self.price_history) < self.z_score_window:
                return orders
                
            # Check if price series is mean-reverting
            if not self.is_mean_reverting():
                logger.info("Price series not mean-reverting, using standard grid")
                return orders
                
            adjusted_orders = []
            
            # Positive Z-score means price is above mean (potentially overbought)
            # Negative Z-score means price is below mean (potentially oversold)
            for order in orders:
                # Deep copy the order to avoid modifying the original
                adjusted_order = order.copy()
                
                # For high positive Z-score (overbought)
                if z_score > self.z_score_threshold:
                    if order['side'] == 'sell':  # Favor sell orders when overbought
                        # Increase size for sell orders
                        adjusted_order['size'] = order['size'] * 1.2
                        logger.info(f"Increased sell order size due to high Z-score: {z_score:.2f}")
                    else:  # Reduce buy orders when overbought
                        adjusted_order['size'] = order['size'] * 0.8
                        logger.info(f"Decreased buy order size due to high Z-score: {z_score:.2f}")
                        
                # For low negative Z-score (oversold)
                elif z_score < -self.z_score_threshold:
                    if order['side'] == 'buy':  # Favor buy orders when oversold
                        # Increase size for buy orders
                        adjusted_order['size'] = order['size'] * 1.2
                        logger.info(f"Increased buy order size due to low Z-score: {z_score:.2f}")
                    else:  # Reduce sell orders when oversold
                        adjusted_order['size'] = order['size'] * 0.8
                        logger.info(f"Decreased sell order size due to low Z-score: {z_score:.2f}")
                        
                adjusted_orders.append(adjusted_order)
                
            return adjusted_orders
            
        except Exception as e:
            logger.error(f"Error adjusting grid based on Z-score: {e}")
            return orders  # Return original orders if adjustment fails

def main():
    """Main function to run the futures grid trading bot"""
    # Configuration
    CONFIG = {
        'symbol': 'DOGE/USDT:USDT',  # Futures symbol format
        'api_key': 'VDpt0WQXIjXul4OBrS',
        'secret': 'z1Rq4a7xfF8AmmAfCjqrJGECGsnjFQJLInH9',
        'capital': 33,  # $35 capital
        'leverage': 40,    # 50x leverage
        'grid_levels': 8,  # 8 levels each side
        'min_grid_spacing': 0.005,  # 0.5% minimum spacing for futures
        'use_testnet': False,  # Use testnet for testing
        'z_score_window': 20,  # Window for Z-score calculation
        'z_score_threshold': 2.0,  # Z-score threshold for mean reversion
        'var_confidence': 0.95,  # 95% confidence for VaR
        'var_days': 1  # 1-day VaR
    }
    
    # Create and run futures grid trader
    grid_trader = FuturesGridTrader(
        symbol=CONFIG['symbol'],
        api_key=CONFIG['api_key'],
        secret=CONFIG['secret'],
        capital=CONFIG['capital'],
        leverage=CONFIG['leverage'],
        grid_levels=CONFIG['grid_levels'],
        min_grid_spacing=CONFIG['min_grid_spacing'],
        use_testnet=CONFIG['use_testnet'],
        z_score_window=CONFIG['z_score_window'],
        z_score_threshold=CONFIG['z_score_threshold'],
        var_confidence=CONFIG['var_confidence'],
        var_days=CONFIG['var_days']
    )
    
    # Run the bot
    grid_trader.run_grid_bot()


if __name__ == "__main__":
    main()


