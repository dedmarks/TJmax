"""
Web Dashboard for Smart Grid Trading Bot
"""

import logging
import asyncio
from typing import Dict
from queue import Queue
import threading

# Web Dashboard imports
try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("WARNING: Flask dependencies not installed. Dashboard will be disabled.")
    print("Install with: pip install flask flask-cors flask-socketio plotly")

# Configure logging
logger = logging.getLogger(__name__)

# HTML Template for Dashboard
if DASHBOARD_AVAILABLE:
    dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Grid Trading Bot Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0;
            color: white;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #333;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #888;
            font-size: 14px;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #fff;
        }
        .positive {
            color: #4ade80;
        }
        .negative {
            color: #f87171;
        }
        .grid-table {
            background: #1a1a1a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th {
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            font-size: 12px;
        }
        .chart-container {
            background: #1a1a1a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .price-bar {
            height: 8px;
            background: #333;
            border-radius: 4px;
            position: relative;
            margin: 5px 0;
        }
        .price-position {
            position: absolute;
            height: 100%;
            background: #667eea;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .controls {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            font-weight: 600;
        }
        button:hover {
            background: #764ba2;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active {
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(74, 222, 128, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(74, 222, 128, 0);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Grid Trading Bot Dashboard</h1>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total P&L</h3>
                <div id="total-pnl" class="stat-value">$0.00</div>
            </div>
            <div class="stat-card">
                <h3>Active Grids</h3>
                <div id="active-grids" class="stat-value">0</div>
            </div>
            <div class="stat-card">
                <h3>Total Trades</h3>
                <div id="total-trades" class="stat-value">0</div>
            </div>
            <div class="stat-card">
                <h3>Available Balance</h3>
                <div id="available-balance" class="stat-value">$0.00</div>
            </div>
        </div>
        
        <div class="grid-table">
            <h2>Active Grid Strategies</h2>
            <table id="grid-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Range</th>
                        <th>Position</th>
                        <th>P&L</th>
                        <th>Trades</th>
                        <th>Runtime</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="grid-table-body">
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>
        </div>
        
        <div class="chart-container">
            <h2>Market Scores</h2>
            <div id="market-scores-chart" style="height: 400px;"></div>
        </div>
        
        <div class="controls">
            <h2>Controls</h2>
            <button id="refresh-btn">Refresh Data</button>
            <button id="scan-markets-btn">Scan Markets</button>
        </div>
    </div>
    
    <script>
        // Connect to SocketIO
        const socket = io();
        
        // Handle connection
        socket.on('connect', () => {
            console.log('Connected to server');
            // Request initial update
            socket.emit('request_update');
        });
        
        // Handle updates
        socket.on('update', (data) => {
            updateDashboard(data);
        });
        
        // Update dashboard with data
        function updateDashboard(data) {
            // Update summary stats
            document.getElementById('total-pnl').textContent = '$' + data.total_pnl.toFixed(2);
            document.getElementById('total-pnl').className = 'stat-value ' + 
                (data.total_pnl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('active-grids').textContent = data.active_grids.length;
            document.getElementById('total-trades').textContent = data.total_trades;
            document.getElementById('available-balance').textContent = '$' + data.available_balance.toFixed(2);
            
            // Update grid table
            const tableBody = document.getElementById('grid-table-body');
            tableBody.innerHTML = '';
            
            data.active_grids.forEach(grid => {
                const row = document.createElement('tr');
                
                // Symbol
                const symbolCell = document.createElement('td');
                symbolCell.textContent = grid.symbol;
                row.appendChild(symbolCell);
                
                // Current price
                const priceCell = document.createElement('td');
                priceCell.textContent = grid.current_price ? grid.current_price.toFixed(4) : 'N/A';
                row.appendChild(priceCell);
                
                // Range
                const rangeCell = document.createElement('td');
                rangeCell.textContent = `${grid.grid_range[0].toFixed(2)} - ${grid.grid_range[1].toFixed(2)}`;
                row.appendChild(rangeCell);
                
                // Position in range
                const positionCell = document.createElement('td');
                if (grid.price_position) {
                    const bar = document.createElement('div');
                    bar.className = 'price-bar';
                    
                    const position = document.createElement('div');
                    position.className = 'price-position';
                    position.style.width = `${grid.price_position}%`;
                    
                    bar.appendChild(position);
                    positionCell.appendChild(bar);
                } else {
                    positionCell.textContent = 'N/A';
                }
                row.appendChild(positionCell);
                
                // P&L
                const pnlCell = document.createElement('td');
                pnlCell.textContent = '$' + grid.net_pnl.toFixed(2);
                pnlCell.className = grid.net_pnl >= 0 ? 'positive' : 'negative';
                row.appendChild(pnlCell);
                
                // Trades
                const tradesCell = document.createElement('td');
                tradesCell.textContent = grid.total_trades;
                row.appendChild(tradesCell);
                
                // Runtime
                const runtimeCell = document.createElement('td');
                runtimeCell.textContent = grid.runtime_hours.toFixed(1) + ' hrs';
                row.appendChild(runtimeCell);
                
                // Actions
                const actionsCell = document.createElement('td');
                const stopButton = document.createElement('button');
                stopButton.textContent = 'Stop';
                stopButton.onclick = () => stopGrid(grid.symbol);
                actionsCell.appendChild(stopButton);
                row.appendChild(actionsCell);
                
                tableBody.appendChild(row);
            });
            
            // If no active grids, show message
            if (data.active_grids.length === 0) {
                const row = document.createElement('tr');
                const cell = document.createElement('td');
                cell.colSpan = 8;
                cell.textContent = 'No active grid strategies';
                cell.style.textAlign = 'center';
                row.appendChild(cell);
                tableBody.appendChild(row);
            }
            
            // Fetch and update market scores
            fetch('/api/market_scores')
                .then(response => response.json())
                .then(scores => {
                    updateMarketScores(scores);
                });
        }
        
        // Update market scores chart
        function updateMarketScores(scores) {
            const symbols = scores.map(s => s.symbol);
            const totalScores = scores.map(s => s.total_score);
            const expectedReturns = scores.map(s => s.expected_daily_return * 100);
            
            const data = [
                {
                    x: symbols,
                    y: totalScores,
                    type: 'bar',
                    name: 'Grid Score',
                    marker: {
                        color: '#667eea'
                    }
                },
                {
                    x: symbols,
                    y: expectedReturns,
                    type: 'bar',
                    name: 'Expected Daily Return (%)',
                    marker: {
                        color: '#4ade80'
                    }
                }
            ];
            
            const layout = {
                barmode: 'group',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                    color: '#e0e0e0'
                },
                margin: {
                    l: 50,
                    r: 50,
                    t: 30,
                    b: 100
                },
                xaxis: {
                    tickangle: -45
                }
            };
            
            Plotly.newPlot('market-scores-chart', data, layout);
        }
        
        // Stop a grid strategy
        function stopGrid(symbol) {
            fetch('/api/stop_grid', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol: symbol })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`Grid strategy for ${symbol} stopped successfully`);
                    socket.emit('request_update');
                } else {
                    alert(`Error: ${data.error}`);
                }
            });
        }
        
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            socket.emit('request_update');
        });
        
        // Scan markets button
        document.getElementById('scan-markets-btn').addEventListener('click', () => {
            fetch('/api/market_scores')
                .then(response => response.json())
                .then(scores => {
                    updateMarketScores(scores);
                    alert('Markets scanned successfully');
                });
        });
        
        // Initial update
        function updateStatus() {
            socket.emit('request_update');
            setTimeout(updateStatus, 30000); // Update every 30 seconds
        }
        
        // Start updates
        updateStatus();
    </script>
</body>
</html>
"""

class GridBotDashboard:
    """Web dashboard for monitoring and controlling the grid bot"""
    
    def __init__(self, bot, ml_optimizer, port: int = 5000):
        self.bot = bot
        self.ml_optimizer = ml_optimizer
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        self.port = port
        self.update_queue = Queue()
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio()
        
        # Start background thread for updates
        self.socketio.start_background_task(self.background_updater)
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return dashboard_html
        
        @self.app.route('/api/status')
        def get_status():
            """Get current bot status"""
            data = self.bot.get_dashboard_data()
            data['ml_enabled'] = len(self.ml_optimizer.performance_history) >= self.ml_optimizer.min_history_required
            data['total_history_records'] = len(self.ml_optimizer.performance_history)
            return jsonify(data)
        
        @self.app.route('/api/market_scores')
        def get_market_scores():
            """Get current market scores"""
            scores = asyncio.run(self.bot.scan_and_rank_markets())
            return jsonify([{
                'symbol': s.symbol,
                'total_score': s.total_score,
                'volatility_score': s.volatility_score,
                'range_score': s.range_score,
                'volume_score': s.volume_score,
                'current_price': s.current_price,
                'expected_daily_return': s.expected_daily_return
            } for s in scores[:10]])
        
        @self.app.route('/api/performance_history')
        def get_performance_history():
            """Get historical performance data"""
            history = []
            for record in self.ml_optimizer.performance_history[-100:]:
                history.append({
                    'timestamp': record['timestamp'],
                    'symbol': record['symbol'],
                    'daily_return': record['performance'].get('daily_return_percent', 0),
                    'trades': record['performance'].get('trades_per_day', 0)
                })
            return jsonify(history)
        
        @self.app.route('/api/feature_importance')
        def get_feature_importance():
            """Get ML feature importance"""
            return jsonify(self.ml_optimizer.feature_importance)
        
        @self.app.route('/api/optimize_grid', methods=['POST'])
        def optimize_grid():
            """Optimize grid parameters for a symbol"""
            data = request.json
            symbol = data.get('symbol')
            risk_preference = data.get('risk_preference', 'balanced')
            
            # Get market data
            market_data = asyncio.run(self.bot.analyzer.collect_market_data(symbol, 24))
            
            if market_data:
                optimized = self.ml_optimizer.optimize_grid_parameters(market_data, risk_preference)
                return jsonify(optimized)
            else:
                return jsonify({'error': 'Could not fetch market data'}), 400
        
        @self.app.route('/api/start_grid', methods=['POST'])
        def start_grid():
            """Manually start a grid strategy"""
            data = request.json
            symbol = data.get('symbol')
            
            # Find or create grid score
            scores = asyncio.run(self.bot.scan_and_rank_markets())
            score = next((s for s in scores if s.symbol == symbol), None)
            
            if score and score.total_score >= 50:
                asyncio.run(self.bot.start_grid_strategy(score))
                return jsonify({'success': True, 'message': f'Grid started for {symbol}'})
            else:
                return jsonify({'error': 'Symbol not suitable for grid trading'}), 400
        
        @self.app.route('/api/stop_grid', methods=['POST'])
        def stop_grid():
            """Stop a grid strategy"""
            data = request.json
            symbol = data.get('symbol')
            
            if symbol in self.bot.active_grids:
                asyncio.run(self.bot.close_grid_strategy(symbol, "Manual stop"))
                return jsonify({'success': True, 'message': f'Grid stopped for {symbol}'})
            else:
                return jsonify({'error': 'No active grid for this symbol'}), 400
    
    def setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Dashboard client connected')
            emit('connected', {'data': 'Connected to Grid Bot'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Dashboard client disconnected')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Send immediate update"""
            self.send_update()
    
    def background_updater(self):
        """Background task to send updates to dashboard"""
        while True:
            self.send_update()
            self.socketio.sleep(5)  # Update every 5 seconds
    
    def send_update(self):
        """Send update to all connected clients"""
        try:
            # Get current data
            data = self.bot.get_dashboard_data()
            
            # Add ML predictions for active grids
            for grid in data['active_grids']:
                symbol = grid['symbol']
                if symbol in self.bot.active_grids:
                    strategy = self.bot.active_grids[symbol]
                    # Add current performance metrics
                    grid['current_price'] = self.get_current_price(symbol)
                    grid['price_position'] = self.calculate_price_position(
                        grid['current_price'],
                        strategy.lower_bound,
                        strategy.upper_bound
                    )
            
            # Add system metrics
            data['system_metrics'] = {
                'uptime_hours': self.calculate_uptime(),
                'ml_model_accuracy': self.get_model_accuracy(),
                'total_volume_24h': self.calculate_total_volume()
            }
            
            self.socketio.emit('update', data)
            
        except Exception as e:
            logger.error(f"Error sending update: {e}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.bot.exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return 0
    
    def calculate_price_position(self, current: float, lower: float, upper: float) -> float:
        """Calculate where price is within the grid range (0-100%)"""
        if upper <= lower:
            return 50
        return ((current - lower) / (upper - lower)) * 100
    
    def calculate_uptime(self) -> float:
        """Calculate bot uptime in hours"""
        # This would be tracked properly in production
        return 0
    
    def get_model_accuracy(self) -> Dict[str, float]:
        """Get ML model accuracy metrics"""
        if hasattr(self.ml_optimizer, 'models'):
            return {
                'profit_r2': 0.75,  # Would be actual R² scores
                'trade_frequency_r2': 0.82,
                'risk_score_r2': 0.68
            }
        return {}
    
    def calculate_total_volume(self) -> float:
        """Calculate total trading volume in last 24h"""
        total = 0
        for strategy in self.bot.active_grids.values():
            total += strategy.total_trades * strategy.calculate_order_size() * 2  # Buy and sell
        return total
    
    def run(self):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on http://localhost:{self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)