const { RestClientV5 } = require('bybit-api');
require('dotenv').config();

// Initialize Bybit API client
const bybitClient = new RestClientV5({
  key: process.env.BYBIT_API_KEY,
  secret: process.env.BYBIT_API_SECRET,
  testnet: process.env.BYBIT_TESTNET === 'true',
});

/**
 * Fetch trade history from Bybit API using closed-pnl endpoint
 * @param {Object} params - Parameters for the trade history request
 * @param {string} params.category - Product category (e.g., 'linear', 'inverse')
 * @param {string} params.symbol - Trading symbol (e.g., 'BTCUSDT')
 * @param {number} params.limit - Number of records to fetch
 * @param {string} params.startTime - Start timestamp in milliseconds
 * @param {string} params.endTime - End timestamp in milliseconds
 * @returns {Promise<Array>} - Array of trade records
 */
async function getTradeHistory(params) {
  try {
    // The closed-pnl endpoint only works for linear and inverse categories
    if (!['linear', 'inverse'].includes(params.category)) {
      throw new Error('The closed-pnl endpoint only supports linear and inverse categories');
    }
    
    const response = await bybitClient.getClosedPnL(params);
    
    // Log the full API response
    console.log('Full Bybit API Response:', JSON.stringify(response, null, 2));
    
    if (response.retCode !== 0) {
      throw new Error(`Bybit API Error: ${response.retMsg}`);
    }
    
    return response.result.list;
  } catch (error) {
    console.error('Error fetching Bybit trade history:', error);
    throw error;
  }
}

/**
 * Convert Bybit closed-pnl data to Journal format
 * @param {Array} trades - Array of closed-pnl records from Bybit
 * @returns {Array} - Array of trades in Journal format
 */
function convertTradesToJournalFormat(trades) {
  return trades.map(trade => {
    console.log('Processing trade:', JSON.stringify(trade, null, 2));
    
    // Determine direction based on side
    const direction = trade.side === 'Buy' ? 'Long' : 'Short';
    
    // Calculate profit/loss
    let pnl = parseFloat(trade.closedPnl || 0);
    let outcome = 'Open';
    
    if (pnl !== 0) {
      outcome = pnl > 0 ? 'Win' : pnl < 0 ? 'Loss' : 'Breakeven';
    }
    
    // Parse dates safely
    let entryDate, exitDate;
    try {
      entryDate = new Date(parseInt(trade.createdTime));
    } catch (e) {
      console.error('Error parsing entry date:', e);
      entryDate = new Date();
    }
    
    try {
      exitDate = new Date(parseInt(trade.updatedTime));
    } catch (e) {
      console.error('Error parsing exit date:', e);
      exitDate = new Date();
    }
    
    // Parse prices and sizes safely
    const entryPrice = parseFloat(trade.avgEntryPrice || trade.entryPrice || 0);
    const exitPrice = parseFloat(trade.avgExitPrice || 0);
    const positionSize = parseFloat(trade.size || trade.qty || 0);
    const fees = parseFloat(trade.cumRealisedPnl || 0) - pnl; // Estimate fees as difference between realized PnL and closed PnL
    
    return {
      instrument: trade.symbol,
      setup: 'Bybit Import', // Default setup name
      direction: direction,
      entryDate: entryDate,
      entryPrice: entryPrice,
      exitDate: exitDate,
      exitPrice: exitPrice,
      stopLoss: 0, // This needs to be set by the user later
      positionSize: positionSize,
      riskAmount: 0, // This needs to be set by the user later
      riskPercentage: 0, // This needs to be set by the user later
      outcome: outcome,
      pnl: pnl,
      fees: fees,
      timeframe: '1h', // Default timeframe, can be adjusted by user
      notes: `Imported from Bybit - Position ID: ${trade.orderId || trade.orderLinkId || 'N/A'}`,
    };
  });
}

module.exports = {
  getTradeHistory,
  convertTradesToJournalFormat
};