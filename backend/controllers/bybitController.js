const { getTradeHistory, convertTradesToJournalFormat } = require('../utils/bybitService');
const Journal = require('../models/Journal');

// @route   GET api/bybit/trades
// @desc    Get trade history from Bybit
// @access  Private
exports.getBybitTrades = async (req, res) => {
  try {
    const { category, symbol, limit, startTime, endTime } = req.query;
    
    // Validate required parameters
    if (!category) {
      return res.status(400).json({ msg: 'Category is required' });
    }
    
    const params = {
      category,
      limit: limit ? parseInt(limit) : 100
    };
    
    // Add optional parameters if provided
    if (symbol) params.symbol = symbol;
    if (startTime) params.startTime = startTime;
    if (endTime) params.endTime = endTime;
    
    const trades = await getTradeHistory(params);
    res.json(trades);
  } catch (error) {
    console.error('Error in getBybitTrades:', error.message);
    res.status(500).send('Server Error');
  }
};

// @route   POST api/bybit/import
// @desc    Import trades from Bybit to journal
// @access  Private
exports.importBybitTrades = async (req, res) => {
  try {
    const { trades } = req.body;
    
    if (!trades || !Array.isArray(trades) || trades.length === 0) {
      return res.status(400).json({ msg: 'No valid trades provided for import' });
    }
    
    // Convert trades to journal format
    const journalEntries = convertTradesToJournalFormat(trades);
    
    // Add user ID to each entry
    const entriesWithUser = journalEntries.map(entry => ({
      ...entry,
      user: req.user.id
    }));
    
    // Save entries to database
    await Journal.insertMany(entriesWithUser);
    
    res.json({ msg: `Successfully imported ${entriesWithUser.length} trades` });
  } catch (error) {
    console.error('Error in importBybitTrades:', error.message);
    res.status(500).send('Server Error');
  }
};