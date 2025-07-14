const express = require('express');
const router = express.Router();
const auth = require('../../middleware/auth');
const { getBybitTrades, importBybitTrades } = require('../../controllers/bybitController');

// @route   GET api/bybit/trades
// @desc    Get trade history from Bybit
// @access  Private
router.get('/trades', auth, getBybitTrades);

// @route   POST api/bybit/import
// @desc    Import trades from Bybit to journal
// @access  Private
router.post('/import', auth, importBybitTrades);

module.exports = router;