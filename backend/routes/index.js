const express = require('express');
const router = express.Router();

// API Routes
router.use('/api/users', require('./api/users'));
router.use('/api/auth', require('./api/auth'));
router.use('/api/journal', require('./api/journal'));
router.use('/api/metrics', require('./api/metrics'));
router.use('/api/upload', require('./api/upload'));
router.use('/api/bybit', require('./api/bybit')); // Add this line

module.exports = router;