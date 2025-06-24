const express = require('express');
const router = express.Router();
const auth = require('../../middleware/auth');
const validation = require('../../middleware/validation');
const { metricsValidation } = require('../../utils/validators');
const {
  generateMetrics,
  getMetrics,
  getMetricsById
} = require('../../controllers/metricsController');

// @route   POST api/metrics
// @desc    Generate metrics for a specific period
// @access  Private
router.post('/', [auth, metricsValidation, validation], generateMetrics);

// @route   GET api/metrics
// @desc    Get all metrics for a user
// @access  Private
router.get('/', auth, getMetrics);

// @route   GET api/metrics/:id
// @desc    Get metrics by ID
// @access  Private
router.get('/:id', auth, getMetricsById);

module.exports = router;