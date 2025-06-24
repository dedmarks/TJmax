const { check } = require('express-validator');

exports.registerValidation = [
  check('name', 'Name is required').not().isEmpty(),
  check('email', 'Please include a valid email').isEmail(),
  check('password', 'Please enter a password with 6 or more characters').isLength({ min: 6 })
];

exports.loginValidation = [
  check('email', 'Please include a valid email').isEmail(),
  check('password', 'Password is required').exists()
];

exports.journalEntryValidation = [
  check('instrument', 'Instrument is required').not().isEmpty(),
  check('setup', 'Setup is required').not().isEmpty(),
  check('direction', 'Direction must be either Long or Short').isIn(['Long', 'Short']),
  check('entryDate', 'Entry date is required').not().isEmpty(),
  check('entryPrice', 'Entry price is required').isNumeric(),
  check('stopLoss', 'Stop loss is required').isNumeric(),
  check('positionSize', 'Position size is required').isNumeric(),
  check('riskAmount', 'Risk amount is required').isNumeric(),
  check('riskPercentage', 'Risk percentage is required').isNumeric(),
  check('timeframe', 'Timeframe is required').not().isEmpty()
];

exports.metricsValidation = [
  check('period', 'Period is required').isIn(['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly', 'Custom']),
  check('startDate', 'Start date is required').not().isEmpty(),
  check('endDate', 'End date is required').not().isEmpty()
];