const express = require('express');
const router = express.Router();
const auth = require('../../middleware/auth');
const validation = require('../../middleware/validation');
const { loginValidation } = require('../../utils/validators');
const { login, getUser } = require('../../controllers/authController');

// @route   GET api/auth
// @desc    Get user by token
// @access  Private
router.get('/', auth, getUser);

// @route   POST api/auth
// @desc    Authenticate user & get token
// @access  Public
router.post('/', loginValidation, validation, login);

module.exports = router;