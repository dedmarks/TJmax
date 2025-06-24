const express = require('express');
const router = express.Router();
const auth = require('../../middleware/auth');
const validation = require('../../middleware/validation');
const { registerValidation } = require('../../utils/validators');
const { registerUser, updateUser } = require('../../controllers/userController');

// @route   POST api/users
// @desc    Register user
// @access  Public
router.post('/', registerValidation, validation, registerUser);

// @route   PUT api/users
// @desc    Update user profile
// @access  Private
router.put('/', auth, updateUser);

module.exports = router;