const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const config = require('config');
const User = require('../models/User');

// @route   POST api/users
// @desc    Register user
// @access  Public
exports.registerUser = async (req, res) => {
  const { name, email, password, initialCapital } = req.body;
  
  console.log('Registration request body:', req.body); // Debug log
  console.log('Initial capital received:', initialCapital); // Debug log

  try {
    // See if user exists
    let user = await User.findOne({ email });

    if (user) {
      return res.status(400).json({ errors: [{ msg: 'User already exists' }] });
    }

    const userInitialCapital = initialCapital || 10000;
    console.log('Using initial capital:', userInitialCapital); // Debug log

    user = new User({
      name,
      email,
      password,
      preferences: {
        initialCapital: userInitialCapital,
        defaultRiskPercentage: 1.0,
        defaultTimeframe: 'Daily',
        tradingHours: {
          start: '08:00',
          end: '16:00'
        }
      }
    });

    // Encrypt password
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(password, salt);

    await user.save();
    console.log('User saved with preferences:', user.preferences); // Debug log

    // Return jsonwebtoken
    const payload = {
      user: {
        id: user.id
      }
    };

    jwt.sign(
      payload,
      config.get('jwtSecret'),
      { expiresIn: config.get('jwtExpiration') },
      (err, token) => {
        if (err) throw err;
        res.json({ token });
      }
    );
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server error');
  }
};

// @route   PUT api/users
// @desc    Update user profile
// @access  Private
exports.updateUser = async (req, res) => {
  const { name, preferences } = req.body;

  try {
    const user = await User.findById(req.user.id);

    if (!user) {
      return res.status(404).json({ msg: 'User not found' });
    }

    if (name) user.name = name;
    if (preferences) {
      console.log('Current preferences:', user.preferences);
      console.log('New preferences:', preferences);
      
      // Replace the entire preferences object instead of merging
      user.preferences = preferences;
      
      console.log('Updated preferences:', user.preferences);
    }

    await user.save();

    res.json(user);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};
exports.updateUser = async (req, res) => {
  const { name, preferences } = req.body;

  try {
    const user = await User.findById(req.user.id);

    if (!user) {
      return res.status(404).json({ msg: 'User not found' });
    }

    if (name) user.name = name;
    if (preferences) user.preferences = { ...user.preferences, ...preferences };

    await user.save();

    res.json(user);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};