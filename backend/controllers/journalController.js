const Journal = require('../models/Journal');
const { calculatePnL, calculateRiskReward } = require('../utils/calculateMetrics');

// @route   POST api/journal
// @desc    Create a journal entry
// @access  Private
exports.createEntry = async (req, res) => {
  try {
    const newEntry = new Journal({
      ...req.body,
      user: req.user.id
    });

    // Calculate PnL if trade is closed
    if (newEntry.exitPrice && newEntry.entryPrice) {
      const { pnl, pnlPercentage } = calculatePnL(
        newEntry.direction,
        newEntry.entryPrice,
        newEntry.exitPrice,
        newEntry.positionSize,
        newEntry.fees
      );
      newEntry.pnl = pnl;
      newEntry.pnlPercentage = pnlPercentage;
      
      // Set outcome based on PnL
      if (pnl > 0) {
        newEntry.outcome = 'Win';
      } else if (pnl < 0) {
        newEntry.outcome = 'Loss';
      } else {
        newEntry.outcome = 'Breakeven';
      }
    } else {
      // Trade is still open
      newEntry.outcome = 'Open';
    }

    // Calculate reward to risk ratio
    if (newEntry.targetPrice && newEntry.stopLoss && newEntry.entryPrice) {
      newEntry.rewardToRisk = calculateRiskReward(
        newEntry.direction,
        newEntry.entryPrice,
        newEntry.targetPrice,
        newEntry.stopLoss
      );
    }

    const entry = await newEntry.save();
    res.json(entry);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};

// @route   GET api/journal
// @desc    Get all journal entries for a user
// @access  Private
exports.getEntries = async (req, res) => {
  try {
    const entries = await Journal.find({ user: req.user.id }).sort({ entryDate: -1 });
    res.json(entries);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};

// @route   GET api/journal/:id
// @desc    Get journal entry by ID
// @access  Private
exports.getEntryById = async (req, res) => {
  try {
    const entry = await Journal.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }

    // Check user
    if (entry.user.toString() !== req.user.id) {
      return res.status(401).json({ msg: 'User not authorized' });
    }

    res.json(entry);
  } catch (err) {
    console.error(err.message);
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }
    res.status(500).send('Server Error');
  }
};

// @route   PUT api/journal/:id
// @desc    Update journal entry
// @access  Private
exports.updateEntry = async (req, res) => {
  try {
    const entry = await Journal.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }

    // Check user
    if (entry.user.toString() !== req.user.id) {
      return res.status(401).json({ msg: 'User not authorized' });
    }

    // Update fields
    const updatedEntry = { ...req.body, updatedAt: Date.now() };

    // Calculate PnL if trade is closed
    if (updatedEntry.exitPrice && updatedEntry.entryPrice) {
      const { pnl, pnlPercentage } = calculatePnL(
        updatedEntry.direction,
        updatedEntry.entryPrice,
        updatedEntry.exitPrice,
        updatedEntry.positionSize,
        updatedEntry.fees
      );
      updatedEntry.pnl = pnl;
      updatedEntry.pnlPercentage = pnlPercentage;
      
      // Set outcome based on PnL
      if (pnl > 0) {
        updatedEntry.outcome = 'Win';
      } else if (pnl < 0) {
        updatedEntry.outcome = 'Loss';
      } else {
        updatedEntry.outcome = 'Breakeven';
      }
    } else {
      // Trade is still open
      updatedEntry.outcome = 'Open';
    }

    // Calculate reward to risk ratio
    if (updatedEntry.targetPrice && updatedEntry.stopLoss && updatedEntry.entryPrice) {
      updatedEntry.rewardToRisk = calculateRiskReward(
        updatedEntry.direction,
        updatedEntry.entryPrice,
        updatedEntry.targetPrice,
        updatedEntry.stopLoss
      );
    }

    const updated = await Journal.findByIdAndUpdate(
      req.params.id,
      { $set: updatedEntry },
      { new: true }
    );

    res.json(updated);
  } catch (err) {
    console.error(err.message);
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }
    res.status(500).send('Server Error');
  }
};


// @route   DELETE api/journal/:id
// @desc    Delete journal entry
// @access  Private
exports.deleteEntry = async (req, res) => {
  try {
    const entry = await Journal.findById(req.params.id);

    if (!entry) {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }

    // Check user
    if (entry.user.toString() !== req.user.id) {
      return res.status(401).json({ msg: 'User not authorized' });
    }

    await Journal.findByIdAndDelete(req.params.id);

    res.json({ msg: 'Journal entry removed' });
  } catch (err) {
    console.error(err.message);
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ msg: 'Journal entry not found' });
    }
    res.status(500).send('Server Error');
  }
};

// @route   GET api/journal/total-profit
// @desc    Get all-time total profit for a user
// @access  Private
exports.getAllTimeTotalProfit = async (req, res) => {
  try {
    const entries = await Journal.find({ 
      user: req.user.id,
      outcome: { $ne: 'Open' } // Exclude open trades
    });
    
    const totalProfit = entries.reduce((sum, entry) => sum + (entry.pnl || 0), 0);
    
    res.json({ totalProfit });
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};