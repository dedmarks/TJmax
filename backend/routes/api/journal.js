const express = require('express');
const router = express.Router();
const auth = require('../../middleware/auth');
const validation = require('../../middleware/validation');
const { journalEntryValidation } = require('../../utils/validators');
const {
  createEntry,
  getEntries,
  getEntryById,
  updateEntry,
  deleteEntry,
  getAllTimeTotalProfit
} = require('../../controllers/journalController');

// @route   POST api/journal
// @desc    Create a journal entry
// @access  Private
router.post('/', [auth, journalEntryValidation, validation], createEntry);

// @route   GET api/journal
// @desc    Get all journal entries for a user
// @access  Private
router.get('/', auth, getEntries);

// @route   GET api/journal/total-profit
// @desc    Get all-time total profit for a user
// @access  Private
router.get('/total-profit', auth, getAllTimeTotalProfit);

// @route   GET api/journal/:id
// @desc    Get journal entry by ID
// @access  Private
router.get('/:id', auth, getEntryById);

// @route   PUT api/journal/:id
// @desc    Update journal entry
// @access  Private
router.put('/:id', auth, updateEntry);

// @route   DELETE api/journal/:id
// @desc    Delete journal entry
// @access  Private
router.delete('/:id', auth, deleteEntry);

module.exports = router;