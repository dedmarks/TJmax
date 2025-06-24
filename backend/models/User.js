const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  password: {
    type: String,
    required: true
  },
  avatar: {
    type: String
  },
  date: {
    type: Date,
    default: Date.now
  },
  preferences: {
    defaultRiskPercentage: {
      type: Number,
      default: 1.0
    },
    defaultTimeframe: {
      type: String,
      default: 'Daily'
    },
    initialCapital: {
      type: Number,
      default: 10000
    },
    tradingHours: {
      start: {
        type: String,
        default: '08:00'
      },
      end: {
        type: String,
        default: '16:00'
      }
    },
    predefinedSetups: [{
      name: {
        type: String,
        required: true
      },
      description: {
        type: String
      }
    }]
  }
});

module.exports = User = mongoose.model('user', UserSchema);