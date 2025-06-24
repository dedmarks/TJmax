const mongoose = require('mongoose');

const JournalSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'user',
    required: true
  },
  instrument: {
    type: String,
    required: true
  },
  setup: {
    type: String,
    required: true
  },
  direction: {
    type: String,
    enum: ['Long', 'Short'],
    required: true
  },
  entryDate: {
    type: Date,
    required: true
  },
  entryPrice: {
    type: Number,
    required: true
  },
  exitDate: {
    type: Date
  },
  exitPrice: {
    type: Number
  },
  stopLoss: {
    type: Number,
    required: true
  },
  targetPrice: {
    type: Number
  },
  positionSize: {
    type: Number,
    required: true
  },
  riskAmount: {
    type: Number,
    required: true
  },
  riskPercentage: {
    type: Number,
    required: true
  },
  rewardToRisk: {
    type: Number
  },
  outcome: {
    type: String,
    enum: ['Win', 'Loss', 'Breakeven', 'Open']
  },
  pnl: {
    type: Number
  },
  pnlPercentage: {
    type: Number
  },
  fees: {
    type: Number,
    default: 0
  },
  notes: {
    type: String
  },
  images: [{
    url: {
      type: String,
      required: true
    },
    description: {
      type: String,
      default: ''
    },
    category: {
      type: String,
      enum: ['chart', 'setup', 'entry', 'exit', 'other'],
      default: 'other'
    },
    uploadedAt: {
      type: Date,
      default: Date.now
    }
  }],
  timeframe: {
    type: String,
    required: true
  },
  marketCondition: {
    type: String,
    enum: ['Trending', 'Ranging', 'Volatile', 'Quiet']
  },
  psychologicalState: {
    before: {
      state: {
        type: String,
        enum: ['Calm', 'Excited', 'Fearful', 'Confident', 'Uncertain', 'Anxious', 'Focused', 'Distracted', 'Optimistic', 'Pessimistic']
      },
      confidence: {
        type: Number,
        min: 1,
        max: 10
      },
      stress: {
        type: Number,
        min: 1,
        max: 10
      },
      focus: {
        type: Number,
        min: 1,
        max: 10
      },
      triggers: [{
        type: String
      }],
      notes: {
        type: String
      }
    },
    during: {
      state: {
        type: String,
        enum: ['Calm', 'Excited', 'Fearful', 'Confident', 'Uncertain', 'Anxious', 'Focused', 'Distracted', 'Optimistic', 'Pessimistic']
      },
      confidence: {
        type: Number,
        min: 1,
        max: 10
      },
      stress: {
        type: Number,
        min: 1,
        max: 10
      },
      focus: {
        type: Number,
        min: 1,
        max: 10
      },
      triggers: [{
        type: String
      }],
      notes: {
        type: String
      }
    },
    after: {
      state: {
        type: String,
        enum: ['Calm', 'Excited', 'Fearful', 'Confident', 'Uncertain', 'Anxious', 'Focused', 'Distracted', 'Optimistic', 'Pessimistic']
      },
      confidence: {
        type: Number,
        min: 1,
        max: 10
      },
      stress: {
        type: Number,
        min: 1,
        max: 10
      },
      focus: {
        type: Number,
        min: 1,
        max: 10
      },
      triggers: [{
        type: String
      }],
      notes: {
        type: String
      }
    },
    lessons: {
      type: String
    }
  },
  maxAdverseExcursion: {
    type: Number
  },
  maxFavorableExcursion: {
    type: Number
  },
  mistakes: [{
    type: String
  }],
  tags: [{
    type: String
  }],
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = Journal = mongoose.model('journal', JournalSchema);