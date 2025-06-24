const mongoose = require('mongoose');

const MetricsSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'user',
    required: true
  },
  period: {
    type: String,
    enum: ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly', 'Custom'],
    required: true
  },
  startDate: {
    type: Date,
    required: true
  },
  endDate: {
    type: Date,
    required: true
  },
  totalTrades: {
    type: Number,
    default: 0
  },
  winningTrades: {
    type: Number,
    default: 0
  },
  losingTrades: {
    type: Number,
    default: 0
  },
  breakEvenTrades: {
    type: Number,
    default: 0
  },
  winRate: {
    type: Number,
    default: 0
  },
  averageWin: {
    type: Number,
    default: 0
  },
  averageLoss: {
    type: Number,
    default: 0
  },
  largestWin: {
    type: Number,
    default: 0
  },
  largestLoss: {
    type: Number,
    default: 0
  },
  profitFactor: {
    type: Number,
    default: 0
  },
  expectancy: {
    type: Number,
    default: 0
  },
  systemQualityNumber: {
    type: Number,
    default: 0
  },
  sharpeRatio: {
    type: Number,
    default: 0
  },
  sortinoRatio: {
    type: Number,
    default: 0
  },
  calmarRatio: {
    type: Number,
    default: 0
  },
  maxDrawdown: {
    type: Number,
    default: 0
  },
  maxDrawdownPercentage: {
    type: Number,
    default: 0
  },
  averageRPerTrade: {
    type: Number,
    default: 0
  },
  netProfit: {
    type: Number,
    default: 0
  },
  netProfitPercentage: {
    type: Number,
    default: 0
  },
  equityCurve: [{
    date: {
      type: Date
    },
    equity: {
      type: Number
    }
  }],
  setupPerformance: [{
    setup: {
      type: String
    },
    count: {
      type: Number
    },
    winRate: {
      type: Number
    },
    expectancy: {
      type: Number
    }
  }],
  timeframePerformance: [{
    timeframe: {
      type: String
    },
    count: {
      type: Number
    },
    winRate: {
      type: Number
    },
    expectancy: {
      type: Number
    }
  }],
  instrumentPerformance: [{
    instrument: {
      type: String
    },
    count: {
      type: Number
    },
    winRate: {
      type: Number
    },
    expectancy: {
      type: Number
    }
  }],
  // Add duration performance analysis
  durationPerformance: [{
    durationCategory: {
      type: String
    },
    count: {
      type: Number
    },
    winRate: {
      type: Number
    },
    expectancy: {
      type: Number
    },
    averagePnl: {
      type: Number
    }
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

module.exports = Metrics = mongoose.model('metrics', MetricsSchema);