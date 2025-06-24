// Calculate PnL for a trade
exports.calculatePnL = (direction, entryPrice, exitPrice, positionSize, fees = 0) => {
  let pnl = 0;
  
  if (direction === 'Long') {
    pnl = (exitPrice - entryPrice) * positionSize - fees;
  } else if (direction === 'Short') {
    pnl = (entryPrice - exitPrice) * positionSize - fees;
  }
  
  const pnlPercentage = (pnl / (entryPrice * positionSize)) * 100;
  
  return { pnl, pnlPercentage };
};

// Calculate reward to risk ratio
exports.calculateRiskReward = (direction, entryPrice, targetPrice, stopLoss) => {
  let reward = 0;
  let risk = 0;
  
  if (direction === 'Long') {
    reward = targetPrice - entryPrice;
    risk = entryPrice - stopLoss;
  } else if (direction === 'Short') {
    reward = entryPrice - targetPrice;
    risk = stopLoss - entryPrice;
  }
  
  return risk > 0 ? reward / risk : 0;
};

// Calculate win rate
exports.calculateWinRate = (winningTrades, totalTrades) => {
  return totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
};

// Calculate expectancy
exports.calculateExpectancy = (winRate, averageWin, averageLoss) => {
  const winRateDecimal = winRate / 100;
  return (winRateDecimal * averageWin) - ((1 - winRateDecimal) * averageLoss);
};

// Calculate equity curve
exports.calculateEquityCurve = (entries, initialCapital = 10000) => {
  if (!entries || entries.length === 0) {
    return [{
      date: new Date(),
      equity: initialCapital
    }];
  }

  // Sort entries by exit date
  const sortedEntries = [...entries].sort((a, b) => 
    new Date(a.exitDate || a.entryDate) - new Date(b.exitDate || b.entryDate)
  );
  
  let currentEquity = initialCapital;
  
  const equityCurve = [{
    date: new Date(sortedEntries[0].entryDate),
    equity: initialCapital
  }];
  
  sortedEntries.forEach(entry => {
    currentEquity += entry.pnl || 0;
    equityCurve.push({
      date: new Date(entry.exitDate || entry.entryDate),
      equity: currentEquity
    });
  });
  
  return equityCurve;
};

// Calculate drawdown
exports.calculateDrawdown = (equityCurve) => {
  let peak = 0;
  let maxDrawdown = 0;
  let maxDrawdownPercentage = 0;
  
  equityCurve.forEach(point => {
    if (point.equity > peak) {
      peak = point.equity;
    }
    
    const drawdown = peak - point.equity;
    const drawdownPercentage = (drawdown / peak) * 100;
    
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
      maxDrawdownPercentage = drawdownPercentage;
    }
  });
  
  return { maxDrawdown, maxDrawdownPercentage };
};

// Calculate Sharpe Ratio
exports.calculateSharpeRatio = (dailyReturns) => {
  if (dailyReturns.length === 0) return 0;
  
  const riskFreeRate = 0.02 / 252; // Assuming 2% annual risk-free rate, divided by trading days
  
  // Calculate average return
  const avgReturn = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
  
  // Calculate standard deviation
  const variance = dailyReturns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / dailyReturns.length;
  const stdDev = Math.sqrt(variance);
  
  // Calculate Sharpe ratio
  return stdDev > 0 ? (avgReturn - riskFreeRate) / stdDev * Math.sqrt(252) : 0; // Annualized
};

// Calculate Sortino Ratio
exports.calculateSortinoRatio = (dailyReturns) => {
  if (dailyReturns.length === 0) return 0;
  
  const riskFreeRate = 0.02 / 252; // Assuming 2% annual risk-free rate, divided by trading days
  
  // Calculate average return
  const avgReturn = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
  
  // Calculate downside deviation (only negative returns)
  const negativeReturns = dailyReturns.filter(ret => ret < 0);
  
  if (negativeReturns.length === 0) return 0; // No negative returns
  
  const downsideVariance = negativeReturns.reduce((sum, ret) => sum + Math.pow(ret, 2), 0) / negativeReturns.length;
  const downsideDeviation = Math.sqrt(downsideVariance);
  
  // Calculate Sortino ratio
  return downsideDeviation > 0 ? (avgReturn - riskFreeRate) / downsideDeviation * Math.sqrt(252) : 0; // Annualized
};

// Calculate Calmar Ratio
exports.calculateCalmarRatio = (equityCurve, maxDrawdownPercentage) => {
  if (equityCurve.length < 2 || maxDrawdownPercentage === 0) return 0;
  
  const firstPoint = equityCurve[0];
  const lastPoint = equityCurve[equityCurve.length - 1];
  
  const totalReturn = (lastPoint.equity - firstPoint.equity) / firstPoint.equity;
  const timeInYears = (lastPoint.date - firstPoint.date) / (365 * 24 * 60 * 60 * 1000);
  
  const annualizedReturn = Math.pow(1 + totalReturn, 1 / timeInYears) - 1;
  
  return maxDrawdownPercentage > 0 ? annualizedReturn / (maxDrawdownPercentage / 100) : 0;
};

// Calculate System Quality Number (SQN)
exports.calculateSystemQualityNumber = (entries) => {
  if (entries.length < 2) return 0;
  
  // Calculate R-multiples (profit/loss divided by initial risk)
  const rMultiples = entries.map(entry => {
    const initialRisk = entry.riskAmount || 1; // Fallback to 1 if not available
    return entry.pnl / initialRisk;
  });
  
  // Calculate mean of R-multiples
  const mean = rMultiples.reduce((sum, r) => sum + r, 0) / rMultiples.length;
  
  // Calculate standard deviation of R-multiples
  const variance = rMultiples.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / rMultiples.length;
  const stdDev = Math.sqrt(variance);
  
  // Calculate SQN
  return stdDev > 0 ? (mean * Math.sqrt(rMultiples.length)) / stdDev : 0;
};


// Calculate trade duration in minutes for intraday trades and days for longer trades
exports.calculateTradeDuration = (entryDate, exitDate) => {
  if (!exitDate) return null;
  
  const entry = new Date(entryDate);
  const exit = new Date(exitDate);
  
  // Calculate difference in milliseconds
  const durationMs = exit.getTime() - entry.getTime();
  
  // If less than a day, return duration in minutes
  if (durationMs < 24 * 60 * 60 * 1000) {
    const durationMinutes = durationMs / (1000 * 60);
    return {
      value: parseFloat(durationMinutes.toFixed(2)),
      unit: 'minutes'
    };
  }
  
  // Otherwise return duration in days
  const durationDays = durationMs / (1000 * 60 * 60 * 24);
  return {
    value: parseFloat(durationDays.toFixed(2)),
    unit: 'days'
  };
};

// Categorize trade duration
exports.categorizeTradeDuration = (duration) => {
  if (duration === null) return 'Open';
  
  // Handle intraday trades with minute-based intervals
  if (duration.unit === 'minutes') {
    if (duration.value <= 15) return '0-15min';
    if (duration.value <= 30) return '15-30min';
    if (duration.value <= 60) return '30min-1h';
    if (duration.value <= 120) return '1h-2h';
    if (duration.value <= 240) return '2h-4h';
    return '4h+';
  }
  
  // Handle multi-day trades
  if (duration.value < 3) return '1-2 Days';
  if (duration.value < 7) return '3-6 Days';
  if (duration.value < 14) return '1-2 Weeks';
  if (duration.value < 30) return '2-4 Weeks';
  return '1+ Month';
};