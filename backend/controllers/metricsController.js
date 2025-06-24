const Journal = require('../models/Journal');
const User = require('../models/User');
const Metrics = require('../models/Metrics');
const { 
  calculateWinRate, 
  calculateExpectancy, 
  calculateSharpeRatio,
  calculateSortinoRatio,
  calculateCalmarRatio,
  calculateEquityCurve,
  calculateDrawdown,
  calculateSystemQualityNumber,
  calculateTradeDuration,
  categorizeTradeDuration
} = require('../utils/calculateMetrics');

// @route   POST api/metrics
// @desc    Generate metrics for a specific period
// @access  Private
exports.generateMetrics = async (req, res) => {
  try {
    const { period, startDate, endDate } = req.body;
    
    // Get user to access initial capital
    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({ msg: 'User not found' });
    }
    
    // Find all journal entries for the user within the date range
    // Find all journal entries for the user within the date range
    const entries = await Journal.find({
      user: req.user.id,
      entryDate: { $gte: new Date(startDate), $lte: new Date(endDate) },
      outcome: { $ne: 'Open' } // Excludes open trades
    });

    if (entries.length === 0) {
      return res.status(400).json({ msg: 'No closed trades found in this period' });
    }

    // Calculate basic metrics
    const totalTrades = entries.length;
    const winningTrades = entries.filter(entry => entry.outcome === 'Win').length;
    const losingTrades = entries.filter(entry => entry.outcome === 'Loss').length;
    const breakEvenTrades = entries.filter(entry => entry.outcome === 'Breakeven').length;
    
    const winRate = calculateWinRate(winningTrades, totalTrades);
    
    // Calculate profit metrics
    const winningEntries = entries.filter(entry => entry.outcome === 'Win');
    const losingEntries = entries.filter(entry => entry.outcome === 'Loss');
    
    const totalProfit = winningEntries.reduce((sum, entry) => sum + entry.pnl, 0);
    const totalLoss = Math.abs(losingEntries.reduce((sum, entry) => sum + entry.pnl, 0));
    
    const averageWin = winningEntries.length > 0 ? totalProfit / winningEntries.length : 0;
    const averageLoss = losingEntries.length > 0 ? totalLoss / losingEntries.length : 0;
    
    const largestWin = winningEntries.length > 0 ? 
      Math.max(...winningEntries.map(entry => entry.pnl)) : 0;
    const largestLoss = losingEntries.length > 0 ? 
      Math.max(...losingEntries.map(entry => Math.abs(entry.pnl))) : 0;
    
    const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Infinity : 0;
    
    // Calculate expectancy
    const expectancy = calculateExpectancy(winRate, averageWin, averageLoss);
    
    // Calculate equity curve
    // Calculate equity curve with user's initial capital
    const initialCapital = user.preferences?.initialCapital || 10000;
    console.log('User preferences:', user.preferences); // Debug log
    console.log('Using initial capital for equity curve:', initialCapital); // Debug log
    const equityCurve = calculateEquityCurve(entries, initialCapital);
    console.log('First equity point:', equityCurve[0]); // Debug log
    
    // Calculate drawdown
    const { maxDrawdown, maxDrawdownPercentage } = calculateDrawdown(equityCurve);
    
    // Calculate advanced ratios
    const dailyReturns = calculateDailyReturns(entries, startDate, endDate);
    const sharpeRatio = calculateSharpeRatio(dailyReturns);
    const sortinoRatio = calculateSortinoRatio(dailyReturns);
    const calmarRatio = calculateCalmarRatio(equityCurve, maxDrawdownPercentage);
    
    // Calculate system quality number
    const systemQualityNumber = calculateSystemQualityNumber(entries);
    
    // Calculate performance by setup, timeframe, and instrument
    const setupPerformance = calculatePerformanceByCategory(entries, 'setup');
    const timeframePerformance = calculatePerformanceByCategory(entries, 'timeframe');
    const instrumentPerformance = calculatePerformanceByCategory(entries, 'instrument');
    
    // Calculate performance by trade duration
    const durationPerformance = calculatePerformanceByDuration(entries);
    
    // Calculate net profit using user's initial capital
    const netProfit = entries.reduce((sum, entry) => sum + entry.pnl, 0);
    const netProfitPercentage = (netProfit / initialCapital) * 100;
    
    // Create or update metrics  document
    let metrics = await Metrics.findOne({
      user: req.user.id,
      period,
      startDate: new Date(startDate),
      endDate: new Date(endDate)
    });
    
    // Update the metrics object to include durationPerformance
    if (metrics) {
      // Update existing metrics
      metrics = await Metrics.findByIdAndUpdate(
        metrics._id,
        {
          $set: {
            totalTrades,
            winningTrades,
            losingTrades,
            breakEvenTrades,
            winRate,
            averageWin,
            averageLoss,
            largestWin,
            largestLoss,
            profitFactor,
            expectancy,
            systemQualityNumber,
            sharpeRatio,
            sortinoRatio,
            calmarRatio,
            maxDrawdown,
            maxDrawdownPercentage,
            netProfit,
            netProfitPercentage,
            equityCurve,
            setupPerformance,
            timeframePerformance,
            instrumentPerformance,
            durationPerformance,
            updatedAt: Date.now()
          }
        },
        { new: true }
      );
    } else {
      // Create new metrics
      // Create new metrics
      metrics = new Metrics({
        user: req.user.id,
        period,
        startDate: new Date(startDate),
        endDate: new Date(endDate),
        totalTrades,
        winningTrades,
        losingTrades,
        breakEvenTrades,
        winRate,
        averageWin,
        averageLoss,
        largestWin,
        largestLoss,
        profitFactor,
        expectancy,
        systemQualityNumber,
        sharpeRatio,
        sortinoRatio,
        calmarRatio,
        maxDrawdown,
        maxDrawdownPercentage,
        netProfit,
        netProfitPercentage,
        equityCurve,
        setupPerformance,
        timeframePerformance,
        instrumentPerformance,
        durationPerformance  // Add this line
      });
      
      await metrics.save();
    }
    
    res.json(metrics);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};

// @route   GET api/metrics
// @desc    Get all metrics for a user
// @access  Private
exports.getMetrics = async (req, res) => {
  try {
    const metrics = await Metrics.find({ user: req.user.id }).sort({ createdAt: -1 });
    res.json(metrics);
  } catch (err) {
    console.error(err.message);
    res.status(500).send('Server Error');
  }
};

// @route   GET api/metrics/:id
// @desc    Get metrics by ID
// @access  Private
exports.getMetricsById = async (req, res) => {
  try {
    const metrics = await Metrics.findById(req.params.id);

    if (!metrics) {
      return res.status(404).json({ msg: 'Metrics not found' });
    }

    // Check user
    if (metrics.user.toString() !== req.user.id) {
      return res.status(401).json({ msg: 'User not authorized' });
    }

    res.json(metrics);
  } catch (err) {
    console.error(err.message);
    if (err.kind === 'ObjectId') {
      return res.status(404).json({ msg: 'Metrics not found' });
    }
    res.status(500).send('Server Error');
  }
};

// Helper function to calculate performance by category (setup, timeframe, instrument)
const calculatePerformanceByCategory = (entries, category) => {
  const categories = {};
  
  // Group entries by category
  entries.forEach(entry => {
    const categoryValue = entry[category];
    if (!categories[categoryValue]) {
      categories[categoryValue] = {
        entries: [],
        wins: 0,
        losses: 0
      };
    }
    
    categories[categoryValue].entries.push(entry);
    if (entry.outcome === 'Win') {
      categories[categoryValue].wins++;
    } else if (entry.outcome === 'Loss') {
      categories[categoryValue].losses++;
    }
  });
  
  // Calculate metrics for each category
  return Object.keys(categories).map(categoryValue => {
    const categoryData = categories[categoryValue];
    const count = categoryData.entries.length;
    const winRate = calculateWinRate(categoryData.wins, count);
    
    // Calculate expectancy for this category
    const winningEntries = categoryData.entries.filter(entry => entry.outcome === 'Win');
    const losingEntries = categoryData.entries.filter(entry => entry.outcome === 'Loss');
    
    const averageWin = winningEntries.length > 0 ?
      winningEntries.reduce((sum, entry) => sum + entry.pnl, 0) / winningEntries.length : 0;
    const averageLoss = losingEntries.length > 0 ?
      Math.abs(losingEntries.reduce((sum, entry) => sum + entry.pnl, 0)) / losingEntries.length : 0;
    
    const expectancy = calculateExpectancy(winRate, averageWin, averageLoss);
    
    return {
      [category]: categoryValue,
      count,
      winRate,
      expectancy
    };
  });
};

// Helper function to calculate daily returns for Sharpe and Sortino ratios
const calculateDailyReturns = (entries, startDate, endDate) => {
  // Create a map of dates to PnL
  const dailyPnL = {};
  
  entries.forEach(entry => {
    const exitDate = entry.exitDate.toISOString().split('T')[0];
    if (!dailyPnL[exitDate]) {
      dailyPnL[exitDate] = 0;
    }
    dailyPnL[exitDate] += entry.pnl;
  });
  
  // Convert to array of daily returns
  const dailyReturns = [];
  const initialCapital = 10000; // This should be configurable
  
  let currentDate = new Date(startDate);
  const endDateObj = new Date(endDate);
  
  while (currentDate <= endDateObj) {
    const dateStr = currentDate.toISOString().split('T')[0];
    const pnl = dailyPnL[dateStr] || 0;
    const dailyReturn = pnl / initialCapital;
    
    dailyReturns.push(dailyReturn);
    
    // Move to next day
    currentDate.setDate(currentDate.getDate() + 1);
  }
  
  return dailyReturns;
};

// Helper function to calculate performance by trade duration
const calculatePerformanceByDuration = (entries) => {
  // Create duration categories
  const durationCategories = {};
  
  // Process each entry
  entries.forEach(entry => {
    // Skip entries without exit date (open trades)
    if (!entry.exitDate) return;
    
    // Calculate duration
    const duration = calculateTradeDuration(entry.entryDate, entry.exitDate);
    const category = categorizeTradeDuration(duration);
    
    // Initialize category if it doesn't exist
    if (!durationCategories[category]) {
      durationCategories[category] = {
        entries: [],
        wins: 0,
        losses: 0,
        totalPnl: 0
      };
    }
    
    // Add entry to category
    durationCategories[category].entries.push(entry);
    durationCategories[category].totalPnl += entry.pnl || 0;
    
    if (entry.outcome === 'Win') {
      durationCategories[category].wins++;
    } else if (entry.outcome === 'Loss') {
      durationCategories[category].losses++;
    }
  });
  
  // Calculate metrics for each category
  return Object.keys(durationCategories).map(category => {
    const categoryData = durationCategories[category];
    const count = categoryData.entries.length;
    const winRate = calculateWinRate(categoryData.wins, count);
    
    // Calculate expectancy for this category
    const winningEntries = categoryData.entries.filter(entry => entry.outcome === 'Win');
    const losingEntries = categoryData.entries.filter(entry => entry.outcome === 'Loss');
    
    const averageWin = winningEntries.length > 0 ?
      winningEntries.reduce((sum, entry) => sum + entry.pnl, 0) / winningEntries.length : 0;
    const averageLoss = losingEntries.length > 0 ?
      Math.abs(losingEntries.reduce((sum, entry) => sum + entry.pnl, 0)) / losingEntries.length : 0;
    
    const expectancy = calculateExpectancy(winRate, averageWin, averageLoss);
    const averagePnl = count > 0 ? categoryData.totalPnl / count : 0;
    
    return {
      durationCategory: category,
      count,
      winRate,
      expectancy,
      averagePnl
    };
  }).sort((a, b) => {
    // Sort by duration category in a logical order
    const order = ['0-15min', '15-30min', '30min-1h', '1h-2h', '2h-4h', '4h+', '1-2 Days', '3-6 Days', '1-2 Weeks', '2-4 Weeks', '1+ Month'];
    return order.indexOf(a.durationCategory) - order.indexOf(b.durationCategory);
  });
};