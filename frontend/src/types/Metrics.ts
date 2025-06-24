export interface EquityPoint {
  date: string;
  equity: number;
}

export interface SetupPerformance {
  setup: string;
  count: number;
  winRate: number;
  expectancy: number;
}

export interface TimeframePerformance {
  timeframe: string;
  count: number;
  winRate: number;
  expectancy: number;
}

export interface InstrumentPerformance {
  instrument: string;
  count: number;
  winRate: number;
  expectancy: number;
}

export interface DurationPerformance {
  durationCategory: string;
  count: number;
  winRate: number;
  expectancy: number;
  averagePnl: number;
}

export interface Metrics {
  _id: string;
  user: string;
  period: 'Daily' | 'Weekly' | 'Monthly' | 'Quarterly' | 'Yearly' | 'Custom';
  startDate: string;
  endDate: string;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  breakEvenTrades: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  expectancy: number;
  systemQualityNumber: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownPercentage: number;
  averageRPerTrade: number;
  netProfit: number;
  netProfitPercentage: number;
  equityCurve: EquityPoint[];
  setupPerformance: SetupPerformance[];
  timeframePerformance: TimeframePerformance[];
  instrumentPerformance: InstrumentPerformance[];
  createdAt: string;
  updatedAt: string;
  durationPerformance: DurationPerformance[];
}

export interface MetricsState {
  metrics: Metrics[];
  current: Metrics | null;
  loading: boolean;
  error: string | null;
}

export interface MetricsFormData {
  period: 'Daily' | 'Weekly' | 'Monthly' | 'Quarterly' | 'Yearly' | 'Custom';
  startDate: Date;
  endDate: Date;
}