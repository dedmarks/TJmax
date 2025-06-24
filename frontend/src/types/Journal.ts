export interface Journal {
  _id: string;
  user: string;
  instrument: string;
  setup: string;
  direction: 'Long' | 'Short';
  entryDate: string;
  entryPrice: number;
  exitDate?: string;
  exitPrice?: number;
  stopLoss: number;
  targetPrice?: number; // Keep for backward compatibility
  takeProfitTargets?: TakeProfitTarget[]; // Add this new field
  positionSize: number;
  initialCapital?: number;
  riskAmount: number;
  riskPercentage: number;
  rewardToRisk?: number;
  outcome?: 'Win' | 'Loss' | 'Breakeven' | 'Open';
  pnl?: number;
  pnlPercentage?: number;
  fees?: number;
  notes?: string;
  images?: string[];
  timeframe: string;
  marketCondition?: 'Trending' | 'Ranging' | 'Volatile' | 'Quiet';
  psychologicalState?: {
    before?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    during?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    after?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    lessons?: string;
  };
  maxAdverseExcursion?: number;
  maxFavorableExcursion?: number;
  mistakes?: string[];
  tags?: string[];
  createdAt: string;
  updatedAt: string;
}

// Add this new interface
export interface TakeProfitTarget {
  price: number;
  percentage: number; // Percentage of position to exit at this target
  description?: string;
}

export interface JournalState {
  journals: Journal[];
  currentJournal: Journal | null;
  current: Journal | null; // Add this line
  filtered: Journal[] | null; // Add this line
  loading: boolean;
  error: string | null;
}

export interface JournalImage {
  url: string;
  description: string;
  category: 'chart' | 'setup' | 'entry' | 'exit' | 'other';
}

export interface JournalFormData {
  instrument: string;
  setup: string;
  direction: 'Long' | 'Short';
  entryDate: Date;
  entryPrice: number;
  exitDate?: Date;
  exitPrice?: number;
  stopLoss: number;
  targetPrice?: number; // Keep for backward compatibility
  takeProfitTargets?: TakeProfitTarget[]; // Add this new field
  positionSize: number;
  initialCapital?: number;
  riskAmount: number;
  riskPercentage: number;
  timeframe: string;
  marketCondition?: 'Trending' | 'Ranging' | 'Volatile' | 'Quiet';
  psychologicalState?: {
    before?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    during?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    after?: {
      state?: string;
      confidence?: number;
      stress?: number;
      focus?: number;
      triggers?: string[];
      notes?: string;
    };
    lessons?: string;
  };
  notes?: string;
  mistakes?: string[];
  tags?: string[];
  images?: JournalImage[];
}