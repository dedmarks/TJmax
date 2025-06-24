export interface PredefinedSetup {
  name: string;
  description?: string;
}

export interface UserPreferences {
  defaultRiskPercentage: number;
  defaultTimeframe: string;
  tradingHours: {
    start: string;
    end: string;
  };
  initialCapital: number;
  predefinedSetups?: PredefinedSetup[];
}

export interface User {
  _id: string;
  name: string;
  email: string;
  avatar?: string;
  preferences: UserPreferences;
  date: string;
}

export interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  user: User | null;
  error: string | null;
}

export interface LoginFormData {
  email: string;
  password: string;
}

export interface RegisterFormData {
  name: string;
  email: string;
  password: string;
  password2: string;
  initialCapital: number;
}