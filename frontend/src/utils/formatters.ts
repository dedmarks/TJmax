import { format } from 'date-fns';

// Format date
export const formatDate = (date: string | Date): string => {
  return format(new Date(date), 'MMM dd, yyyy');
};

// Format time
export const formatTime = (date: string | Date): string => {
  return format(new Date(date), 'h:mm a');
};

// Format currency
export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
};

// Format percentage
export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

// Format number with 2 decimal places
export const formatNumber = (value: number): string => {
  return value.toFixed(2);
};