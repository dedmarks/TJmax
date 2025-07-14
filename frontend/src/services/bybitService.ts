import axios from 'axios';
import { Journal, JournalFormData } from '../types/Journal';

const API_URL = 'http://localhost:5000/api/bybit';

// Get trade history from Bybit
export const getBybitTrades = async (params: {
  category: string;
  symbol?: string;
  limit?: number;
  startTime?: number;
  endTime?: number;
}) => {
  const response = await axios.get(`${API_URL}/trades`, { params });
  return response.data;
};

// Import trades from Bybit to journal
export const importBybitTrades = async (trades: any[]): Promise<{ msg: string }> => {
  const response = await axios.post(`${API_URL}/import`, { trades });
  return response.data;
};