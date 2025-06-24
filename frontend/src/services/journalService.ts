import axios from 'axios';
import { Journal, JournalFormData } from '../types/Journal';

const API_URL = 'http://localhost:5000/api/journal';

// Get all journal entries
export const getJournalEntries = async (): Promise<Journal[]> => {
  const response = await axios.get(API_URL);
  return response.data;
};

// Get journal entry by ID
export const getJournalEntryById = async (id: string): Promise<Journal> => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

// Create new journal entry
export const createJournalEntry = async (formData: JournalFormData): Promise<Journal> => {
  const response = await axios.post(API_URL, formData);
  return response.data;
};

// Update journal entry
export const updateJournalEntry = async (id: string, formData: Partial<JournalFormData>): Promise<Journal> => {
  const response = await axios.put(`${API_URL}/${id}`, formData);
  return response.data;
};

// Delete journal entry
export const deleteJournalEntry = async (id: string): Promise<void> => {
  await axios.delete(`${API_URL}/${id}`);
};

export const getAllTimeTotalProfit = async (): Promise<{ totalProfit: number }> => {
  const response = await axios.get(`${API_URL}/total-profit`);
  return response.data;
};
