import axios from 'axios';
import { LoginFormData, RegisterFormData, User } from '../types/User';

const API_URL = 'http://localhost:5000/api';

// Register user
export const register = async (formData: RegisterFormData): Promise<{ token: string }> => {
  const response = await axios.post(`${API_URL}/users`, formData);
  return response.data;
};

// Login user
export const login = async (formData: LoginFormData): Promise<{ token: string }> => {
  const response = await axios.post(`${API_URL}/auth`, formData);
  return response.data;
};

// Get user data
export const getUser = async (): Promise<User> => {
  const response = await axios.get(`${API_URL}/auth`);
  return response.data;
};

// Update user profile
export const updateProfile = async (formData: Partial<User>): Promise<User> => {
  const response = await axios.put(`${API_URL}/users`, formData);
  return response.data;
};