import axios from 'axios';
import { Metrics, MetricsFormData } from '../types/Metrics';

const API_URL = '/api/metrics';

// Get all metrics
export const getMetricsData = async (): Promise<Metrics[]> => {
  const response = await axios.get(API_URL);
  return response.data;
};

// Get metrics by ID
export const getMetricsById = async (id: string): Promise<Metrics> => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

// Generate metrics
export const generateMetricsData = async (metricsData: MetricsFormData): Promise<Metrics> => {
  const response = await axios.post(API_URL, metricsData);
  return response.data;
};