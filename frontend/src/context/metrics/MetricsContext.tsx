import { createContext } from 'react';
import { Metrics, MetricsFormData, MetricsState } from '../../types/Metrics';

interface MetricsContextInterface extends MetricsState {
  getMetrics: () => Promise<void>;
  getCurrent: (id: string) => Promise<void>;
  generateMetrics: (metricsData: MetricsFormData) => Promise<void>;
  clearCurrent: () => void;
  clearMetrics: () => void;
}

const MetricsContext = createContext<MetricsContextInterface>({
  metrics: [],
  current: null,
  loading: true,
  error: null,
  getMetrics: async () => {},
  getCurrent: async () => {},
  generateMetrics: async () => {},
  clearCurrent: () => {},
  clearMetrics: () => {},
});

export default MetricsContext;