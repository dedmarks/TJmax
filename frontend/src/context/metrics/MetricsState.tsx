import React, { useReducer } from 'react';
import MetricsContext from './MetricsContext';
import metricsReducer from './metricsReducer';
import { getMetricsData, getMetricsById, generateMetricsData } from '../../services/metricsService';
import { Metrics, MetricsFormData, MetricsState as IMetricsState } from '../../types/Metrics';

interface Props {
  children: React.ReactNode;
}

const MetricsState: React.FC<Props> = ({ children }) => {
  const initialState: IMetricsState = {
    metrics: [],
    current: null,
    loading: true,
    error: null,
  };

  const [state, dispatch] = useReducer(metricsReducer, initialState);

  // Get all metrics
  const getMetrics = async (): Promise<void> => {
    try {
      const res = await getMetricsData();

      dispatch({
        type: 'GET_METRICS',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'METRICS_ERROR',
        payload: err.response?.data.msg || 'Failed to fetch metrics',
      });
    }
  };

  // Get metrics by ID
  const getCurrent = async (id: string): Promise<void> => {
    try {
      const res = await getMetricsById(id);

      dispatch({
        type: 'GET_CURRENT',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'METRICS_ERROR',
        payload: err.response?.data.msg || 'Failed to fetch metrics',
      });
    }
  };

  // Generate metrics
  const generateMetrics = async (metricsData: MetricsFormData): Promise<void> => {
    try {
      const res = await generateMetricsData(metricsData);

      dispatch({
        type: 'GENERATE_METRICS',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'METRICS_ERROR',
        payload: err.response?.data.msg || 'Failed to generate metrics',
      });
    }
  };

  // Clear current metrics
  const clearCurrent = (): void => {
    dispatch({ type: 'CLEAR_CURRENT' });
  };

  // Clear metrics
  const clearMetrics = (): void => {
    dispatch({ type: 'CLEAR_METRICS' });
  };

  return (
    <MetricsContext.Provider
      value={{
        metrics: state.metrics,
        current: state.current,
        loading: state.loading,
        error: state.error,
        getMetrics,
        getCurrent,
        generateMetrics,
        clearCurrent,
        clearMetrics,
      }}
    >
      {children}
    </MetricsContext.Provider>
  );
};

export default MetricsState;