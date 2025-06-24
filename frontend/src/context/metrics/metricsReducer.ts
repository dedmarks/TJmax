import { Metrics, MetricsState } from '../../types/Metrics';

type MetricsAction =
  | { type: 'GET_METRICS'; payload: Metrics[] }
  | { type: 'GET_CURRENT'; payload: Metrics }
  | { type: 'GENERATE_METRICS'; payload: Metrics }
  | { type: 'CLEAR_CURRENT' }
  | { type: 'METRICS_ERROR'; payload: string }
  | { type: 'CLEAR_METRICS' };

const metricsReducer = (state: MetricsState, action: MetricsAction): MetricsState => {
  switch (action.type) {
    case 'GET_METRICS':
      return {
        ...state,
        metrics: action.payload,
        loading: false,
      };
    case 'GET_CURRENT':
      return {
        ...state,
        current: action.payload,
        loading: false,
      };
    case 'GENERATE_METRICS':
      return {
        ...state,
        metrics: [action.payload, ...state.metrics],
        current: action.payload,
        loading: false,
      };
    case 'CLEAR_CURRENT':
      return {
        ...state,
        current: null,
      };
    case 'METRICS_ERROR':
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
    case 'CLEAR_METRICS':
      return {
        ...state,
        metrics: [],
        current: null,
        error: null,
      };
    default:
      return state;
  }
};

export default metricsReducer;