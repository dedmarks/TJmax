import { Alert, AlertState } from '../../types/Alert';

type AlertAction =
  | { type: 'SET_ALERT'; payload: Alert }
  | { type: 'REMOVE_ALERT'; payload: string };

const alertReducer = (state: AlertState, action: AlertAction): AlertState => {
  switch (action.type) {
    case 'SET_ALERT':
      return {
        ...state,
        alerts: [...state.alerts, action.payload],
      };
    case 'REMOVE_ALERT':
      return {
        ...state,
        alerts: state.alerts.filter((alert) => alert.id !== action.payload),
      };
    default:
      return state;
  }
};

export default alertReducer;