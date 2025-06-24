import React, { useReducer } from 'react';
import { v4 as uuidv4 } from 'uuid';
import AlertContext from './AlertContext';
import alertReducer from './alertReducer';
import { Alert, AlertState as IAlertState } from '../../types/Alert';

interface Props {
  children: React.ReactNode;
}

const AlertState: React.FC<Props> = ({ children }) => {
  const initialState: IAlertState = {
    alerts: [],
  };

  const [state, dispatch] = useReducer(alertReducer, initialState);

  // Set Alert
  const setAlert = (msg: string, type: string, timeout = 5000): void => {
    const id = uuidv4();
    dispatch({
      type: 'SET_ALERT',
      payload: { msg, type, id },
    });

    setTimeout(() => dispatch({ type: 'REMOVE_ALERT', payload: id }), timeout);
  };

  return (
    <AlertContext.Provider
      value={{
        alerts: state.alerts,
        setAlert,
      }}
    >
      {children}
    </AlertContext.Provider>
  );
};

export default AlertState;