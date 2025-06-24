import { createContext } from 'react';
import { AlertState } from '../../types/Alert';

interface AlertContextInterface extends AlertState {
  setAlert: (msg: string, type: string, timeout?: number) => void;
}

const AlertContext = createContext<AlertContextInterface>({
  alerts: [],
  setAlert: () => {},
});

export default AlertContext;