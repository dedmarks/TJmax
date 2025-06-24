import React, { useContext } from 'react';
import { Alert as MuiAlert, AlertTitle, Stack } from '@mui/material';
import AlertContext from '../../context/alert/AlertContext';

const Alert: React.FC = () => {
  const alertContext = useContext(AlertContext);
  const { alerts } = alertContext;

  if (alerts.length === 0) return null;

  return (
    <Stack spacing={2} sx={{ mb: 3 }}>
      {alerts.map((alert) => (
        <MuiAlert 
          key={alert.id} 
          severity={alert.type as 'error' | 'warning' | 'info' | 'success'}
          variant="outlined"
        >
          <AlertTitle>{alert.type.charAt(0).toUpperCase() + alert.type.slice(1)}</AlertTitle>
          {alert.msg}
        </MuiAlert>
      ))}
    </Stack>
  );
};

export default Alert;