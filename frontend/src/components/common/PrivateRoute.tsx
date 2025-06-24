import React, { useContext } from 'react';
import { Navigate } from 'react-router-dom';
import AuthContext from '../../context/auth/AuthContext';
import Spinner from './Spinner';

interface PrivateRouteProps {
  component: React.ComponentType;
}

const PrivateRoute: React.FC<PrivateRouteProps> = ({ component: Component }) => {
  const authContext = useContext(AuthContext);
  const { isAuthenticated, loading } = authContext;

  if (loading) return <Spinner />;
  if (!isAuthenticated) return <Navigate to="/login" />;
  
  return <Component />;
};

export default PrivateRoute;