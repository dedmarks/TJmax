import React, { useReducer } from 'react';
import AuthContext from './AuthContext';
import authReducer from './authReducer';
import { register, login, getUser, updateProfile } from '../../services/authService';
import setAuthToken from '../../utils/setAuthToken';
import { AuthState as IAuthState, LoginFormData, RegisterFormData, User } from '../../types/User';

interface Props {
  children: React.ReactNode;
}

const AuthState: React.FC<Props> = ({ children }) => {
  const initialState: IAuthState = {
    token: localStorage.getItem('token'),
    isAuthenticated: false,
    loading: true,
    user: null,
    error: null,
  };

  const [state, dispatch] = useReducer(authReducer, initialState);

  // Load User
  // Load User
  const loadUser = async (): Promise<void> => {
    if (localStorage.token) {
      setAuthToken(localStorage.token);
    }

    try {
      const userData = await getUser();

      dispatch({
        type: 'USER_LOADED',
        payload: userData,
      });
    } catch (err: any) {
      dispatch({ 
        type: 'AUTH_ERROR',
        payload: err.response?.data.msg || 'Authentication error' 
      });
    }
  };

  // Register User
  // Register User
  const registerUser = async (formData: RegisterFormData): Promise<void> => {
    try {
      const res = await register(formData);
  
      // Set the token in localStorage and axios headers
      setAuthToken(res.token);
  
      dispatch({
        type: 'REGISTER_SUCCESS',
        payload: res,
      });
  
      await loadUser();
    } catch (err: any) {
      dispatch({
        type: 'REGISTER_FAIL',
        payload: err.response?.data.msg || 'Registration failed',
      });
    }
  };

  // Login User
  // Login User
  const loginUser = async (formData: LoginFormData): Promise<void> => {
    try {
      const res = await login(formData);
  
      // Set the token in localStorage and axios headers
      setAuthToken(res.token);
  
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: res,
      });
  
      await loadUser();
    } catch (err: any) {
      dispatch({
        type: 'LOGIN_FAIL',
        payload: err.response?.data.msg || 'Invalid credentials',
      });
    }
  };

  // Update Profile
  const updateUserProfile = async (formData: Partial<User>): Promise<void> => {
    try {
      const res = await updateProfile(formData);

      dispatch({
        type: 'UPDATE_PROFILE_SUCCESS',
        payload: res,
      });
    } catch (err: any) {
      dispatch({
        type: 'UPDATE_PROFILE_FAIL',
        payload: err.response?.data.msg || 'Failed to update profile',
      });
    }
  };

  // Logout
  const logout = (): void => {
    dispatch({ type: 'LOGOUT' });
  };

  // Clear Errors
  const clearErrors = (): void => {
    dispatch({ type: 'CLEAR_ERRORS' });
  };

  return (
    <AuthContext.Provider
      value={{
        token: state.token,
        isAuthenticated: state.isAuthenticated,
        loading: state.loading,
        user: state.user,
        error: state.error,
        register: registerUser,
        login: loginUser,
        loadUser,
        updateProfile: updateUserProfile,
        logout,
        clearErrors,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export default AuthState;