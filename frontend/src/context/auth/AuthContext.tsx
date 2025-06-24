import { createContext } from 'react';
import { AuthState, LoginFormData, RegisterFormData, User } from '../../types/User';

interface AuthContextInterface extends AuthState {
  register: (formData: RegisterFormData) => Promise<void>;
  login: (formData: LoginFormData) => Promise<void>;
  loadUser: () => Promise<void>;
  updateProfile: (formData: Partial<User>) => Promise<void>;
  logout: () => void;
  clearErrors: () => void;
}

const AuthContext = createContext<AuthContextInterface>({
  token: localStorage.getItem('token'),
  isAuthenticated: false,
  loading: true,
  user: null,
  error: null,
  register: async () => {},
  login: async () => {},
  loadUser: async () => {},
  updateProfile: async () => {},
  logout: () => {},
  clearErrors: () => {},
});

export default AuthContext;