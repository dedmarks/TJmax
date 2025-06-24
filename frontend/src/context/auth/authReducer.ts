import { AuthState } from '../../types/User';

type AuthAction =
  | { type: 'REGISTER_SUCCESS'; payload: { token: string } }
  | { type: 'REGISTER_FAIL'; payload: string }
  | { type: 'USER_LOADED'; payload: any }
  | { type: 'AUTH_ERROR'; payload?: string } // Update this line to make payload optional
  | { type: 'LOGIN_SUCCESS'; payload: { token: string } }
  | { type: 'LOGIN_FAIL'; payload: string }
  | { type: 'LOGOUT' }
  | { type: 'CLEAR_ERRORS' }
  | { type: 'UPDATE_PROFILE_SUCCESS'; payload: any }
  | { type: 'UPDATE_PROFILE_FAIL'; payload: string };

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case 'USER_LOADED':
      return {
        ...state,
        isAuthenticated: true,
        loading: false,
        user: action.payload,
      };
    case 'REGISTER_SUCCESS':
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        token: action.payload.token,
        isAuthenticated: true,
        loading: false,
      };
    case 'REGISTER_FAIL':
    case 'AUTH_ERROR':
    case 'LOGIN_FAIL':
    case 'LOGOUT':
      return {
        ...state,
        token: null,
        isAuthenticated: false,
        loading: false,
        user: null,
        error: action.type === 'LOGOUT' ? null : (action.payload || null),
      };
    case 'UPDATE_PROFILE_SUCCESS':
      return {
        ...state,
        user: action.payload,
        loading: false,
      };
    case 'UPDATE_PROFILE_FAIL':
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
    case 'CLEAR_ERRORS':
      return {
        ...state,
        error: null,
      };
    default:
      return state;
  }
};

export default authReducer;