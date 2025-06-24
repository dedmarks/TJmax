import React, { useContext, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Alert from './components/layout/Alert';
import Dashboard from './components/dashboard/Dashboard';
import Register from './components/auth/Register';
import Login from './components/auth/Login';
import Profile from './components/auth/Profile';
import EntryForm from './components/journal/EntryForm';
import JournalList from './components/journal/JournalList';
import TradeView from './components/journal/TradeView';
import PrivateRoute from './components/common/PrivateRoute';

import AuthState from './context/auth/AuthState';
import JournalState from './context/journal/JournalState';
import MetricsState from './context/metrics/MetricsState';
import AlertState from './context/alert/AlertState';
import { ThemeProvider } from './context/theme/ThemeContext';
import AuthContext from './context/auth/AuthContext';

import setAuthToken from './utils/setAuthToken';

import './App.css';

if (localStorage.token) {
  setAuthToken(localStorage.token);
}

// Create a separate component to use context hooks
const AppContent: React.FC = () => {
  const authContext = useContext(AuthContext);

  useEffect(() => {
    if (localStorage.token) {
      authContext.loadUser();
    }
  }, []);

  return (
    <div className="app">
      <Navbar />
      <div className="container">
        <Alert />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login />} />
          <Route path="/profile" element={<PrivateRoute component={Profile} />} />
          <Route path="/journal" element={<JournalList />} />
          <Route path="/journal/new" element={<EntryForm />} />
          <Route path="/journal/:id" element={<EntryForm />} />
          <Route path="/journal/view/:id" element={<TradeView />} />
        </Routes>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AuthState>
        <JournalState>
          <MetricsState>
            <AlertState>
              <Router>
                <AppContent />
              </Router>
            </AlertState>
          </MetricsState>
        </JournalState>
      </AuthState>
    </ThemeProvider>
  );
};

export default App;
