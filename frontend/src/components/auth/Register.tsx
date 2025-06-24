import React, { useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthContext from '../../context/auth/AuthContext';
import AlertContext from '../../context/alert/AlertContext';

const Register: React.FC = () => {
  const navigate = useNavigate();
  const authContext = useContext(AuthContext);
  const alertContext = useContext(AlertContext);

  const { register, error, clearErrors, isAuthenticated } = authContext;
  const { setAlert } = alertContext;

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }

    if (error) {
      setAlert(error, 'danger');
      clearErrors();
    }
  }, [error, isAuthenticated, navigate]);

  const [user, setUser] = useState({
    name: '',
    email: '',
    password: '',
    password2: '',
    initialCapital: 10000,
  });

  const { name, email, password, password2, initialCapital } = user;

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
    setUser({ ...user, [e.target.name]: value });
  };

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (name === '' || email === '' || password === '') {
      setAlert('Please enter all fields', 'danger');
    } else if (password !== password2) {
      setAlert('Passwords do not match', 'danger');
    } else if (initialCapital <= 0) {
      setAlert('Initial capital must be greater than 0', 'danger');
    } else {
      console.log('Registering with initial capital:', initialCapital); // Debug log
      register({
        name,
        email,
        password,
        password2,
        initialCapital
      });
    }
  };

  return (
    <div className="form-container">
      <h1>
        Account <span className="text-primary">Register</span>
      </h1>
      <form onSubmit={onSubmit}>
        <div className="form-group">
          <label htmlFor="name">Name</label>
          <input
            type="text"
            name="name"
            value={name}
            onChange={onChange}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="email">Email Address</label>
          <input
            type="email"
            name="email"
            value={email}
            onChange={onChange}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="initialCapital">Initial Capital ($)</label>
          <input
            type="number"
            name="initialCapital"
            value={initialCapital}
            onChange={onChange}
            required
            min="1"
            step="0.01"
            placeholder="10000"
          />
          <small className="form-text">Enter your starting trading capital</small>
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            name="password"
            value={password}
            onChange={onChange}
            required
            minLength={6}
          />
        </div>
        <div className="form-group">
          <label htmlFor="password2">Confirm Password</label>
          <input
            type="password"
            name="password2"
            value={password2}
            onChange={onChange}
            required
            minLength={6}
          />
        </div>
        <input
          type="submit"
          value="Register"
          className="btn btn-primary btn-block"
        />
      </form>
    </div>
  );
};

export default Register;