import React, { useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import {
  TextField,
  Button,
  Typography,
  Container,
  Box,
  Paper,
} from '@mui/material';
import AuthContext from '../../context/auth/AuthContext';
import AlertContext from '../../context/alert/AlertContext';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const authContext = useContext(AuthContext);
  const alertContext = useContext(AlertContext);

  const { login, error, isAuthenticated, clearErrors } = authContext;
  const { setAlert } = alertContext;

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }

    if (error) {
      setAlert(error, 'error');
      clearErrors();
    }
  }, [error, isAuthenticated, clearErrors, setAlert, navigate]);

  const validationSchema = Yup.object({
    email: Yup.string().email('Invalid email address').required('Email is required'),
    password: Yup.string().required('Password is required'),
  });

  return (
    <Container maxWidth="sm">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Login
          </Typography>
          <Typography variant="body1" gutterBottom>
            Sign in to your account
          </Typography>
          <Formik
            initialValues={{
              email: '',
              password: '',
            }}
            validationSchema={validationSchema}
            onSubmit={(values, { setSubmitting }) => {
              login(values);
              setSubmitting(false);
            }}
          >
            {({ isSubmitting, errors, touched }) => (
              <Form>
                <Box sx={{ mb: 2 }}>
                  <Field
                    as={TextField}
                    fullWidth
                    id="email"
                    name="email"
                    label="Email"
                    variant="outlined"
                    error={touched.email && Boolean(errors.email)}
                    helperText={touched.email && errors.email}
                  />
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Field
                    as={TextField}
                    fullWidth
                    id="password"
                    name="password"
                    label="Password"
                    type="password"
                    variant="outlined"
                    error={touched.password && Boolean(errors.password)}
                    helperText={touched.password && errors.password}
                  />
                </Box>

                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={isSubmitting}
                  fullWidth
                  size="large"
                >
                  Login
                </Button>
              </Form>
            )}
          </Formik>
        </Paper>
      </Box>
    </Container>
  );
};

export default Login;