import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { createTheme, ThemeProvider as MuiThemeProvider, PaletteMode } from '@mui/material';
import { CssBaseline } from '@mui/material';

interface ThemeContextType {
  mode: PaletteMode;
  toggleColorMode: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  mode: 'light',
  toggleColorMode: () => {},
});

export const useTheme = () => useContext(ThemeContext);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [mode, setMode] = useState<PaletteMode>(() => {
    const savedMode = localStorage.getItem('themeMode');
    return (savedMode as PaletteMode) || 'light';
  });

  const toggleColorMode = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setMode(newMode);
    localStorage.setItem('themeMode', newMode);
  };

  useEffect(() => {
    localStorage.setItem('themeMode', mode);
  }, [mode]);

  const theme = createTheme({
    palette: {
      mode,
      primary: {
        main: '#2563eb',
        light: '#60a5fa',
        dark: '#1d4ed8',
        contrastText: '#ffffff',
      },
      secondary: {
        main: '#7c3aed',
        light: '#a78bfa',
        dark: '#5b21b6',
      },
      error: {
        main: '#ef4444',
        light: '#f87171',
        dark: '#dc2626',
      },
      warning: {
        main: '#f59e0b',
        light: '#fbbf24',
        dark: '#d97706',
      },
      info: {
        main: '#06b6d4',
        light: '#22d3ee',
        dark: '#0891b2',
      },
      success: {
        main: '#10b981',
        light: '#34d399',
        dark: '#059669',
      },
      ...(mode === 'light'
        ? {
            // Light mode
            background: {
              default: '#f8fafc',
              paper: '#ffffff',
            },
            text: {
              primary: '#1f2937',
              secondary: '#6b7280',
            },
          }
        : {
            // Dark mode
            background: {
              default: '#0f172a',
              paper: '#1e293b',
            },
            text: {
              primary: '#f1f5f9',
              secondary: '#94a3b8',
            },
            divider: '#374151',
          }),
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 700,
        marginBottom: '1.5rem',
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 600,
        marginBottom: '1.25rem',
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
        marginBottom: '1rem',
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 600,
        marginBottom: '0.875rem',
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 600,
        marginBottom: '0.75rem',
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 600,
        marginBottom: '0.5rem',
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.5,
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            borderRadius: 8,
            padding: '10px 20px',
            fontWeight: 600,
            fontSize: '0.875rem',
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: mode === 'light' 
              ? '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
            border: mode === 'light' ? '1px solid #e5e7eb' : '1px solid #374151',
            '&:hover': {
              boxShadow: mode === 'light'
                ? '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                : '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
            },
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            marginBottom: 16,
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
              '& fieldset': {
                borderColor: mode === 'light' ? '#d1d5db' : '#4b5563',
              },
              '&:hover fieldset': {
                borderColor: mode === 'light' ? '#9ca3af' : '#6b7280',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#2563eb',
                borderWidth: 2,
              },
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: mode === 'light'
              ? '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: mode === 'light'
              ? '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
            borderBottom: mode === 'light' ? '1px solid #e5e7eb' : '1px solid #374151',
          },
        },
      },
    },
  });

  return (
    <ThemeContext.Provider value={{ mode, toggleColorMode }}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeContext;