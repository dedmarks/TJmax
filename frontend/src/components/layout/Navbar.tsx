import React, { useContext, useState, useEffect } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Box,
  Menu,
  MenuItem,
  Container,
  useMediaQuery,
  useTheme as useMuiTheme,
  Tooltip,
  Chip,
  Avatar,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import PersonIcon from '@mui/icons-material/Person';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ListAltIcon from '@mui/icons-material/ListAlt';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import LightModeIcon from '@mui/icons-material/LightMode';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import AuthContext from '../../context/auth/AuthContext';
import MetricsContext from '../../context/metrics/MetricsContext';
import { useTheme } from '../../context/theme/ThemeContext';
import { getAllTimeTotalProfit } from '../../services/journalService';

const Navbar: React.FC = () => {
  const authContext = useContext(AuthContext);
  const metricsContext = useContext(MetricsContext);
  const { isAuthenticated, user, logout } = authContext;
  const { current } = metricsContext;
  const muiTheme = useMuiTheme();
  const { mode, toggleColorMode } = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [allTimeTotalProfit, setAllTimeTotalProfit] = useState<number>(0);
  
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const onLogout = () => {
    logout();
    handleMenuClose();
  };

  // Fetch all-time total profit on component mount and when user changes
  useEffect(() => {
    if (isAuthenticated && user) {
      fetchAllTimeTotalProfit();
    }
  }, [isAuthenticated, user]);

  const fetchAllTimeTotalProfit = async () => {
    try {
      const { totalProfit } = await getAllTimeTotalProfit();
      setAllTimeTotalProfit(totalProfit);
    } catch (error) {
      console.error('Failed to fetch all-time total profit:', error);
    }
  };

  // Calculate current account balance using all-time data
  const getAccountBalance = () => {
    const initialCapital = user?.preferences?.initialCapital || 10000;
    return initialCapital + allTimeTotalProfit;
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const ThemeToggleButton = () => (
    <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}>
      <IconButton
        color="inherit"
        onClick={toggleColorMode}
        sx={{ ml: 1 }}
      >
        {mode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
      </IconButton>
    </Tooltip>
  );

  const AccountBalanceChip = () => (
    <Chip
      icon={<AccountBalanceWalletIcon />}
      label={formatCurrency(getAccountBalance())}
      variant="outlined"
      sx={{
        color: 'inherit',
        borderColor: 'rgba(255, 255, 255, 0.3)',
        '& .MuiChip-icon': {
          color: 'inherit'
        },
        mr: 1
      }}
    />
  );

  const ProfileButton = () => (
    <Tooltip title="Profile Settings">
      <IconButton
        color="inherit"
        component={RouterLink}
        to="/profile"
        sx={{ ml: 1 }}
      >
        <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
          <PersonIcon fontSize="small" />
        </Avatar>
      </IconButton>
    </Tooltip>
  );

  const authLinks = (
    <>
      {isMobile ? (
        <>
          <ThemeToggleButton />
          <IconButton
            edge="end"
            color="inherit"
            aria-label="menu"
            onClick={handleMenuOpen}
          >
            <MenuIcon />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem component={RouterLink} to="/dashboard" onClick={handleMenuClose}>
              <DashboardIcon sx={{ mr: 1 }} /> Dashboard
            </MenuItem>
            <MenuItem component={RouterLink} to="/journal" onClick={handleMenuClose}>
              <ListAltIcon sx={{ mr: 1 }} /> Journal
            </MenuItem>
            <MenuItem component={RouterLink} to="/bybit-import" onClick={handleMenuClose}>
              <CloudDownloadIcon sx={{ mr: 1 }} /> Bybit Import
            </MenuItem>
            <MenuItem component={RouterLink} to="/profile" onClick={handleMenuClose}>
              <PersonIcon sx={{ mr: 1 }} /> Profile
            </MenuItem>
            <MenuItem onClick={onLogout}>
              <ExitToAppIcon sx={{ mr: 1 }} /> Logout
            </MenuItem>
          </Menu>
        </>
      ) : (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/dashboard"
            startIcon={<DashboardIcon />}
            sx={{ mr: 1 }}
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/journal"
            startIcon={<ListAltIcon />}
            sx={{ mr: 1 }}
          >
            Journal
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/bybit-import"
            startIcon={<CloudDownloadIcon />}
            sx={{ mr: 1 }}
          >
            Bybit Import
          </Button>
          <Button
            color="inherit"
            onClick={onLogout}
            startIcon={<ExitToAppIcon />}
            sx={{ mr: 1 }}
          >
            Logout
          </Button>
          <AccountBalanceChip />
          <ProfileButton />
          <ThemeToggleButton />
        </Box>
      )}
    </>
  );

  const guestLinks = (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Button color="inherit" component={RouterLink} to="/register" sx={{ mr: 1 }}>
        Register
      </Button>
      <Button color="inherit" component={RouterLink} to="/login" sx={{ mr: 1 }}>
        Login
      </Button>
      <ThemeToggleButton />
    </Box>
  );

  return (
    <AppBar position="static">
      <Container maxWidth="lg">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            component={RouterLink}
            to="/"
            sx={{
              flexGrow: 1,
              textDecoration: 'none',
              color: 'inherit',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <AccountCircleIcon sx={{ mr: 1 }} />
            TJournal
          </Typography>
          {isAuthenticated ? authLinks : guestLinks}
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;