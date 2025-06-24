import React, { useState, useContext, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Divider,
  Alert,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper
} from '@mui/material';
import {
  Person,
  AccountBalance,
  TrendingUp,
  Schedule,
  Save,
  Add,
  Delete,
  Edit
} from '@mui/icons-material';
import AuthContext from '../../context/auth/AuthContext';
import AlertContext from '../../context/alert/AlertContext';
import { PredefinedSetup } from '../../types/User';

const Profile: React.FC = () => {
  const authContext = useContext(AuthContext);
  const alertContext = useContext(AlertContext);
  const { user, updateProfile } = authContext;
  const { setAlert } = alertContext;

  const [formData, setFormData] = useState({
    name: '',
    initialCapital: 10000,
    defaultRiskPercentage: 1.0,
    defaultTimeframe: 'Daily',
    tradingHoursStart: '08:00',
    tradingHoursEnd: '16:00'
  });

  const [predefinedSetups, setPredefinedSetups] = useState<PredefinedSetup[]>([]);
  const [loading, setLoading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [currentSetup, setCurrentSetup] = useState<PredefinedSetup>({ name: '', description: '' });
  const [editIndex, setEditIndex] = useState<number | null>(null);

  useEffect(() => {
    if (user) {
      setFormData({
        name: user.name || '',
        initialCapital: user.preferences?.initialCapital || 10000,
        defaultRiskPercentage: user.preferences?.defaultRiskPercentage || 1.0,
        defaultTimeframe: user.preferences?.defaultTimeframe || 'Daily',
        tradingHoursStart: user.preferences?.tradingHours?.start || '08:00',
        tradingHoursEnd: user.preferences?.tradingHours?.end || '16:00'
      });
      setPredefinedSetups(user.preferences?.predefinedSetups || []);
    }
  }, [user]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'initialCapital' || name === 'defaultRiskPercentage' 
        ? parseFloat(value) || 0 
        : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      console.log('Saving predefined setups:', predefinedSetups); // Add this for debugging
      await updateProfile({
        name: formData.name,
        preferences: {
          initialCapital: formData.initialCapital,
          defaultRiskPercentage: formData.defaultRiskPercentage,
          defaultTimeframe: formData.defaultTimeframe,
          tradingHours: {
            start: formData.tradingHoursStart,
            end: formData.tradingHoursEnd
          },
          predefinedSetups: predefinedSetups
        }
      });
      setAlert('Profile updated successfully', 'success');
    } catch (error) {
      console.error('Profile update error:', error); // Add this for debugging
      setAlert('Failed to update profile', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (setup?: PredefinedSetup, index?: number) => {
    if (setup && index !== undefined) {
      setCurrentSetup(setup);
      setEditIndex(index);
    } else {
      setCurrentSetup({ name: '', description: '' });
      setEditIndex(null);
    }
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setCurrentSetup({ name: '', description: '' });
    setEditIndex(null);
  };

  const handleSetupChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setCurrentSetup(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveSetup = () => {
    if (!currentSetup.name.trim()) {
      setAlert('Setup name is required', 'error');
      return;
    }

    if (editIndex !== null) {
      // Edit existing setup
      const updatedSetups = [...predefinedSetups];
      updatedSetups[editIndex] = currentSetup;
      setPredefinedSetups(updatedSetups);
    } else {
      // Add new setup
      setPredefinedSetups([...predefinedSetups, currentSetup]);
    }

    handleCloseDialog();
  };

  const handleDeleteSetup = (index: number) => {
    const updatedSetups = predefinedSetups.filter((_, i) => i !== index);
    setPredefinedSetups(updatedSetups);
  };

  if (!user) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Person />
        Profile Settings
      </Typography>
      
      <Card>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              {/* Personal Information */}
              <Grid size={{xs:12}}>
                <Typography variant="h6" gutterBottom>
                  Personal Information
                </Typography>
                <Divider sx={{ mb: 2 }} />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Email"
                  value={user.email}
                  disabled
                  helperText="Email cannot be changed"
                />
              </Grid>

              {/* Trading Preferences */}
              <Grid size={{xs:12}}>
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Trading Preferences
                </Typography>
                <Divider sx={{ mb: 2 }} />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Initial Capital"
                  name="initialCapital"
                  type="number"
                  value={formData.initialCapital}
                  onChange={handleChange}
                  InputProps={{
                    startAdornment: <InputAdornment position="start"><AccountBalance /></InputAdornment>,
                    inputProps: { min: 0, step: 0.2 }
                  }}
                  helperText="Your starting trading capital for performance calculations"
                  required
                />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Default Risk Percentage"
                  name="defaultRiskPercentage"
                  type="number"
                  value={formData.defaultRiskPercentage}
                  onChange={handleChange}
                  InputProps={{
                    startAdornment: <InputAdornment position="start"><TrendingUp /></InputAdornment>,
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                    inputProps: { min: 0.1, max: 10, step: 0.1 }
                  }}
                  helperText="Default risk percentage per trade"
                  required
                />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <FormControl fullWidth>
                  <InputLabel>Default Timeframe</InputLabel>
                  <Select
                    name="defaultTimeframe"
                    value={formData.defaultTimeframe}
                    onChange={(e) => handleChange(e as any)}
                    label="Default Timeframe"
                  >
                    <MenuItem value="Scalping">Scalping</MenuItem>
                    <MenuItem value="Intraday">Intraday</MenuItem>
                    <MenuItem value="Daily">Daily</MenuItem>
                    <MenuItem value="Swing">Swing</MenuItem>
                    <MenuItem value="Position">Position</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              {/* Trading Hours */}
              <Grid size={{xs:12}}>
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Trading Hours
                </Typography>
                <Divider sx={{ mb: 2 }} />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Trading Start Time"
                  name="tradingHoursStart"
                  type="time"
                  value={formData.tradingHoursStart}
                  onChange={handleChange}
                  InputProps={{
                    startAdornment: <InputAdornment position="start"><Schedule /></InputAdornment>
                  }}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              
              <Grid size={{xs:12, md:6}}>
                <TextField
                  fullWidth
                  label="Trading End Time"
                  name="tradingHoursEnd"
                  type="time"
                  value={formData.tradingHoursEnd}
                  onChange={handleChange}
                  InputProps={{
                    startAdornment: <InputAdornment position="start"><Schedule /></InputAdornment>
                  }}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>

              {/* Predefined Setups */}
              <Grid size={{xs:12}}>
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Predefined Trade Setups
                </Typography>
                <Divider sx={{ mb: 2 }} />
              </Grid>
              
              <Grid size={{xs:12}}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="body2">
                    Define your common trade setups to quickly select them when adding new trades
                  </Typography>
                  <Button 
                    variant="contained" 
                    startIcon={<Add />} 
                    size="small"
                    onClick={() => handleOpenDialog()}
                  >
                    Add Setup
                  </Button>
                </Box>
                
                {predefinedSetups.length > 0 ? (
                  <Paper variant="outlined" sx={{ mb: 3 }}>
                    <List>
                      {predefinedSetups.map((setup, index) => (
                        <ListItem
                          key={index}
                          secondaryAction={
                            <Box>
                              <IconButton edge="end" onClick={() => handleOpenDialog(setup, index)}>
                                <Edit />
                              </IconButton>
                              <IconButton edge="end" onClick={() => handleDeleteSetup(index)}>
                                <Delete />
                              </IconButton>
                            </Box>
                          }
                        >
                          <ListItemText 
                            primary={setup.name} 
                            secondary={setup.description || 'No description'} 
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                ) : (
                  <Alert severity="info" sx={{ mb: 3 }}>
                    You haven't defined any trade setups yet. Click "Add Setup" to create your first one.
                  </Alert>
                )}
              </Grid>

              <Grid size={{xs:12}}>
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={loading}
                    startIcon={<Save />}
                    sx={{ minWidth: 150 }}
                  >
                    {loading ? 'Saving...' : 'Save Changes'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </form>
        </CardContent>
      </Card>
      
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Note:</strong> Changes to your initial capital will affect future performance calculations. 
          Historical metrics will be recalculated based on the new initial capital value.
        </Typography>
      </Alert>

      {/* Setup Dialog */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{editIndex !== null ? 'Edit Setup' : 'Add New Setup'}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              fullWidth
              label="Setup Name"
              name="name"
              value={currentSetup.name}
              onChange={handleSetupChange}
              required
              margin="normal"
              placeholder="e.g., Breakout, Support/Resistance, VWAP Bounce"
            />
            <TextField
              fullWidth
              label="Description (Optional)"
              name="description"
              value={currentSetup.description || ''}
              onChange={handleSetupChange}
              margin="normal"
              multiline
              rows={3}
              placeholder="Describe your setup strategy, entry conditions, etc."
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSaveSetup} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Profile;