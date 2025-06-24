import React, { useState, useContext, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Formik, Form, Field, ErrorMessage, FormikHelpers } from 'formik';
import * as Yup from 'yup';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Grid,
  Stepper,
  Step,
  StepLabel,
  Chip,
  InputAdornment,
  IconButton,
  Tooltip,
  Alert,
  LinearProgress,
  Fade,
  Paper,
  Divider,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Calculate,
  Psychology,
  Notes,
  Save,
  Cancel,
  AutoFixHigh,
  Timeline,
  AttachMoney,
  Speed,
  Info,
  Image,
  Add,
  Remove
} from '@mui/icons-material';
import JournalContext from '../../context/journal/JournalContext';
import AlertContext from '../../context/alert/AlertContext';
import AuthContext from '../../context/auth/AuthContext';
import { JournalFormData, JournalImage, TakeProfitTarget } from '../../types/Journal';
import { uploadImageFromBase64 } from '../../services/uploadService';


const steps = [
  { label: 'Trade Setup', icon: <Timeline /> },
  { label: 'Risk Management', icon: <AttachMoney /> },
  { label: 'Notes & Images', icon: <Notes /> }
];

const EntryForm: React.FC = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const journalContext = useContext(JournalContext);
  const alertContext = useContext(AlertContext);
  const authContext = useContext(AuthContext);

  const { addJournal, updateJournal, current, getCurrent, clearCurrent, error } = journalContext;
  const { setAlert } = alertContext;
  const { user } = authContext;

  const [isEditing, setIsEditing] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [autoCalculate, setAutoCalculate] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (id) {
      setIsEditing(true);
      getCurrent(id);
    } else {
      clearCurrent();
      setIsEditing(false);
    }

    return () => {
      clearCurrent();
    };
  }, [id]);

  useEffect(() => {
    if (error) {
      setAlert(error, 'error');
    }
  }, [error, setAlert]);

  const initialValues: JournalFormData = {
    instrument: current?.instrument || '',
    setup: current?.setup || '',
    direction: current?.direction || 'Long',
    entryDate: current?.entryDate ? new Date(current.entryDate) : new Date(),
    entryPrice: current?.entryPrice || 0,
    exitDate: current?.exitDate ? new Date(current.exitDate) : undefined,
    exitPrice: current?.exitPrice || undefined,
    stopLoss: current?.stopLoss || 0,
    targetPrice: current?.targetPrice || undefined,
    takeProfitTargets: current?.takeProfitTargets || [],
    positionSize: current?.positionSize || 0,
    riskAmount: current?.riskAmount || 0,
    riskPercentage: current?.riskPercentage || (user?.preferences.defaultRiskPercentage || 1),
    timeframe: current?.timeframe || (user?.preferences.defaultTimeframe || 'Daily'),
    marketCondition: current?.marketCondition || undefined,
    psychologicalState: current?.psychologicalState || {},
    notes: current?.notes || '',
    mistakes: current?.mistakes || [],
    tags: current?.tags || [],
    images: Array.isArray(current?.images) 
      ? current?.images.map((img: string | JournalImage) => {
          if (typeof img === 'string') {
            return {
              url: img,
              description: '',
              category: 'chart' as const
            };
          }
          return img as JournalImage;
        }) 
      : [],
  };

  const validationSchema = Yup.object({
    instrument: Yup.string().required('Instrument is required'),
    setup: Yup.string().required('Setup is required'),
    direction: Yup.string().required('Direction is required'),
    entryDate: Yup.date().required('Entry date is required'),
    entryPrice: Yup.number().required('Entry price is required').positive('Entry price must be positive'),
    stopLoss: Yup.number().required('Stop loss is required'),
    positionSize: Yup.number().required('Position size is required').positive('Position size must be positive'),
    riskAmount: Yup.number().required('Risk amount is required').positive('Risk amount must be positive'),
    riskPercentage: Yup.number().required('Risk percentage is required').positive('Risk percentage must be positive'),
    timeframe: Yup.string().required('Timeframe is required'),
  });

  const onSubmit = async (values: JournalFormData, { setSubmitting }: FormikHelpers<JournalFormData>) => {
    try {
      if (isEditing && id) {
        await updateJournal(id, values);
        setAlert('Journal entry updated successfully', 'success');
      } else {
        await addJournal(values);
        setAlert('Journal entry added successfully', 'success');
      }
      navigate('/journal');
    } catch (err) {
      setAlert('Failed to save journal entry', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const handlePaste = async (e: React.ClipboardEvent, setFieldValue: any, currentImages: JournalImage[]) => {
    const items = e.clipboardData.items;
    
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        e.preventDefault();
        
        const blob = items[i].getAsFile();
        if (!blob) continue;
        
        try {
          setIsUploading(true);
          
          // Convert blob to base64
          const reader = new FileReader();
          reader.onload = async (event) => {
            if (!event.target || typeof event.target.result !== 'string') return;
            
            try {
              const base64Data = event.target.result;
              const uploadedImage = await uploadImageFromBase64(base64Data);
              
              // Create a new journal image
              const newImage: JournalImage = {
                url: uploadedImage.url,
                description: '',
                category: 'chart'
              };
              
              // Update form values with the new image
              const updatedImages = [...(currentImages || []), newImage];
              setFieldValue('images', updatedImages);
              
              setAlert('Image uploaded successfully', 'success');
            } catch (error) {
              setAlert('Failed to upload image', 'error');
              console.error('Image upload error:', error);
            } finally {
              setIsUploading(false);
            }
          };
          
          reader.readAsDataURL(blob);
        } catch (error) {
          setIsUploading(false);
          setAlert('Failed to process image', 'error');
          console.error('Image processing error:', error);
        }
      }
    }
  };
  
  // New function to handle file selection
  const handleFileSelect = async (files: FileList | null, setFieldValue: any, currentImages: JournalImage[]) => {
    if (!files || files.length === 0) return;
    
    try {
      setIsUploading(true);
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Check if file is an image
        if (!file.type.startsWith('image/')) {
          setAlert(`File "${file.name}" is not an image`, 'warning');
          continue;
        }
        
        // Convert file to base64
        const reader = new FileReader();
        reader.onload = async (event) => {
          if (!event.target || typeof event.target.result !== 'string') return;
          
          try {
            const base64Data = event.target.result;
            const uploadedImage = await uploadImageFromBase64(base64Data);
            
            // Create a new journal image
            const newImage: JournalImage = {
              url: uploadedImage.url,
              description: '',
              category: 'chart'
            };
            
            // Update form values with the new image
            const updatedImages = [...(currentImages || []), newImage];
            setFieldValue('images', updatedImages);
            
            setAlert(`Image "${file.name}" uploaded successfully`, 'success');
          } catch (error) {
            setAlert(`Failed to upload image "${file.name}"`, 'error');
            console.error('Image upload error:', error);
          } finally {
            if (i === files.length - 1) {
              setIsUploading(false);
            }
          }
        };
        
        reader.readAsDataURL(file);
      }
    } catch (error) {
      setIsUploading(false);
      setAlert('Failed to process images', 'error');
      console.error('Image processing error:', error);
    }
  };
  
  // New function to handle drag and drop
  const handleDrop = (e: React.DragEvent, setFieldValue: any, currentImages: JournalImage[]) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    handleFileSelect(files, setFieldValue, currentImages);
  };
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleRemoveImage = (index: number, images: JournalImage[], setFieldValue: any) => {
    const updatedImages = images.filter((_, i) => i !== index);
    setFieldValue('images', updatedImages);
  };

  const calculateRiskAmount = (entryPrice: number, stopLoss: number, positionSize: number, direction: string) => {
    if (entryPrice <= 0 || positionSize <= 0) return 0;
    
    if (direction === 'Long') {
      return stopLoss > 0 ? (entryPrice - stopLoss) * positionSize : 0;
    } else {
      return stopLoss > 0 ? (stopLoss - entryPrice) * positionSize : 0;
    }
  };

  const calculateRiskPercentage = (riskAmount: number) => {
    const accountSize = user?.preferences?.initialCapital || 100000;
    return (riskAmount / accountSize) * 100;
  };

  const calculateRewardToRisk = (entryPrice: number, targetPrice: number | undefined, stopLoss: number, direction: string, takeProfitTargets?: TakeProfitTarget[]) => {
    // If we have take profit targets, calculate weighted average reward-to-risk
    if (takeProfitTargets && takeProfitTargets.length > 0) {
      let totalRewardToRisk = 0;
      let totalPercentage = 0;
      
      takeProfitTargets.forEach(target => {
        const targetRewardToRisk = calculateSingleRewardToRisk(entryPrice, target.price, stopLoss, direction);
        totalRewardToRisk += targetRewardToRisk * (target.percentage / 100);
        totalPercentage += target.percentage / 100;
      });
      
      // If total percentage is less than 1, use the single target for the remaining percentage
      if (totalPercentage < 1 && targetPrice) {
        const remainingPercentage = 1 - totalPercentage;
        const singleTargetRewardToRisk = calculateSingleRewardToRisk(entryPrice, targetPrice, stopLoss, direction);
        totalRewardToRisk += singleTargetRewardToRisk * remainingPercentage;
      }
      
      return totalRewardToRisk;
    }
    
    // Fall back to single target calculation
    return calculateSingleRewardToRisk(entryPrice, targetPrice, stopLoss, direction);
  };
  
  // Helper function for single target reward-to-risk calculation
  const calculateSingleRewardToRisk = (entryPrice: number, targetPrice: number | undefined, stopLoss: number, direction: string) => {
    if (!targetPrice || entryPrice <= 0 || stopLoss <= 0) return 0;
    
    if (direction === 'Long') {
      const risk = entryPrice - stopLoss;
      const reward = targetPrice - entryPrice;
      return risk > 0 ? reward / risk : 0;
    } else {
      const risk = stopLoss - entryPrice;
      const reward = entryPrice - targetPrice;
      return risk > 0 ? reward / risk : 0;
    }
  };

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  // In the renderStepContent function, add ImageUpload to the third step (Psychology & Notes)
  const renderStepContent = (step: number, values: any, setFieldValue: any) => {
    switch (step) {
      case 0:
        return (
          <Fade in timeout={500}>
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <Timeline color="primary" />
                Trade Setup
              </Typography>
              <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Instrument"
                    name="instrument"
                    value={values.instrument}
                    onChange={(e) => setFieldValue('instrument', e.target.value)}
                    variant="outlined"
                    required
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <Timeline />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                

<Grid size={{ xs: 12, md: 6 }}>
  {user?.preferences.predefinedSetups && user.preferences.predefinedSetups.length > 0 ? (
    <FormControl fullWidth variant="outlined" required>
      <InputLabel>Setup</InputLabel>
      <Select
        label="Setup"
        name="setup"
        value={values.setup}
        onChange={(e) => setFieldValue('setup', e.target.value)}
      >
        {user.preferences.predefinedSetups.map((setup, index) => (
          <MenuItem key={index} value={setup.name}>
            <Box>
              <Typography variant="body1">{setup.name}</Typography>
              {setup.description && (
                <Typography variant="caption" color="textSecondary">
                  {setup.description}
                </Typography>
              )}
            </Box>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  ) : (
    <TextField
      fullWidth
      label="Setup"
      name="setup"
      value={values.setup}
      onChange={(e) => setFieldValue('setup', e.target.value)}
      variant="outlined"
      required
      placeholder="e.g., Breakout, Support/Resistance"
    />
  )}
</Grid>

                
                <Grid size={{ xs: 12, md: 6 }}>
                  <FormControl fullWidth>
                    <InputLabel>Direction</InputLabel>
                    <Select
                      value={values.direction}
                      label="Direction"
                      onChange={(e) => setFieldValue('direction', e.target.value)}
                      startAdornment={
                        <InputAdornment position="start">
                          {values.direction === 'Long' ? <TrendingUp color="success" /> : <TrendingDown color="error" />}
                        </InputAdornment>
                      }
                    >
                      <MenuItem value="Long">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingUp color="success" /> Long
                        </Box>
                      </MenuItem>
                      <MenuItem value="Short">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingDown color="error" /> Short
                        </Box>
                      </MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <FormControl fullWidth>
                    <InputLabel>Timeframe</InputLabel>
                    <Select
                      value={values.timeframe}
                      label="Timeframe"
                      onChange={(e) => setFieldValue('timeframe', e.target.value)}
                    >
                      <MenuItem value="1min">1 Minute</MenuItem>
                      <MenuItem value="5min">5 Minutes</MenuItem>
                      <MenuItem value="15min">15 Minutes</MenuItem>
                      <MenuItem value="30min">30 Minutes</MenuItem>
                      <MenuItem value="1hour">1 Hour</MenuItem>
                      <MenuItem value="4hour">4 Hours</MenuItem>
                      <MenuItem value="Daily">Daily</MenuItem>
                      <MenuItem value="Weekly">Weekly</MenuItem>
                      <MenuItem value="Monthly">Monthly</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Entry Date"
                    name="entryDate"
                    type="datetime-local"
                    value={values.entryDate ? new Date(values.entryDate).toISOString().slice(0, 16) : ''}
                    onChange={(e) => setFieldValue('entryDate', new Date(e.target.value))}
                    variant="outlined"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Entry Price"
                    name="entryPrice"
                    type="number"
                    value={values.entryPrice || ''}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value) || 0;
                      setFieldValue('entryPrice', value);
                      
                      if (autoCalculate) {
                        const riskAmount = calculateRiskAmount(value, values.stopLoss, values.positionSize, values.direction);
                        setFieldValue('riskAmount', riskAmount);
                        setFieldValue('riskPercentage', calculateRiskPercentage(riskAmount));
                        
                        if (values.targetPrice) {
                          setFieldValue('rewardToRisk', calculateRewardToRisk(value, values.targetPrice, values.stopLoss, values.direction));
                        }
                      }
                    }}
                    variant="outlined"
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                    inputProps={{ step: 0.01 }}
                  />
                </Grid>
                
                {showAdvanced && (
                  <>
                    <Grid size={{ xs: 12, md: 6 }}>
                      <TextField
                        fullWidth
                        label="Exit Date"
                        name="exitDate"
                        type="datetime-local"
                        value={values.exitDate ? new Date(values.exitDate).toISOString().slice(0, 16) : ''}
                        onChange={(e) => setFieldValue('exitDate', e.target.value ? new Date(e.target.value) : undefined)}
                        variant="outlined"
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    
                    <Grid size={{ xs: 12, md: 6 }}>
                      <TextField
                        fullWidth
                        label="Exit Price"
                        name="exitPrice"
                        type="number"
                        value={values.exitPrice || ''}
                        onChange={(e) => setFieldValue('exitPrice', parseFloat(e.target.value) || undefined)}
                        variant="outlined"
                        InputProps={{
                          startAdornment: <InputAdornment position="start">$</InputAdornment>,
                        }}
                        inputProps={{ step: 0.01 }}
                      />
                    </Grid>
                  </>
                )}
                
                <Grid size={{ xs: 12 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={showAdvanced}
                        onChange={(e) => setShowAdvanced(e.target.checked)}
                        color="primary"
                      />
                    }
                    label="Show Advanced Fields"
                  />
                </Grid>
              </Grid>
            </Box>
          </Fade>
        );
        
      case 1:
        return (
          <Fade in timeout={500}>
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <AttachMoney color="primary" />
                Risk Management
              </Typography>
              
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  Auto-calculation is {autoCalculate ? 'enabled' : 'disabled'}. 
                  {autoCalculate ? 'Risk metrics will update automatically.' : 'You can manually adjust risk values.'}
                </Typography>
              </Alert>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={autoCalculate}
                    onChange={(e) => setAutoCalculate(e.target.checked)}
                    color="primary"
                  />
                }
                label="Auto-calculate risk metrics"
                sx={{ mb: 3 }}
              />
              
              <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Stop Loss"
                    name="stopLoss"
                    type="number"
                    value={values.stopLoss || ''}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value) || 0;
                      setFieldValue('stopLoss', value);
                      
                      if (autoCalculate) {
                        const riskAmount = calculateRiskAmount(values.entryPrice, value, values.positionSize, values.direction);
                        setFieldValue('riskAmount', riskAmount);
                        setFieldValue('riskPercentage', calculateRiskPercentage(riskAmount));
                        
                        if (values.targetPrice) {
                          setFieldValue('rewardToRisk', calculateRewardToRisk(values.entryPrice, values.targetPrice, value, values.direction));
                        }
                      }
                    }}
                    variant="outlined"
                    required
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                    inputProps={{ step: 0.01 }}
                  />
                </Grid>
                
                <Grid size={{ xs: 12 }}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AttachMoney color="primary" />
                      Take Profit Targets
                    </Typography>
                    
                    <Paper sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
                      {/* Single target price field (for backward compatibility) */}
                      <Grid container spacing={2} sx={{ mb: 2 }}>
                        <Grid size={{ xs: 12, md: 6 }}>
                          <TextField
                            fullWidth
                            label="Target Price"
                            name="targetPrice"
                            type="number"
                            value={values.targetPrice || ''}
                            onChange={(e) => {
                              const value = parseFloat(e.target.value) || undefined;
                              setFieldValue('targetPrice', value);
                              
                              if (autoCalculate && value) {
                                setFieldValue('rewardToRisk', calculateRewardToRisk(
                                  values.entryPrice, 
                                  value, 
                                  values.stopLoss, 
                                  values.direction,
                                  values.takeProfitTargets
                                ));
                              }
                            }}
                            variant="outlined"
                            InputProps={{
                              startAdornment: <InputAdornment position="start">$</InputAdornment>,
                            }}
                            inputProps={{ step: 0.01 }}
                          />
                        </Grid>
                      </Grid>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Typography variant="subtitle2" gutterBottom>
                        Multiple Take Profit Targets
                      </Typography>
                      
                      {/* Multiple take profit targets */}
                      {values.takeProfitTargets && values.takeProfitTargets.map((target: TakeProfitTarget, index: number) => (
                        <Grid container spacing={2} key={index} sx={{ mb: 2, alignItems: 'center' }}>
                          <Grid size={{ xs: 12, md: 4 }}>
                            <TextField
                              fullWidth
                              label={`Target ${index + 1} Price`}
                              type="number"
                              value={target.price || ''}
                              onChange={(e) => {
                                const value = parseFloat(e.target.value) || 0;
                                const updatedTargets = [...values.takeProfitTargets];
                                updatedTargets[index] = { ...updatedTargets[index], price: value };
                                setFieldValue('takeProfitTargets', updatedTargets);
                                
                                if (autoCalculate) {
                                  setFieldValue('rewardToRisk', calculateRewardToRisk(
                                    values.entryPrice, 
                                    values.targetPrice, 
                                    values.stopLoss, 
                                    values.direction,
                                    updatedTargets
                                  ));
                                }
                              }}
                              variant="outlined"
                              InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                              }}
                              inputProps={{ step: 0.01 }}
                            />
                          </Grid>
                          <Grid size={{ xs: 12, md: 3 }}>
                            <TextField
                              fullWidth
                              label="Position %"
                              type="number"
                              value={target.percentage || ''}
                              onChange={(e) => {
                                const value = parseFloat(e.target.value) || 0;
                                const updatedTargets = [...values.takeProfitTargets];
                                updatedTargets[index] = { ...updatedTargets[index], percentage: value };
                                setFieldValue('takeProfitTargets', updatedTargets);
                              }}
                              variant="outlined"
                              InputProps={{
                                endAdornment: <InputAdornment position="end">%</InputAdornment>,
                              }}
                              inputProps={{ min: 0, max: 100 }}
                            />
                          </Grid>
                          <Grid size={{ xs: 12, md: 4 }}>
                            <TextField
                              fullWidth
                              label="Description (optional)"
                              value={target.description || ''}
                              onChange={(e) => {
                                const updatedTargets = [...values.takeProfitTargets];
                                updatedTargets[index] = { ...updatedTargets[index], description: e.target.value };
                                setFieldValue('takeProfitTargets', updatedTargets);
                              }}
                              variant="outlined"
                            />
                          </Grid>
                          <Grid size={{ xs: 12, md: 1 }}>
                            <IconButton 
                              color="error" 
                              onClick={() => {
                                const updatedTargets = values.takeProfitTargets.filter((_: any, i: number) => i !== index);
                                setFieldValue('takeProfitTargets', updatedTargets);
                              }}
                            >
                              <Remove />
                            </IconButton>
                          </Grid>
                        </Grid>
                      ))}
                      
                      <Button
                        variant="outlined"
                        startIcon={<Add />}
                        onClick={() => {
                          const newTarget: TakeProfitTarget = {
                            price: values.direction === 'Long' ? values.entryPrice * 1.05 : values.entryPrice * 0.95,
                            percentage: 50,
                            description: ''
                          };
                          const updatedTargets = [...(values.takeProfitTargets || []), newTarget];
                          setFieldValue('takeProfitTargets', updatedTargets);
                        }}
                        sx={{ mt: 1 }}
                      >
                        Add Take Profit Target
                      </Button>
                      
                      {values.takeProfitTargets && values.takeProfitTargets.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            Total position percentage: {values.takeProfitTargets.reduce((sum: number, target: TakeProfitTarget) => sum + (target.percentage || 0), 0)}%
                          </Typography>
                          {values.takeProfitTargets.reduce((sum: number, target: TakeProfitTarget) => sum + (target.percentage || 0), 0) !== 100 && (
                            <Alert severity="warning" sx={{ mt: 1 }}>
                              Total position percentage should equal 100%
                            </Alert>
                          )}
                        </Box>
                      )}
                    </Paper>
                  </Box>
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Position Size"
                    name="positionSize"
                    type="number"
                    value={values.positionSize || ''}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value) || 0;
                      setFieldValue('positionSize', value);
                      
                      if (autoCalculate) {
                        const riskAmount = calculateRiskAmount(values.entryPrice, values.stopLoss, value, values.direction);
                        setFieldValue('riskAmount', riskAmount);
                        setFieldValue('riskPercentage', calculateRiskPercentage(riskAmount));
                      }
                    }}
                    variant="outlined"
                    required
                    placeholder="Number of shares/contracts"
                  />
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Risk Amount"
                    name="riskAmount"
                    type="number"
                    value={values.riskAmount}
                    onChange={(e) => !autoCalculate && setFieldValue('riskAmount', parseFloat(e.target.value) || 0)}
                    variant="outlined"
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                      readOnly: autoCalculate,
                    }}
                    sx={{
                      '& .MuiInputBase-input': {
                        backgroundColor: autoCalculate ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                      },
                    }}
                  />
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Risk Percentage"
                    name="riskPercentage"
                    type="number"
                    value={values.riskPercentage}
                    onChange={(e) => !autoCalculate && setFieldValue('riskPercentage', parseFloat(e.target.value) || 0)}
                    variant="outlined"
                    InputProps={{
                      endAdornment: <InputAdornment position="end">%</InputAdornment>,
                      readOnly: autoCalculate,
                    }}
                    sx={{
                      '& .MuiInputBase-input': {
                        backgroundColor: autoCalculate ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                      },
                    }}
                  />
                </Grid>
                
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    label="Reward to Risk Ratio"
                    name="rewardToRisk"
                    type="number"
                    value={calculateRewardToRisk(
                      values.entryPrice, 
                      values.targetPrice, 
                      values.stopLoss, 
                      values.direction,
                      values.takeProfitTargets
                    ).toFixed(2)}
                    variant="outlined"
                    InputProps={{
                      readOnly: true,
                      startAdornment: (
                        <InputAdornment position="start">
                          <Calculate />
                        </InputAdornment>
                      ),
                    }}
                    sx={{
                      '& .MuiInputBase-input': {
                        backgroundColor: 'rgba(0, 0, 0, 0.04)',
                      },
                    }}
                  />
                </Grid>
              </Grid>
              
              {values.riskPercentage > 2 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  Risk percentage is above 2%. Consider reducing position size.
                </Alert>
              )}
              
              {calculateRewardToRisk(values.entryPrice, values.targetPrice, values.stopLoss, values.direction) < 1 && values.targetPrice && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  Reward to risk ratio is below 1:1. Consider adjusting your target or stop loss.
                </Alert>
              )}
            </Box>
          </Fade>
        );
        
      case 2:
        return (
          <Fade in timeout={500}>
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <Psychology color="primary" />
                Psychology & Notes
              </Typography>
              
              <Grid container spacing={3}>
                <Grid size={{xs:12,md:6}}>
                  <FormControl fullWidth>
                    <InputLabel>Market Condition</InputLabel>
                    <Select
                      value={values.marketCondition || ''}
                      label="Market Condition"
                      onChange={(e) => setFieldValue('marketCondition', e.target.value)}
                    >
                      <MenuItem value="">Select Market Condition</MenuItem>
                      <MenuItem value="Trending">Trending</MenuItem>
                      <MenuItem value="Ranging">Ranging</MenuItem>
                      <MenuItem value="Volatile">Volatile</MenuItem>
                      <MenuItem value="Quiet">Quiet</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{xs:12,md:6}}>
                  <FormControl fullWidth>
                    <InputLabel>Psychological State (Before)</InputLabel>
                    <Select
                      value={values.psychologicalState?.before || ''}
                      label="Psychological State (Before)"
                      onChange={(e) => setFieldValue('psychologicalState.before', e.target.value)}
                    >
                      <MenuItem value="">Select State</MenuItem>
                      <MenuItem value="Calm">😌 Calm</MenuItem>
                      <MenuItem value="Excited">🚀 Excited</MenuItem>
                      <MenuItem value="Fearful">😰 Fearful</MenuItem>
                      <MenuItem value="Confident">💪 Confident</MenuItem>
                      <MenuItem value="Uncertain">🤔 Uncertain</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{xs:12,md:6}}>
                  <FormControl fullWidth>
                    <InputLabel>Psychological State (During)</InputLabel>
                    <Select
                      value={values.psychologicalState?.during || ''}
                      label="Psychological State (During)"
                      onChange={(e) => setFieldValue('psychologicalState.during', e.target.value)}
                    >
                      <MenuItem value="">Select State</MenuItem>
                      <MenuItem value="Calm">😌 Calm</MenuItem>
                      <MenuItem value="Excited">🚀 Excited</MenuItem>
                      <MenuItem value="Fearful">😰 Fearful</MenuItem>
                      <MenuItem value="Confident">💪 Confident</MenuItem>
                      <MenuItem value="Uncertain">🤔 Uncertain</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{xs:12,md:6}}>
                  <FormControl fullWidth>
                    <InputLabel>Psychological State (After)</InputLabel>
                    <Select
                      value={values.psychologicalState?.after || ''}
                      label="Psychological State (After)"
                      onChange={(e) => setFieldValue('psychologicalState.after', e.target.value)}
                    >
                      <MenuItem value="">Select State</MenuItem>
                      <MenuItem value="Calm">😌 Calm</MenuItem>
                      <MenuItem value="Excited">🚀 Excited</MenuItem>
                      <MenuItem value="Fearful">😰 Fearful</MenuItem>
                      <MenuItem value="Confident">💪 Confident</MenuItem>
                      <MenuItem value="Uncertain">🤔 Uncertain</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid size={{xs:12}}>
                  <TextField
                    fullWidth
                    label="Notes"
                    name="notes"
                    multiline
                    rows={4}
                    value={values.notes}
                    onChange={(e) => setFieldValue('notes', e.target.value)}
                    variant="outlined"
                    placeholder="Add your trade notes, observations, and lessons learned..."
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start" sx={{ alignSelf: 'flex-start', mt: 1 }}>
                          <Notes />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{xs:12}}>
  <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
    <Image color="primary" />
    Images
  </Typography>
  
  <Paper
    sx={{
      p: 3,
      border: '2px dashed #ccc',
      borderRadius: 2,
      backgroundColor: '#f9f9f9',
      textAlign: 'center',
      mb: 2,
      cursor: 'pointer',
      '&:hover': {
        borderColor: 'primary.main',
        backgroundColor: '#f0f7ff'
      }
    }}
    onDragOver={handleDragOver}
    onDrop={(e) => handleDrop(e, setFieldValue, values.images)}
    onPaste={(e) => handlePaste(e, setFieldValue, values.images || [])}
    tabIndex={0}
  >
    <input
      type="file"
      accept="image/*"
      multiple
      id="image-upload"
      style={{ display: 'none' }}
      onChange={(e) => handleFileSelect(e.target.files, setFieldValue, values.images)}
    />
    <label htmlFor="image-upload" style={{ cursor: 'pointer', width: '100%', height: '100%', display: 'block' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Image fontSize="large" color="primary" />
        <Typography variant="body1">
          Drag & drop images here, or click to select files
        </Typography>
        <Typography variant="caption" color="textSecondary">
          You can also paste images directly (Ctrl+V)
        </Typography>
        <Button
          variant="contained"
          component="span"
          startIcon={<Image />}
        >
          Select Images
        </Button>
      </Box>
    </label>
  </Paper>
  
  {isUploading && (
    <Box sx={{ width: '100%', mt: 2, mb: 2 }}>
      <Typography variant="caption" color="textSecondary">
        Uploading images...
      </Typography>
      <LinearProgress />
    </Box>
  )}
  
  {values.images && values.images.length > 0 && (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Uploaded Images ({values.images.length})
      </Typography>
      <Grid container spacing={2}>
        {values.images.map((image: JournalImage, index: number) => (
          <Grid size={{xs:12, sm:6, md:4}} key={index}>
            <Paper
              elevation={2}
              sx={{
                p: 1,
                position: 'relative',
                '&:hover .image-actions': {
                  opacity: 1
                }
              }}
            >
              <img 
                src={image.url.startsWith('/') ? `http://localhost:5000${image.url}` : image.url} 
                alt={image.description || `Image ${index + 1}`}
                style={{ width: '100%', height: 'auto', display: 'block' }}
              />
              <Box
                className="image-actions"
                sx={{
                  position: 'absolute',
                  top: 0,
                  right: 0,
                  p: 0.5,
                  opacity: 0,
                  transition: 'opacity 0.2s',
                  backgroundColor: 'rgba(0,0,0,0.5)',
                  borderRadius: '0 0 0 4px'
                }}
              >
                <IconButton
                  size="small"
                  color="error"
                  onClick={() => handleRemoveImage(index, values.images || [], setFieldValue)}
                  sx={{ color: 'white' }}
                >
                  <Cancel />
                </IconButton>
              </Box>
              <TextField
                fullWidth
                size="small"
                placeholder="Image description"
                value={image.description || ''}
                onChange={(e) => {
                  const updatedImages = [...values.images];
                  updatedImages[index] = {
                    ...updatedImages[index],
                    description: e.target.value
                  };
                  setFieldValue('images', updatedImages);
                }}
                sx={{ mt: 1 }}
              />
              <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                <InputLabel>Category</InputLabel>
                <Select
                  value={image.category || 'chart'}
                  label="Category"
                  onChange={(e) => {
                    const updatedImages = [...values.images];
                    updatedImages[index] = {
                      ...updatedImages[index],
                      category: e.target.value as any
                    };
                    setFieldValue('images', updatedImages);
                  }}
                >
                  <MenuItem value="chart">Chart</MenuItem>
                  <MenuItem value="setup">Setup</MenuItem>
                  <MenuItem value="entry">Entry</MenuItem>
                  <MenuItem value="exit">Exit</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  )}
</Grid>
                
                <Grid size={{xs:12,md:6}}>
                  <TextField
                    fullWidth
                    label="Mistakes (comma separated)"
                    name="mistakes"
                    value={values.mistakes?.join(', ') || ''}
                    onChange={(e) => {
                      const value = e.target.value;
                      const mistakesArray = value.split(',').map((item) => item.trim()).filter(Boolean);
                      setFieldValue('mistakes', mistakesArray);
                    }}
                    variant="outlined"
                    placeholder="e.g., Entered too early, Ignored stop loss"
                  />
                  <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {values.mistakes?.map((mistake: string, index: number) => (
                      <Chip
                        key={index}
                        label={mistake}
                        size="small"
                        color="error"
                        variant="outlined"
                        onDelete={() => {
                          const newMistakes = values.mistakes.filter((_: any, i: number) => i !== index);
                          setFieldValue('mistakes', newMistakes);
                        }}
                      />
                    ))}
                  </Box>
                </Grid>
                
                <Grid size={{xs:12,md:6}}>
                  <TextField
                    fullWidth
                    label="Tags (comma separated)"
                    name="tags"
                    value={values.tags?.join(', ') || ''}
                    onChange={(e) => {
                      const value = e.target.value;
                      const tagsArray = value.split(',').map((item) => item.trim()).filter(Boolean);
                      setFieldValue('tags', tagsArray);
                    }}
                    variant="outlined"
                    placeholder="e.g., Breakout, High volume, News event"
                  />
                  <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {values.tags?.map((tag: string, index: number) => (
                      <Chip
                        key={index}
                        label={tag}
                        size="small"
                        color="primary"
                        variant="outlined"
                        onDelete={() => {
                          const newTags = values.tags.filter((_: any, i: number) => i !== index);
                          setFieldValue('tags', newTags);
                        }}
                      />
                    ))}
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Fade>
        );
        
      default:
        return null;
    }
  };

  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', p: 3 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 4 }}>
          <AutoFixHigh color="primary" />
          {isEditing ? 'Edit Trade' : 'Add New Trade'}
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel
                icon={step.icon}
                sx={{
                  '& .MuiStepLabel-iconContainer': {
                    color: index === activeStep ? 'primary.main' : 'grey.400',
                  },
                }}
              >
                {step.label}
              </StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Formik
          initialValues={initialValues}
          validationSchema={validationSchema}
          onSubmit={onSubmit}
          enableReinitialize={isEditing}
        >
          {({ values, setFieldValue, isSubmitting, errors, touched }) => (
            <Form>
              <Box sx={{ mb: 4 }}>
                {renderStepContent(activeStep, values, setFieldValue)}
              </Box>
              
              <Divider sx={{ my: 3 }} />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  {activeStep > 0 && (
                    <Button
                      onClick={handleBack}
                      startIcon={<Cancel />}
                      variant="outlined"
                      sx={{ mr: 1 }}
                    >
                      Back
                    </Button>
                  )}
                </Box>
                
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={() => navigate('/journal')}
                    startIcon={<Cancel />}
                  >
                    Cancel
                  </Button>
                  
                  {activeStep === steps.length - 1 ? (
                    <Button
                      type="submit"
                      variant="contained"
                      disabled={isSubmitting}
                      startIcon={<Save />}
                      sx={{
                        minWidth: 140,
                        background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                        '&:hover': {
                          background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
                        },
                      }}
                    >
                      {isSubmitting ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress sx={{ height: 20 }} />
                          Saving...
                        </Box>
                      ) : (
                        isEditing ? 'Update Trade' : 'Save Trade'
                      )}
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      endIcon={<Speed />}
                    >
                      Next
                    </Button>
                  )}
                </Box>
              </Box>
              
              {isSubmitting && (
                <LinearProgress sx={{ mt: 2, borderRadius: 1 }} />
              )}
            </Form>
          )}
        </Formik>
      </Paper>
    </Box>
  );
};

export default EntryForm;

