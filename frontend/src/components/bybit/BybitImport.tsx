import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Alert,
  CircularProgress,
  Snackbar,
  SelectChangeEvent
} from '@mui/material';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { getBybitTrades, importBybitTrades } from '../../services/bybitService';

const BybitImport: React.FC = () => {
  const [category, setCategory] = useState<string>('linear');
  const [symbol, setSymbol] = useState<string>('');
  const [limit, setLimit] = useState<number>(100);
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  
  const handleCategoryChange = (event: SelectChangeEvent) => {
    setCategory(event.target.value as string);
  };
  
  const handleFetchTrades = async () => {
    setLoading(true);
    setError(null);
    setTrades([]);
    
    try {
      const params: any = { category };
      if (symbol) params.symbol = symbol;
      if (limit) params.limit = limit;
      if (startDate) params.startTime = startDate.getTime();
      if (endDate) params.endTime = endDate.getTime();
      
      const fetchedTrades = await getBybitTrades(params);
      setTrades(fetchedTrades);
      setSuccess(`Successfully fetched ${fetchedTrades.length} trades`);
    } catch (err: any) {
      setError(err.response?.data?.msg || 'Failed to fetch trades from Bybit');
    } finally {
      setLoading(false);
    }
  };
  
  const handleImportTrades = async () => {
    if (trades.length === 0) {
      setError('No trades to import');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await importBybitTrades(trades);
      setSuccess(result.msg);
      setTrades([]);
    } catch (err: any) {
      setError(err.response?.data?.msg || 'Failed to import trades');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Import Trades from Bybit
        </Typography>
        
        <Grid container spacing={2}>
          <Grid size={{xs:12, md:4}} >
            <FormControl fullWidth>
              <InputLabel>Category</InputLabel>
              <Select
                value={category}
                label="Category"
                onChange={handleCategoryChange}
              >
                <MenuItem value="linear">USDT Perpetual</MenuItem>
                <MenuItem value="inverse">Inverse Perpetual</MenuItem>
                <MenuItem value="spot">Spot</MenuItem>
                <MenuItem value="option">Option</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid size={{xs:12, md:4}}>
            <TextField
              fullWidth
              label="Symbol (e.g., BTCUSDT)"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="Leave empty for all symbols"
            />
          </Grid>
          
          <Grid size={{xs:12, md:4}}>
            <TextField
              fullWidth
              type="number"
              label="Limit"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value))}
              InputProps={{ inputProps: { min: 1, max: 1000 } }}
            />
          </Grid>
          
          <Grid size={{xs:12, md:4}}>
            <DateTimePicker
              label="Start Date"
              value={startDate}
              onChange={(newValue) => setStartDate(newValue)}
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
          
          <Grid size={{xs:12, md:4}}>
            <DateTimePicker
              label="End Date"
              value={endDate}
              onChange={(newValue) => setEndDate(newValue)}
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleFetchTrades}
            disabled={loading || !category}
          >
            {loading ? <CircularProgress size={24} /> : 'Fetch Trades'}
          </Button>
          
          <Button
            variant="contained"
            color="success"
            onClick={handleImportTrades}
            disabled={loading || trades.length === 0}
          >
            Import {trades.length} Trades
          </Button>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
        
        {trades.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6">
              {trades.length} Trades Found
            </Typography>
            <Typography variant="body2">
              Ready to import. Click the "Import" button above to add these trades to your journal.
            </Typography>
          </Box>
        )}
        
        <Snackbar
          open={!!success}
          autoHideDuration={6000}
          onClose={() => setSuccess(null)}
          message={success}
        />
      </Paper>
    </LocalizationProvider>
  );
};

export default BybitImport;