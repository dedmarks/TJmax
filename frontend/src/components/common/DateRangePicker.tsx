import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Chip,
  Stack,
  Typography,
  Fade,
  useTheme,
  alpha
} from '@mui/material';
import {
  CalendarToday,
  DateRange,
  TrendingUp,
  AccessTime,
  Refresh
} from '@mui/icons-material';
import { format, subDays, subWeeks, subMonths, startOfWeek, startOfMonth, startOfYear } from 'date-fns';

interface DateRangePickerProps {
  startDate: Date;
  endDate: Date;
  onChange: (startDate: Date, endDate: Date) => void;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({ startDate, endDate, onChange }) => {
  const theme = useTheme();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const presets = [
    {
      label: 'Last 7 Days',
      value: '7d',
      icon: <AccessTime />,
      startDate: subDays(new Date(), 7),
      endDate: new Date(),
      color: 'primary'
    },
    {
      label: 'Last 30 Days',
      value: '30d',
      icon: <CalendarToday />,
      startDate: subDays(new Date(), 30),
      endDate: new Date(),
      color: 'secondary'
    },
    {
      label: 'This Week',
      value: 'week',
      icon: <TrendingUp />,
      startDate: startOfWeek(new Date()),
      endDate: new Date(),
      color: 'success'
    },
    {
      label: 'This Month',
      value: 'month',
      icon: <DateRange />,
      startDate: startOfMonth(new Date()),
      endDate: new Date(),
      color: 'info'
    },
    {
      label: 'This Year',
      value: 'year',
      icon: <TrendingUp />,
      startDate: startOfYear(new Date()),
      endDate: new Date(),
      color: 'warning'
    }
  ];

  const handlePresetClick = (preset: any) => {
    setSelectedPreset(preset.value);
    onChange(preset.startDate, preset.endDate);
  };

  const handleCustomDateChange = () => {
    setSelectedPreset(null);
  };

  const formatDateForInput = (date: Date) => {
    return format(date, 'yyyy-MM-dd');
  };

  const getDaysDifference = () => {
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  return (
    <Box sx={{ py: 2 }}>
      <Box sx={{ mb: 3 }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 1
          }}
        >
          <DateRange />
          Date Range Selection
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Analyzing {getDaysDifference()} days of trading data
        </Typography>
      </Box>

      {/* Quick Presets */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Quick Presets
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {presets.map((preset) => (
            <Fade in key={preset.value} timeout={300}>
              <Chip
                icon={preset.icon}
                label={preset.label}
                onClick={() => handlePresetClick(preset)}
                color={selectedPreset === preset.value ? preset.color as any : 'default'}
                variant={selectedPreset === preset.value ? 'filled' : 'outlined'}
                sx={{
                  fontWeight: 500,
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: 2
                  },
                  ...(selectedPreset === preset.value && {
                    boxShadow: 2,
                    transform: 'translateY(-1px)'
                  })
                }}
              />
            </Fade>
          ))}
        </Stack>
      </Box>

      {/* Custom Date Range */}
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
          Custom Range
        </Typography>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="end">
          <TextField
            label="Start Date"
            type="date"
            value={formatDateForInput(startDate)}
            onChange={(e) => {
              handleCustomDateChange();
              onChange(new Date(e.target.value), endDate);
            }}
            InputLabelProps={{ shrink: true }}
            size="small"
            sx={{
              flex: 1,
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-1px)',
                  boxShadow: 1
                }
              }
            }}
          />
          <TextField
            label="End Date"
            type="date"
            value={formatDateForInput(endDate)}
            onChange={(e) => {
              handleCustomDateChange();
              onChange(startDate, new Date(e.target.value));
            }}
            InputLabelProps={{ shrink: true }}
            size="small"
            sx={{
              flex: 1,
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-1px)',
                  boxShadow: 1
                }
              }
            }}
          />
          <Button
            variant="contained"
            startIcon={<Refresh />}
            onClick={() => onChange(startDate, endDate)}
            sx={{
              borderRadius: 2,
              px: 3,
              py: 1,
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              '&:hover': {
                background: `linear-gradient(135deg, ${theme.palette.primary.dark}, ${theme.palette.secondary.dark})`,
                transform: 'translateY(-2px)',
                boxShadow: 3
              },
              transition: 'all 0.3s ease'
            }}
          >
            Apply
          </Button>
        </Stack>
      </Box>

      {/* Date Range Summary */}
      <Box
        sx={{
          mt: 3,
          p: 2,
          borderRadius: 2,
          background: alpha(theme.palette.info.main, 0.1),
          border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`
        }}
      >
        <Typography variant="body2" color="info.main" sx={{ fontWeight: 500 }}>
          📊 Selected Range: {format(startDate, 'MMM dd, yyyy')} - {format(endDate, 'MMM dd, yyyy')}
          ({getDaysDifference()} days)
        </Typography>
      </Box>
    </Box>
  );
};

export default DateRangePicker;