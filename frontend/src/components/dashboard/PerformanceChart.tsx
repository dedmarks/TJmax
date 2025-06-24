import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { format } from 'date-fns';
import { EquityPoint } from '../../types/Metrics';
import { Box, useTheme } from '@mui/material';

interface PerformanceChartProps {
  data: EquityPoint[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ data }) => {
  const theme = useTheme();
  
  const formattedData = data.map((point) => ({
    date: format(new Date(point.date), 'MM/dd/yyyy'),
    equity: point.equity,
    formattedEquity: `$${point.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }));

  // Determine if the equity is trending up or down
  const isPositiveTrend = data.length > 1 && data[data.length - 1].equity > data[0].equity;
  
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 2,
            p: 2,
            boxShadow: theme.shadows[4]
          }}
        >
          <Box sx={{ fontWeight: 600, mb: 1, color: theme.palette.text.primary }}>
            {label}
          </Box>
          <Box sx={{ 
            color: isPositiveTrend ? theme.palette.success.main : theme.palette.error.main,
            fontWeight: 600
          }}>
            Equity: {payload[0].payload.formattedEquity}
          </Box>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box sx={{ width: '100%', height: 350 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart 
          data={formattedData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop 
                offset="5%" 
                stopColor={isPositiveTrend ? theme.palette.success.main : theme.palette.error.main} 
                stopOpacity={0.3}
              />
              <stop 
                offset="95%" 
                stopColor={isPositiveTrend ? theme.palette.success.main : theme.palette.error.main} 
                stopOpacity={0.05}
              />
            </linearGradient>
          </defs>
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={theme.palette.divider}
            opacity={0.3}
          />
          <XAxis 
            dataKey="date" 
            stroke={theme.palette.text.secondary}
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            stroke={theme.palette.text.secondary}
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="equity"
            stroke={isPositiveTrend ? theme.palette.success.main : theme.palette.error.main}
            strokeWidth={3}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ 
              r: 6, 
              fill: isPositiveTrend ? theme.palette.success.main : theme.palette.error.main,
              stroke: theme.palette.background.paper,
              strokeWidth: 2
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PerformanceChart;