import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { Box, Typography, useTheme } from '@mui/material';
import { DurationPerformance as DurationPerformanceType } from '../../types/Metrics';

interface DurationPerformanceProps {
  data: DurationPerformanceType[];
}

const DurationPerformance: React.FC<DurationPerformanceProps> = ({ data }) => {
  const theme = useTheme();
  
  const formattedData = data.map((duration) => ({
    name: duration.durationCategory,
    winRate: duration.winRate,
    expectancy: duration.expectancy,
    averagePnl: duration.averagePnl,
    count: duration.count
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            padding: 2,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 2,
            boxShadow: theme.shadows[8],
            minWidth: 200
          }}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1, color: theme.palette.text.primary }}>
            {label}
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.primary.main, mb: 0.5 }}>
            Win Rate: {payload[0].value?.toFixed(1)}%
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.secondary.main, mb: 0.5 }}>
            Expectancy: ${payload[1]?.value?.toFixed(2)}
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.success.main, mb: 0.5 }}>
            Avg P&L: ${payload[2]?.value?.toFixed(2)}
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.text.secondary }}>
            Trade Count: {payload[3]?.value}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box sx={{ width: '100%', height: 400 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={formattedData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          barCategoryGap="20%"
        >
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={theme.palette.divider}
            opacity={0.3}
          />
          <XAxis 
            dataKey="name" 
            axisLine={{ stroke: theme.palette.divider }}
            tickLine={{ stroke: theme.palette.divider }}
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis 
            yAxisId="left" 
            orientation="left" 
            stroke={theme.palette.primary.main}
            tickLine={{ stroke: theme.palette.primary.main }}
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
            label={{ 
              value: 'Win Rate (%)', 
              angle: -90, 
              position: 'insideLeft',
              fill: theme.palette.text.secondary,
              fontSize: 12
            }}
          />
          <YAxis 
            yAxisId="right"
            orientation="right"
            stroke={theme.palette.secondary.main}
            tickLine={{ stroke: theme.palette.secondary.main }}
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
            label={{ 
              value: 'Expectancy ($)', 
              angle: 90, 
              position: 'insideRight',
              fill: theme.palette.text.secondary,
              fontSize: 12
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey="winRate" 
            name="Win Rate" 
            yAxisId="left" 
            fill={theme.palette.primary.main} 
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="expectancy" 
            name="Expectancy" 
            yAxisId="right" 
            fill={theme.palette.secondary.main}
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="averagePnl" 
            name="Avg P&L" 
            yAxisId="right" 
            fill={theme.palette.success.main}
            radius={[4, 4, 0, 0]}
            hide
          />
          <Bar 
            dataKey="count" 
            name="Count" 
            yAxisId="right" 
            fill={theme.palette.info.main}
            radius={[4, 4, 0, 0]}
            hide
          />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default DurationPerformance;