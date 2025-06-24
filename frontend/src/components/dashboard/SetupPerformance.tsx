import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Box, Typography, useTheme } from '@mui/material';
import { SetupPerformance as SetupPerformanceType } from '../../types/Metrics';

interface SetupPerformanceProps {
  data: SetupPerformanceType[];
}

const SetupPerformance: React.FC<SetupPerformanceProps> = ({ data }) => {
  const theme = useTheme();
  
  const formattedData = data.map((setup, index) => ({
    name: setup.setup.length > 10 ? `${setup.setup.substring(0, 10)}...` : setup.setup,
    fullName: setup.setup,
    winRate: setup.winRate,
    expectancy: setup.expectancy,
    index
  }));

  // Color palette for bars
  const colors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.info.main,
    theme.palette.error.main,
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
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
            {data.fullName}
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.primary.main, mb: 0.5 }}>
            Win Rate: {payload[0].value?.toFixed(1)}%
          </Typography>
          <Typography variant="body2" sx={{ color: theme.palette.secondary.main }}>
            Expectancy: ${payload[1]?.value?.toFixed(2)}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  const CustomXAxisTick = (props: any) => {
    const { x, y, payload } = props;
    return (
      <g transform={`translate(${x},${y})`}>
        <text
          x={0}
          y={0}
          dy={16}
          textAnchor="middle"
          fill={theme.palette.text.secondary}
          fontSize={12}
          fontWeight={500}
        >
          {payload.value}
        </text>
      </g>
    );
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
            tick={<CustomXAxisTick />}
            axisLine={{ stroke: theme.palette.divider }}
            tickLine={{ stroke: theme.palette.divider }}
          />
          <YAxis 
            yAxisId="left" 
            orientation="left" 
            stroke={theme.palette.primary.main}
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
            axisLine={{ stroke: theme.palette.divider }}
            tickLine={{ stroke: theme.palette.divider }}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            stroke={theme.palette.secondary.main}
            tick={{ fill: theme.palette.text.secondary, fontSize: 12 }}
            axisLine={{ stroke: theme.palette.divider }}
            tickLine={{ stroke: theme.palette.divider }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,0,0,0.1)' }} />
          <Bar 
            yAxisId="left" 
            dataKey="winRate" 
            name="Win Rate (%)" 
            radius={[4, 4, 0, 0]}
            opacity={0.8}
          >
            {formattedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
          <Bar 
            yAxisId="right" 
            dataKey="expectancy" 
            name="Expectancy ($)" 
            radius={[4, 4, 0, 0]}
            opacity={0.6}
          >
            {formattedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[(index + 3) % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default SetupPerformance;