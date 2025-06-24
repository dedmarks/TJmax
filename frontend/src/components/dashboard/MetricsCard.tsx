import React from 'react';
import { Card, CardContent, Typography, Box, useTheme } from '@mui/material';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { IconName, IconPrefix } from '@fortawesome/fontawesome-svg-core';

interface MetricsCardProps {
  title: string;
  value: string | number;
  icon: string;
  color: 'primary' | 'success' | 'warning' | 'danger' | 'info';
}

const MetricsCard: React.FC<MetricsCardProps> = ({ title, value, icon, color }) => {
  const theme = useTheme();
  
  // Map color prop to theme color
  const getColor = () => {
    switch (color) {
      case 'primary':
        return theme.palette.primary.main;
      case 'success':
        return theme.palette.success.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'danger':
        return theme.palette.error.main;
      case 'info':
        return theme.palette.info.main;
      default:
        return theme.palette.primary.main;
    }
  };

  // Parse icon string to get prefix and name for FontAwesome
  const getIconParts = (): [IconPrefix, IconName] => {
    const parts = icon.split(' ');
    if (parts.length === 2) {
      return [parts[0].replace('fa-', '') as IconPrefix, parts[1] as IconName];
    }
    return ['fas', icon.replace('fa-', '') as IconName];
  };

  return (
    <Card sx={{ bgcolor: getColor(), color: 'white', height: '100%' }}>
      <CardContent sx={{ display: 'flex', alignItems: 'center', p: 3 }}>
        <Box sx={{ mr: 2, fontSize: '2.5rem' }}>
          <FontAwesomeIcon icon={getIconParts()} />
        </Box>
        <Box>
          <Typography variant="subtitle1" component="h4">
            {title}
          </Typography>
          <Typography variant="h4" component="h2">
            {value}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MetricsCard;