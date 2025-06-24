import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Box,
  Chip
} from '@mui/material';
import { Link } from 'react-router-dom';
import { format } from 'date-fns';
import { Journal } from '../../types/Journal';

interface RecentTradesProps {
  trades: Journal[];
}

const RecentTrades: React.FC<RecentTradesProps> = ({ trades }) => {
  const getOutcomeColor = (outcome: string | undefined) => {
    switch (outcome?.toLowerCase()) {
      case 'win': return 'success';
      case 'loss': return 'error';
      case 'breakeven': return 'warning';
      default: return 'default';
    }
  };

  return (
    <TableContainer sx={{ 
      maxHeight: 400,
      '& .MuiTableCell-root': {
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        maxWidth: {
          xs: '80px',
          sm: '120px',
          md: '150px'
        }
      },
      '& .MuiTableCell-root:nth-of-type(3)': { // Setup column
        maxWidth: {
          xs: '100px',
          sm: '150px',
          md: '200px'
        }
      }
    }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Instrument</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Setup</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Direction</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Outcome</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>P&L</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trades.map((trade) => (
            <TableRow key={trade._id} hover>
              <TableCell>
                {format(new Date(trade.entryDate), 'MM/dd/yyyy')}
              </TableCell>
              <TableCell title={trade.instrument}>
                {trade.instrument}
              </TableCell>
              <TableCell title={trade.setup}>
                {trade.setup}
              </TableCell>
              <TableCell>{trade.direction}</TableCell>
              <TableCell>
                <Chip 
                  label={trade.outcome || 'Open'} 
                  color={getOutcomeColor(trade.outcome)}
                  size="small"
                />
              </TableCell>
              <TableCell>
                {trade.pnl ? (
                  <Box 
                    component="span" 
                    sx={{ 
                      color: trade.pnl >= 0 ? 'success.main' : 'error.main',
                      fontWeight: 600
                    }}
                  >
                    ${trade.pnl.toFixed(2)}
                  </Box>
                ) : (
                  'N/A'
                )}
              </TableCell>
              <TableCell>
                <Button 
                  component={Link} 
                  to={`/journal/view/${trade._id}`} 
                  variant="contained" 
                  size="small"
                  sx={{ minWidth: 'auto', px: 2 }}
                >
                  View
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default RecentTrades;