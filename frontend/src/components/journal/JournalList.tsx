import React, { useContext, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Container,
  Fade,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  TrendingUp,
  TrendingDown,
  Remove as RemoveIcon
} from '@mui/icons-material';
import JournalContext from '../../context/journal/JournalContext';
import JournalItem from './JournalItem';
import Spinner from '../common/Spinner';
import { formatDate, formatCurrency } from '../../utils/formatters';

const JournalList: React.FC = () => {
  const journalContext = useContext(JournalContext);
  const { journals, loading, getJournals, deleteJournal } = journalContext;

  useEffect(() => {
    getJournals();
  }, []);

  const getOutcomeIcon = (outcome: string | undefined) => {
    switch (outcome) {
      case 'Win':
        return <TrendingUp sx={{ color: 'success.main', fontSize: 20 }} />;
      case 'Loss':
        return <TrendingDown sx={{ color: 'error.main', fontSize: 20 }} />;
      default:
        return <RemoveIcon sx={{ color: 'text.secondary', fontSize: 20 }} />;
    }
  };

  const getOutcomeColor = (outcome: string | undefined) => {
    switch (outcome) {
      case 'Win':
        return 'success';
      case 'Loss':
        return 'error';
      case 'Breakeven':
        return 'warning';
      default:
        return 'default';
    }
  };

  if (loading) {
    return <Spinner />;
  }

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Fade in timeout={600}>
        <Box>
          {/* Header */}
          <Card 
            elevation={1}
            sx={{ 
              mb: 4,
              borderRadius: 3,
              border: '1px solid',
              borderColor: 'divider'
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Typography 
                    variant="h3" 
                    component="h1" 
                    sx={{ 
                      background: 'linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      fontWeight: 700,
                      mb: 1
                    }}
                  >
                    Trading Journal 📊
                  </Typography>
                  <Typography variant="h6" color="text.secondary">
                    Track and analyze your trading performance
                  </Typography>
                </Box>
                <Button
                  component={Link}
                  to="/journal/new"
                  variant="contained"
                  startIcon={<AddIcon />}
                  sx={{
                    borderRadius: 3,
                    px: 3,
                    py: 1.5,
                    background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)',
                      transform: 'translateY(-2px)',
                      boxShadow: 3
                    },
                    transition: 'all 0.2s ease'
                  }}
                >
                  Add Trade
                </Button>
              </Box>
            </CardContent>
          </Card>

          {/* Journal Table */}
          {journals.length === 0 ? (
            <Card 
              elevation={1}
              sx={{ 
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <CardContent sx={{ p: 6, textAlign: 'center' }}>
                <TrendingUp sx={{ fontSize: 64, color: 'text.secondary', mb: 2, opacity: 0.5 }} />
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  No journal entries found
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  Start adding your trades to track your performance and improve your trading strategy!
                </Typography>
                <Button
                  component={Link}
                  to="/journal/new"
                  variant="contained"
                  startIcon={<AddIcon />}
                  sx={{
                    borderRadius: 3,
                    px: 4,
                    py: 1.5,
                    background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)',
                      transform: 'translateY(-2px)',
                      boxShadow: 3
                    },
                    transition: 'all 0.2s ease'
                  }}
                >
                  Add Your First Trade
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card 
              elevation={1}
              sx={{ 
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'divider',
                overflow: 'hidden'
              }}
            >
              <TableContainer>
                <Table sx={{ minWidth: 650 }}>
                  <TableHead>
                    <TableRow sx={{ 
                      background: (theme) => 
                        theme.palette.mode === 'dark' 
                          ? 'linear-gradient(135deg, #374151 0%, #4b5563 100%)'
                          : 'linear-gradient(135deg, rgb(194, 200, 206) 0%, #e2e8f0 100%)',
                      '& .MuiTableCell-root': {
                        borderBottom: (theme) => `1px solid ${theme.palette.divider}`
                      }
                    }}>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Date</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Instrument</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Setup</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Direction</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Outcome</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>P&L</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>R:R</TableCell>
                      <TableCell sx={{ fontWeight: 600, color: 'text.primary' }}>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {journals.map((journal, index) => (
                      <TableRow 
                        key={journal._id}
                        sx={{
                          '&:hover': {
                            backgroundColor: 'action.hover',
                            transform: 'scale(1.01)',
                          },
                          transition: 'all 0.2s ease',
                          animation: `fadeInUp 0.6s ease ${index * 0.1}s both`
                        }}
                      >
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {formatDate(journal.entryDate)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {journal.instrument}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={journal.setup} 
                            size="small" 
                            variant="outlined"
                            sx={{ borderRadius: 2 }}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={journal.direction}
                            size="small"
                            color={journal.direction === 'Long' ? 'success' : 'error'}
                            sx={{ borderRadius: 2, fontWeight: 600 }}
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getOutcomeIcon(journal.outcome)}
                            <Chip 
                              label={journal.outcome || 'Open'}
                              size="small"
                              color={getOutcomeColor(journal.outcome) as any}
                              sx={{ borderRadius: 2, fontWeight: 600 }}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontWeight: 600,
                              color: journal.pnl && journal.pnl > 0 ? 'success.main' : 
                                     journal.pnl && journal.pnl < 0 ? 'error.main' : 'text.secondary'
                            }}
                          >
                            {journal.pnl ? formatCurrency(journal.pnl) : '-'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {journal.rewardToRisk ? `${journal.rewardToRisk.toFixed(2)}:1` : '-'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Tooltip title="Edit Trade">
                              <IconButton 
                                component={Link}
                                to={`/journal/${journal._id}`}
                                size="small"
                                sx={{ 
                                  color: 'primary.main',
                                  '&:hover': {
                                    backgroundColor: 'primary.light',
                                    color: 'white'
                                  }
                                }}
                              >
                                <EditIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Delete Trade">
                              <IconButton 
                                onClick={() => deleteJournal(journal._id)}
                                size="small"
                                sx={{ 
                                  color: 'error.main',
                                  '&:hover': {
                                    backgroundColor: 'error.light',
                                    color: 'white'
                                  }
                                }}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Card>
          )}
        </Box>
      </Fade>
    </Container>
  );
};

export default JournalList;