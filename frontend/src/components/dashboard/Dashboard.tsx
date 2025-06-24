import React, { useContext, useEffect, useState } from 'react';
import { 
  Box, 
  Grid, // Make sure Grid is imported
  Card, 
  CardContent, 
  Typography, 
  Paper,
  Container,
  Fade,
  Grow
} from '@mui/material';
import { 
  TrendingUp,
  TrendingDown,
  Assessment,
  AccountBalance,
  Timeline,
  ShowChart
} from '@mui/icons-material';
import JournalContext from '../../context/journal/JournalContext';
import MetricsContext from '../../context/metrics/MetricsContext';
import AuthContext from '../../context/auth/AuthContext';
import PerformanceChart from './PerformanceChart';
import SetupPerformance from './SetupPerformance';
import RecentTrades from './RecentTrades';
import DateRangePicker from '../common/DateRangePicker';
import DurationPerformance from './DurationPerformance';

const Dashboard: React.FC = () => {
  const journalContext = useContext(JournalContext);
  const metricsContext = useContext(MetricsContext);
  const authContext = useContext(AuthContext);

  const { journals, getJournals } = journalContext;
  const { current, getMetrics, generateMetrics } = metricsContext;
  const { user } = authContext;

  const [dateRange, setDateRange] = useState({
    startDate: new Date(new Date().setMonth(new Date().getMonth() - 1)),
    endDate: new Date(),
  });

  useEffect(() => {
    getJournals();
    getMetrics();
  }, []);

  const onDateRangeChange = (startDate: Date, endDate: Date) => {
    setDateRange({ startDate, endDate });
    generateMetrics({
      period: 'Custom',
      startDate,
      endDate,
    });
  };

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Fade in timeout={600}>
        <Box>
          {/* Welcome Header */}
          <Box sx={{ mb: 4 }}>
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
              {getGreeting()}, {user?.name}!
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Here's your trading performance overview
            </Typography>
          </Box>

          {/* Date Range Picker */}
          <Grow in timeout={800}>
            <Paper 
              elevation={1} 
              sx={{ 
                p: 3, 
                mb: 4, 
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <DateRangePicker
                startDate={dateRange.startDate}
                endDate={dateRange.endDate}
                onChange={onDateRangeChange}
              />
            </Paper>
          </Grow>

          {/* Metrics Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {current && (
              <>
                <Grid size={{xs:12, sm:6, md: 3}}>
                  <Grow in timeout={1000}>
                    <Card 
                      elevation={1}
                      sx={{ 
                        height: '100%',
                        borderRadius: 3,
                        border: '1px solid',
                        borderColor: 'divider',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: 4,
                          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                          borderRadius: '12px 12px 0 0'
                        },
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 3
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <TrendingUp sx={{ color: 'success.main', mr: 1 }} />
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Win Rate
                          </Typography>
                        </Box>
                        <Typography variant="h3" sx={{ fontWeight: 700, color: 'text.primary' }}>
                          {current.winRate != null ? (current.winRate).toFixed(1) : '0.0'}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grow>
                </Grid>

                <Grid size={{xs:12, sm:6, md: 3}}>
                  <Grow in timeout={1200}>
                    <Card 
                      elevation={1}
                      sx={{ 
                        height: '100%',
                        borderRadius: 3,
                        border: '1px solid',
                        borderColor: 'divider',
                        position: 'relative',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: 4,
                          background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
                          borderRadius: '12px 12px 0 0'
                        },
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 3
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Assessment sx={{ color: 'info.main', mr: 1 }} />
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Profit Factor
                          </Typography>
                        </Box>
                        <Typography variant="h3" sx={{ fontWeight: 700, color: 'text.primary' }}>
                          {current.profitFactor != null ? current.profitFactor.toFixed(2) : '0.00'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grow>
                </Grid>

                <Grid size={{xs:12, sm:6, md: 3}}>
                  <Grow in timeout={1400}>
                    <Card 
                      elevation={1}
                      sx={{ 
                        height: '100%',
                        borderRadius: 3,
                        border: '1px solid',
                        borderColor: 'divider',
                        position: 'relative',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: 4,
                          background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                          borderRadius: '12px 12px 0 0'
                        },
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 3
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <AccountBalance sx={{ color: 'primary.main', mr: 1 }} />
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Expectancy
                          </Typography>
                        </Box>
                        <Typography variant="h3" sx={{ fontWeight: 700, color: 'text.primary' }}>
                          ${current.expectancy != null ? current.expectancy.toFixed(2) : '0.00'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grow>
                </Grid>

                <Grid size={{xs:12, sm:6, md: 3}}>
                  <Grow in timeout={1600}>
                    <Card 
                      elevation={1}
                      sx={{ 
                        height: '100%',
                        borderRadius: 3,
                        border: '1px solid',
                        borderColor: 'divider',
                        position: 'relative',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: 4,
                          background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                          borderRadius: '12px 12px 0 0'
                        },
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: 3
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <ShowChart sx={{ color: 'warning.main', mr: 1 }} />
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            SQN Score
                          </Typography>
                        </Box>
                        <Typography variant="h3" sx={{ fontWeight: 700, color: 'text.primary' }}>
                          {current.systemQualityNumber != null ? current.systemQualityNumber.toFixed(2) : '0.00'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grow>
                </Grid>
              </>
            )}
          </Grid>

          {/* Charts and Performance */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid size={{xs:12, lg:8}}>
              <Grow in timeout={1800}>
                <Card 
                  elevation={1}
                  sx={{ 
                    height: '100%',
                    borderRadius: 3,
                    border: '1px solid',
                    borderColor: 'divider'
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Timeline sx={{ color: 'primary.main', mr: 1 }} />
                      <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Equity Curve
                      </Typography>
                    </Box>
                    {current && current.equityCurve.length > 0 ? (
                      <PerformanceChart data={current.equityCurve} />
                    ) : (
                      <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
                        <Assessment sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                        <Typography variant="h6">No equity data available</Typography>
                        <Typography variant="body2">Start trading to see your performance curve</Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grow>
            </Grid>

            <Grid size={{xs:12, lg:4}}>
              <Grow in timeout={2000}>
                <Card 
                  elevation={1}
                  sx={{ 
                    height: '100%',
                    borderRadius: 3,
                    border: '1px solid',
                    borderColor: 'divider',
                    minHeight: 400
                  }}
                >
                  <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <TrendingUp sx={{ color: 'success.main', mr: 1 }} />
                      <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Setup Performance
                      </Typography>
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      {current && current.setupPerformance.length > 0 ? (
                        <SetupPerformance data={current.setupPerformance} />
                      ) : (
                        <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
                          <Assessment sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                          <Typography variant="h6">No setup data</Typography>
                          <Typography variant="body2">Add trades to analyze setups</Typography>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grow>
            </Grid>
          </Grid>

          {/* Trade Duration Analysis */}
          <Grow in timeout={2000}>
            <Card 
              elevation={1}
              sx={{ 
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'divider',
                mb: 4
              }}
            >
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Timeline sx={{ color: 'info.main', mr: 1 }} />
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    Trade Duration Analysis
                  </Typography>
                </Box>
                {current && current.durationPerformance && current.durationPerformance.length > 0 ? (
                  <DurationPerformance data={current.durationPerformance} />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
                    <Assessment sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                    <Typography variant="h6">No duration data available</Typography>
                    <Typography variant="body2">Complete more trades to see duration analysis</Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grow>
          
          {/* Recent Trades */}
          <Grow in timeout={2200}>
            <Card 
              elevation={1}
              sx={{ 
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Timeline sx={{ color: 'secondary.main', mr: 1 }} />
                  <Typography variant="h5" sx={{ fontWeight: 600 }}>
                    Recent Trades
                  </Typography>
                </Box>
                {journals.length > 0 ? (
                  <RecentTrades trades={journals.slice(0, 5)} />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 6, color: 'text.secondary' }}>
                    <Assessment sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                    <Typography variant="h6">No recent trades</Typography>
                    <Typography variant="body2">Start adding your trades to see them here</Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grow>
        </Box>
      </Fade>
    </Container>
  );
};

export default Dashboard;