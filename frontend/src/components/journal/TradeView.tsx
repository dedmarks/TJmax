import React, { useContext, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Button,
  Container,
  Fade,
  ImageList,
  ImageListItem,
  IconButton,
  Modal
} from '@mui/material';
import {
  Timeline,
  AttachMoney,
  Psychology,
  Image,
  ArrowBack,
  ZoomIn,
  Close
} from '@mui/icons-material';
import JournalContext from '../../context/journal/JournalContext';
import AlertContext from '../../context/alert/AlertContext';
import Spinner from '../common/Spinner';
import { formatDate, formatCurrency } from '../../utils/formatters';
import { JournalImage } from '../../types/Journal';

const TradeView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const journalContext = useContext(JournalContext);
  const alertContext = useContext(AlertContext);
  const [openImageModal, setOpenImageModal] = React.useState(false);
  const [selectedImage, setSelectedImage] = React.useState<string>('');
  
  const { current, getCurrent, clearCurrent, error } = journalContext;
  const { setAlert } = alertContext;
  
  useEffect(() => {
    if (id) {
      getCurrent(id);
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
  
  if (!current) {
    return <Spinner />;
  }
  
  const getOutcomeColor = (outcome: string | undefined) => {
    switch (outcome?.toLowerCase()) {
      case 'win': return 'success';
      case 'loss': return 'error';
      case 'breakeven': return 'warning';
      default: return 'default';
    }
  };
  
  // Add state for the full-screen image modal

  
  const handleOpenImageModal = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setOpenImageModal(true);
  };
  
  const handleCloseImageModal = () => {
    setOpenImageModal(false);
  };
  
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Fade in timeout={500}>
        <Box>
          {/* Header with back button */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Button
              startIcon={<ArrowBack />}
              onClick={() => navigate(-1)}
              sx={{ mb: 2 }}
            >
              Back
            </Button>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
              Trade Details
            </Typography>
            <Box sx={{ width: 100 }} /> {/* Empty box for alignment */}
          </Box>
          
          {/* Trade Overview Card */}
          <Card sx={{ mb: 4, borderRadius: 2, boxShadow: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Timeline color="primary" />
                  Trade Overview
                </Typography>
                <Chip 
                  label={current.outcome || 'Open'} 
                  color={getOutcomeColor(current.outcome) as any}
                  sx={{ fontWeight: 600 }}
                />
              </Box>
              
              <Grid container spacing={3}>
                <Grid size={{xs:12, md:6}}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Instrument</Typography>
                    <Typography variant="h6">{current.instrument}</Typography>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Setup</Typography>
                    <Typography variant="h6">{current.setup}</Typography>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Direction</Typography>
                    <Chip 
                      label={current.direction} 
                      color={current.direction === 'Long' ? 'success' : 'error'}
                      sx={{ fontWeight: 600 }}
                    />
                  </Box>
                </Grid>
                
                <Grid size={{xs:12, md:6}}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Entry Date</Typography>
                    <Typography variant="h6">{formatDate(current.entryDate)}</Typography>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Entry Price</Typography>
                    <Typography variant="h6">{formatCurrency(current.entryPrice)}</Typography>
                  </Box>
                  
                  {current.exitDate && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="text.secondary">Exit Date</Typography>
                      <Typography variant="h6">{formatDate(current.exitDate)}</Typography>
                    </Box>
                  )}
                  
                  {current.exitPrice && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="text.secondary">Exit Price</Typography>
                      <Typography variant="h6">{formatCurrency(current.exitPrice)}</Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>
            </CardContent>
          </Card>
          
          {/* Risk Management Card */}
          <Card sx={{ mb: 4, borderRadius: 2, boxShadow: 3 }}>
            <CardContent>
              <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <AttachMoney color="primary" />
                Risk Management
              </Typography>
              
              <Grid container spacing={3}>
                <Grid size={{xs:12, md:6}}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Position Size</Typography>
                    <Typography variant="h6">{formatCurrency(current.positionSize)}</Typography>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Stop Loss</Typography>
                    <Typography variant="h6">{formatCurrency(current.stopLoss)}</Typography>
                  </Box>
                </Grid>
                
                <Grid size={{xs:12, md:6}}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Risk Amount</Typography>
                    <Typography variant="h6">{formatCurrency(current.riskAmount)}</Typography>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary">Risk Percentage</Typography>
                    <Typography variant="h6">{current.riskPercentage}%</Typography>
                  </Box>
                </Grid>
                
                {current.targetPrice && (
                  <Grid size={{xs:12, md:6}}>
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="text.secondary">Target Price</Typography>
                      <Typography variant="h6">{formatCurrency(current.targetPrice)}</Typography>
                    </Box>
                  </Grid>
                )}
                
                {current.rewardToRisk && (
                  <Grid size={{xs:12, md:6}}>
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="text.secondary">Reward to Risk</Typography>
                      <Typography variant="h6">{current.rewardToRisk.toFixed(2)}:1</Typography>
                    </Box>
                  </Grid>
                )}
                
                {current.pnl !== undefined && (
                  <Grid size={{xs:12, md:6}}>
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" color="text.secondary">P&L</Typography>
                      <Typography 
                        variant="h6" 
                        sx={{ 
                          color: current.pnl > 0 ? 'success.main' : 
                                 current.pnl < 0 ? 'error.main' : 'text.primary'
                        }}
                      >
                        {formatCurrency(current.pnl)}
                      </Typography>
                    </Box>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
          
          {/* Psychology & Notes Card */}
          <Card sx={{ mb: 4, borderRadius: 2, boxShadow: 3 }}>
            <CardContent>
              <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                <Psychology color="primary" />
                Psychology & Notes
              </Typography>
              
              {current.marketCondition && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="text.secondary">Market Condition</Typography>
                  <Chip label={current.marketCondition} sx={{ mt: 1 }} />
                </Box>
              )}
              
              {current.psychologicalState && (
                <Box>
                  {current.psychologicalState.before && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>Before Trade</Typography>
                      {current.psychologicalState.before.state && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">State</Typography>
                          <Typography variant="body1">{current.psychologicalState.before.state}</Typography>
                        </Box>
                      )}
                      {current.psychologicalState.before.notes && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">Notes</Typography>
                          <Typography variant="body1">{current.psychologicalState.before.notes}</Typography>
                        </Box>
                      )}
                    </Box>
                  )}
                  
                  {current.psychologicalState.during && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>During Trade</Typography>
                      {current.psychologicalState.during.state && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">State</Typography>
                          <Typography variant="body1">{current.psychologicalState.during.state}</Typography>
                        </Box>
                      )}
                      {current.psychologicalState.during.notes && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">Notes</Typography>
                          <Typography variant="body1">{current.psychologicalState.during.notes}</Typography>
                        </Box>
                      )}
                    </Box>
                  )}
                  
                  {current.psychologicalState.after && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>After Trade</Typography>
                      {current.psychologicalState.after.state && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">State</Typography>
                          <Typography variant="body1">{current.psychologicalState.after.state}</Typography>
                        </Box>
                      )}
                      {current.psychologicalState.after.notes && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="text.secondary">Notes</Typography>
                          <Typography variant="body1">{current.psychologicalState.after.notes}</Typography>
                        </Box>
                      )}
                    </Box>
                  )}
                  
                  {current.psychologicalState.lessons && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>Lessons Learned</Typography>
                      <Typography variant="body1">{current.psychologicalState.lessons}</Typography>
                    </Box>
                  )}
                </Box>
              )}
              
              {current.notes && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="text.secondary">General Notes</Typography>
                  <Typography variant="body1">{current.notes}</Typography>
                </Box>
              )}
              
              {current.mistakes && current.mistakes.length > 0 && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="text.secondary">Mistakes</Typography>
                  <Box sx={{ mt: 1 }}>
                    {current.mistakes.map((mistake, index) => (
                      <Chip 
                        key={index} 
                        label={mistake} 
                        sx={{ mr: 1, mb: 1 }} 
                        color="error" 
                        variant="outlined" 
                      />
                    ))}
                  </Box>
                </Box>
              )}
              
              {current.tags && current.tags.length > 0 && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" color="text.secondary">Tags</Typography>
                  <Box sx={{ mt: 1 }}>
                    {current.tags.map((tag, index) => (
                      <Chip 
                        key={index} 
                        label={tag} 
                        sx={{ mr: 1, mb: 1 }} 
                        color="primary" 
                        variant="outlined" 
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
          
          {/* Images Card */}
          {current.images && current.images.length > 0 && (
            <Card sx={{ mb: 4, borderRadius: 2, boxShadow: 3 }}>
              <CardContent>
                <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <Image color="primary" />
                  Images
                </Typography>
                
                <ImageList cols={2} gap={16}>
                  {current.images.map((image: string | JournalImage, index: number) => {
                    const imgUrl = typeof image === 'string' ? image : image.url;
                    const imgDesc = typeof image === 'string' ? '' : image.description;
                    const imgCategory = typeof image === 'string' ? 'chart' : image.category;
                    
                    // Fix the URL by prepending the base URL if it's a relative path
                    const fullImgUrl = imgUrl.startsWith('/') 
                      ? `http://localhost:5000${imgUrl}` 
                      : imgUrl;
                    
                    return (
                      <ImageListItem 
                        key={index} 
                        sx={{ 
                          overflow: 'hidden', 
                          borderRadius: 2,
                          cursor: 'pointer',
                          position: 'relative',
                          '&:hover .zoom-icon': {
                            opacity: 1
                          }
                        }}
                        onClick={() => handleOpenImageModal(fullImgUrl)}
                      >
                        <img 
                          src={fullImgUrl} 
                          alt={imgDesc || `Trade image ${index + 1}`} 
                          loading="lazy"
                          style={{ width: '100%', height: 'auto' }}
                        />
                        <Box 
                          className="zoom-icon"
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            backgroundColor: 'rgba(0,0,0,0.5)',
                            borderRadius: '50%',
                            padding: 1,
                            opacity: 0,
                            transition: 'opacity 0.3s'
                          }}
                        >
                          <ZoomIn sx={{ color: 'white', fontSize: 40 }} />
                        </Box>
                        {(imgDesc || imgCategory) && (
                          <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.5)', color: 'white' }}>
                            {imgCategory && (
                              <Chip 
                                label={imgCategory} 
                                size="small" 
                                sx={{ mb: 1 }} 
                                color="primary" 
                              />
                            )}
                            {imgDesc && (
                              <Typography variant="body2">{imgDesc}</Typography>
                            )}
                          </Box>
                        )}
                      </ImageListItem>
                    );
                  })}
                </ImageList>
              </CardContent>
            </Card>
          )}
          
          {/* Full-screen Image Modal */}
          <Modal
            open={openImageModal}
            onClose={handleCloseImageModal}
            aria-labelledby="full-screen-image"
            aria-describedby="full-screen-view-of-trade-image"
          >
            <Box sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '90vw',
              height: '90vh',
              bgcolor: 'background.paper',
              boxShadow: 24,
              p: 0,
              outline: 'none',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <Box sx={{
                display: 'flex',
                justifyContent: 'flex-end',
                p: 1,
                bgcolor: 'rgba(0,0,0,0.8)'
              }}>
                <IconButton 
                  onClick={handleCloseImageModal} 
                  sx={{ color: 'white' }}
                  aria-label="close"
                >
                  <Close />
                </IconButton>
              </Box>
              <Box sx={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'black',
                overflow: 'auto'
              }}>
                {selectedImage && (
                  <img
                    src={selectedImage}
                    alt="Full-screen view"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain'
                    }}
                  />
                )}
              </Box>
            </Box>
          </Modal>
        </Box>
      </Fade>
    </Container>
  );
};

export default TradeView;