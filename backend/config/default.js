module.exports = {
  mongoURI: process.env.MONGO_URI || 'mongodb://localhost:27017/tradingjournal',
  jwtSecret: process.env.JWT_SECRET || 'zN4dGliGwC837KZq',
  jwtExpiration: 3600 * 24 // 24 hours
};