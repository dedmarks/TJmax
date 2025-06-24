const mongoose = require('mongoose');
const config = require('config');
const db = config.get('mongoURI');

const connectDB = async () => {
    mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/parentpilot', {
        useNewUrlParser: true,
        useUnifiedTopology: true,
      })
      .then(() => console.log('Connected to MongoDB'))
      .catch(err => console.error('MongoDB connection error:', err));
};

module.exports = connectDB;