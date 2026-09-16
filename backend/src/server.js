const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

const documentRoutes = require('./routes/documentRoutes');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

app.use('/api/documents', documentRoutes);
app.get('/api/documents/test', (req, res) => {
  res.json({ message: 'Document routes are working' });
});
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Smart Notes Generator API is running'
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
