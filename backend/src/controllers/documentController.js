const axios = require('axios');
const FormData = require('form-data');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const uploadDocument = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Please upload a valid PDF file.' });
  }

  if (req.file.mimetype !== 'application/pdf') {
    return res.status(400).json({ error: 'File must be a PDF.' });
  }

  try {
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname || 'document.pdf',
      contentType: req.file.mimetype || 'application/pdf',
    });

    const response = await axios.post(`${ML_SERVICE_URL}/extract-text`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error extracting text from ML service:', error.message);
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(500).json({ error: 'Failed to communicate with ML service.' });
  }
};

module.exports = {
  uploadDocument
};

