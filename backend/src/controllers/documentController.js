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

    const extractResponse = await axios.post(`${ML_SERVICE_URL}/extract-text`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    const extractedText = extractResponse.data.text;
    const pages = extractResponse.data.pages;

    const summaryResponse = await axios.post(`${ML_SERVICE_URL}/generate-summary`, {
      text: extractedText,
    });

    return res.status(200).json({
      text: extractedText,
      pages,
      notes: summaryResponse.data.summary,
    });
  } catch (error) {
    console.error('Error processing document with ML service:', error.message);
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(500).json({ error: 'Failed to communicate with ML service.' });
  }
};

const generateQuiz = async (req, res) => {
  const { text } = req.body || {};

  if (typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'Text is required to generate quiz questions.' });
  }

  try {
    const response = await axios.post(`${ML_SERVICE_URL}/generate-quiz`, { text });
    return res.status(200).json(response.data);
  } catch (error) {
    console.error('Error generating quiz from ML service:', error.message);
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(500).json({ error: 'Failed to communicate with ML service.' });
  }
};

const generateNotes = async (req, res) => {
  const { text } = req.body || {};

  if (typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'Text is required to generate notes.' });
  }

  try {
    const response = await axios.post(`${ML_SERVICE_URL}/generate-notes`, { text });
    return res.status(200).json({ notes: response.data.notes });
  } catch (error) {
    console.error('Error generating notes from ML service:', error.message);
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(500).json({ error: 'Failed to communicate with ML service.' });
  }
};

module.exports = {
  uploadDocument,
  generateQuiz,
  generateNotes,
};

