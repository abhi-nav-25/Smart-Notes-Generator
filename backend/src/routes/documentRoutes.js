const express = require('express');
const multer = require('multer');
const { uploadDocument, generateQuiz, generateNotes } = require('../controllers/documentController');

const router = express.Router();

const storage = multer.memoryStorage();

const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'application/pdf') {
    cb(null, true);
  } else {
    cb(null, false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter
});

router.post('/upload', upload.single('file'), uploadDocument);
router.post('/quiz', generateQuiz);
router.post('/generate-notes', generateNotes);

module.exports = router;
