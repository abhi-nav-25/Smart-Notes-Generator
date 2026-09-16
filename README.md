# Smart Notes Generator

Smart Notes Generator is an AI-powered study assistant that transforms PDF study materials into structured notes and AI-generated quiz questions. It helps students understand, revise, and practice content from their study documents.

## Live Demo

[Open Smart Notes Generator](https://smart-notes-generator-gamma.vercel.app/)

## Features

- Upload PDF study materials.
- Extract text from uploaded documents.
- Generate structured study notes.
- Generate quiz questions from extracted content.
- View generated notes and quiz questions through a responsive interface.
- Use AI-powered content generation through the Gemini API.
- Separate frontend, backend, and ML service architecture.

## Tech Stack

### Frontend

- React
- Vite
- Tailwind CSS
- JavaScript

### Backend

- Node.js
- Express.js
- Axios
- Multer

### ML Service

- Python
- FastAPI
- PDF text extraction
- Gemini API

### Deployment

- Vercel — Frontend
- Render — Backend and ML Service

## Project Structure

```text
smart-notes-generator/
├── frontend/       # React + Vite + Tailwind CSS interface
├── backend/        # Node.js + Express REST API
├── ml_service/     # Python FastAPI ML service
├── README.md       # Project documentation
└── .gitignore      # Git ignore rules
