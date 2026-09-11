import { useState } from 'react'
import './App.css'

const normalizeChoice = (value, options = []) => {
  const raw = typeof value === 'string' ? value.trim() : ''
  if (!raw) return ''

  const normalized = raw.toLowerCase()
  const letterMatch = normalized.match(/^[a-d]$/)

  if (letterMatch) {
    return (options[letterMatch[0].charCodeAt(0) - 97] || letterMatch[0]).toLowerCase()
  }

  return normalized
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [extractedText, setExtractedText] = useState('')
  const [notes, setNotes] = useState('')
  const [quizQuestions, setQuizQuestions] = useState([])
  const [selectedAnswers, setSelectedAnswers] = useState({})
  const [quizScore, setQuizScore] = useState(null)
  const [isExtractedOpen, setIsExtractedOpen] = useState(false)
  const [isQuizModalOpen, setIsQuizModalOpen] = useState(false)
  const [quizSubmitted, setQuizSubmitted] = useState(false)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isQuizLoading, setIsQuizLoading] = useState(false)

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null
    setSelectedFile(file)
    setExtractedText('')
    setNotes('')
    setQuizQuestions([])
    setSelectedAnswers({})
    setQuizScore(null)
    setQuizSubmitted(false)
    setIsQuizModalOpen(false)
    setError('')
    setIsExtractedOpen(false)
  }

  const handleUpload = async (event) => {
    event.preventDefault()

    if (!selectedFile) {
      setError('Please choose a PDF file first.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    setIsLoading(true)
    setError('')
    setNotes('')
    setQuizQuestions([])
    setSelectedAnswers({})
    setQuizScore(null)
    setQuizSubmitted(false)
    setIsQuizModalOpen(false)

    try {
      const response = await fetch('http://localhost:5000/api/documents/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Unable to extract text from this PDF.')
      }

      const nextText = data.text || ''
      const nextNotes = data.notes || ''

      setExtractedText(nextText)
      setNotes(nextNotes)

      if (!nextText && !nextNotes) {
        throw new Error('The server did not return any extracted text or notes.')
      }

      if (!nextNotes) {
        setError('The PDF was processed, but no AI notes were returned.')
      }
    } catch (uploadError) {
      setError(uploadError.message || 'Something went wrong while uploading the PDF.')
      setExtractedText('')
      setNotes('')
      setQuizQuestions([])
      setSelectedAnswers({})
    } finally {
      setIsLoading(false)
    }
  }

  const handleAttemptQuiz = async () => {
    if (!extractedText.trim()) {
      setError('Upload a PDF with extracted text before attempting the quiz.')
      return
    }

    setIsQuizLoading(true)
    setError('')
    setQuizSubmitted(false)
    setQuizScore(null)
    setSelectedAnswers({})
    setIsQuizModalOpen(true)

    try {
      const response = await fetch('http://localhost:5000/api/documents/quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: extractedText }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Unable to generate quiz questions.')
      }

      const items = Array.isArray(data.questions)
        ? data.questions
        : Array.isArray(data.quiz)
          ? data.quiz
          : Array.isArray(data)
            ? data
            : []

      setQuizQuestions(items.map((question, index) => {
        const options = Array.isArray(question.options)
          ? question.options
          : Array.isArray(question.choices)
            ? question.choices
            : []

        const correctAnswer = question.correctAnswer ?? question.answer ?? question.correct ?? question.correctAnswerText ?? null

        return {
          question: question.question || question.prompt || question.text || `Question ${index + 1}`,
          options,
          correctAnswer,
        }
      }))
    } catch (quizError) {
      setError(quizError.message || 'Something went wrong while generating the quiz.')
      setQuizQuestions([])
      setIsQuizModalOpen(false)
    } finally {
      setIsQuizLoading(false)
    }
  }

  const handleAnswerChange = (questionIndex, value) => {
    setSelectedAnswers((current) => ({
      ...current,
      [questionIndex]: value,
    }))
  }

  const handleSubmitQuiz = () => {
    if (!quizQuestions.length) {
      setError('Generate a quiz before submitting your answers.')
      return
    }

    let score = 0

    quizQuestions.forEach((question, index) => {
      const selectedValue = selectedAnswers[index]
      const questionOptions = Array.isArray(question.options) ? question.options : []
      const correctChoice = question.correctAnswer

      if (
        normalizeChoice(selectedValue, questionOptions) === normalizeChoice(correctChoice, questionOptions)
      ) {
        score += 1
      }
    })

    setQuizSubmitted(true)
    setError('')
    setQuizScore(score)
  }

  const closeQuizModal = () => {
    setIsQuizModalOpen(false)
    setQuizSubmitted(false)
    setQuizScore(null)
  }

  return (
    <main className="app-shell">
      <section className="intro">
        <p className="eyebrow">PDF workspace</p>
        <h1>Smart Notes Generator</h1>
        <p className="subtitle">Turn a document into clear, searchable text.</p>
      </section>

      <form className="upload-panel" onSubmit={handleUpload}>
        <div className="file-picker">
          <label htmlFor="pdf-upload">Choose a PDF</label>
          <input id="pdf-upload" type="file" accept="application/pdf,.pdf" onChange={handleFileChange} />
          <p className="file-name">{selectedFile?.name || 'No file selected'}</p>
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Extracting...' : 'Upload & Extract'}
        </button>
      </form>

      {isLoading && <p className="status" role="status">Reading your PDF and generating notes...</p>}
      {error && <p className="error" role="alert">{error}</p>}

      <section className="text-panel" aria-live="polite">
        <div className="panel-heading">
          <h2>AI Summary</h2>
          {notes && <span>{notes.length.toLocaleString()} characters</span>}
        </div>
        <div className={`text-output${notes ? ' has-content' : ''}`}>
          {isLoading ? 'Generating summary...' : (notes || 'Your AI summary will appear here.')}
        </div>

        {notes && (
          <div className="quiz-actions">
            <button type="button" onClick={handleAttemptQuiz} disabled={isQuizLoading}>
              {isQuizLoading ? 'Generating Quiz...' : 'Attempt Quiz'}
            </button>
          </div>
        )}
      </section>

      <section className="text-panel" aria-live="polite">
        <button
          type="button"
          className="collapsible-toggle"
          onClick={() => setIsExtractedOpen((current) => !current)}
        >
          {isExtractedOpen ? 'Hide Extracted Text' : 'View Extracted Text'}
        </button>

        {isExtractedOpen && (
          <div className="collapsible-panel">
            <div className="panel-heading">
              <h2>Extracted text</h2>
              {extractedText && <span>{extractedText.length.toLocaleString()} characters</span>}
            </div>
            <div className={`text-output${extractedText ? ' has-content' : ''}`}>
              {extractedText || 'Your extracted text will appear here.'}
            </div>
          </div>
        )}
      </section>

      {isQuizModalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="quiz-modal">
            <div className="modal-header">
              <h2>Quiz</h2>
              <button type="button" className="close-button" onClick={closeQuizModal}>
                Close
              </button>
            </div>

            {quizQuestions.map((question, index) => {
              const selectedValue = selectedAnswers[index]
              const isCorrect =
                normalizeChoice(selectedValue, question.options) === normalizeChoice(question.correctAnswer, question.options)

              return (
                <div className="quiz-question" key={`${question.question}-${index}`}>
                  <p className="question-text">{index + 1}. {question.question}</p>

                  <div className="options-list">
                    {(Array.isArray(question.options) ? question.options : []).map((option, optionIndex) => (
                      <label className="option-row" key={`${question.question}-${optionIndex}`}>
                        <input
                          type="radio"
                          name={`question-${index}`}
                          value={option}
                          checked={selectedValue === option}
                          onChange={() => handleAnswerChange(index, option)}
                        />
                        <span>{option}</span>
                      </label>
                    ))}
                  </div>

                  {quizSubmitted && !isCorrect && (
                    <p className="quiz-feedback">
                      Correct answer: <strong>{question.correctAnswer}</strong>
                    </p>
                  )}
                </div>
              )
            })}

            <div className="quiz-actions modal-actions">
              <button type="button" onClick={handleSubmitQuiz}>Submit Quiz</button>
            </div>

            {quizSubmitted && quizScore !== null && (
              <p className="score-box" role="status">
                Your score: {quizScore} / {quizQuestions.length}
              </p>
            )}
          </div>
        </div>
      )}
    </main>
  )
}

export default App
