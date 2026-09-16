import { useState } from 'react'
import './App.css'
const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000'
const HISTORY_STORAGE_KEY = 'smart-notes-history'

const readHistory = () => {
  try {
    const savedHistory = localStorage.getItem(HISTORY_STORAGE_KEY)
    const parsedHistory = savedHistory ? JSON.parse(savedHistory) : []
    return Array.isArray(parsedHistory) ? parsedHistory : []
  } catch {
    return []
  }
}

const writeHistory = (history) => {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history))
}

const readApiResponse = async (response) => {
  const responseText = await response.text()
  const contentType = response.headers.get('content-type') || ''
  let data

  if (contentType.includes('application/json')) {
    try {
      data = JSON.parse(responseText)
    } catch {
      data = { error: 'The server returned invalid JSON.' }
    }
  } else {
    const plainText = responseText.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
    data = { error: plainText || 'The server returned an unexpected response.' }
  }

  if (!response.ok) {
    const message = data.error || data.detail || `Request failed with status ${response.status}.`
    throw new Error(`Request failed (${response.status}): ${message}`)
  }

  return data
}

const normalizeChoice = (value, options = []) => {
  if (Number.isInteger(value)) return value

  const raw = typeof value === 'string' ? value.trim() : ''
  if (!raw) return ''

  const normalized = raw.toLowerCase()
  const letterMatch = normalized.match(/^[a-d]$/)

  if (letterMatch) {
    return letterMatch[0].charCodeAt(0) - 97
  }

  const optionIndex = options.findIndex((option) => option.toLowerCase() === normalized)
  return optionIndex >= 0 ? optionIndex : normalized
}

const downloadText = (content, filename) => {
  if (!content.trim()) return

  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

const renderNotes = (text) => {
  const rawLines = text.split(/\r?\n/)
  const lines = []
  rawLines.forEach((line) => {
    const trimmed = line.trim()
    const isStructure = /^(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)/.test(trimmed) || /^##?\s*(overview|key concepts|main components|important details|examples|quick revision|conclusion)/i.test(trimmed)
    if (!trimmed) {
      if (lines.length && lines[lines.length - 1] !== '') lines.push('')
    } else if (isStructure || !lines.length || lines[lines.length - 1] === '') {
      lines.push(trimmed)
    } else {
      lines[lines.length - 1] = `${lines[lines.length - 1]} ${trimmed}`
    }
  })
  const elements = []
  let paragraphLines = []
  let listItems = []
  let listType = null

  const flushParagraph = () => {
    if (paragraphLines.length) {
      elements.push(<p key={`paragraph-${elements.length}`}>{paragraphLines.join(' ')}</p>)
      paragraphLines = []
    }
  }

  const flushList = () => {
    if (listItems.length) {
      const List = listType === 'ordered' ? 'ol' : 'ul'
      elements.push(
        <List key={`list-${elements.length}`}>
          {listItems.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
        </List>,
      )
      listItems = []
      listType = null
    }
  }

  lines.forEach((line) => {
    const trimmedLine = line.trim()
    const headingMatch = trimmedLine.match(/^(#{1,6})\s+(.+)$/)
    const unorderedMatch = trimmedLine.match(/^[-*+]\s+(.+)$/)
    const orderedMatch = trimmedLine.match(/^\d+[.)]\s+(.+)$/)

    if (!trimmedLine) {
      flushParagraph()
      flushList()
    } else if (headingMatch) {
      flushParagraph()
      flushList()
      const Heading = `h${headingMatch[1].length}`
      elements.push(<Heading key={`heading-${elements.length}`}>{headingMatch[2]}</Heading>)
    } else if (unorderedMatch || orderedMatch) {
      flushParagraph()
      const nextListType = orderedMatch ? 'ordered' : 'unordered'
      if (listType && listType !== nextListType) flushList()
      listType = nextListType
      listItems.push((orderedMatch || unorderedMatch)[1])
    } else {
      flushList()
      paragraphLines.push(trimmedLine)
    }
  })

  flushParagraph()
  flushList()

  return elements
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [extractedText, setExtractedText] = useState('')
  const [notes, setNotes] = useState('')
  const [generatedNotes, setGeneratedNotes] = useState('')
  const [quizQuestions, setQuizQuestions] = useState([])
  const [selectedAnswers, setSelectedAnswers] = useState({})
  const [quizScore, setQuizScore] = useState(null)
  const [isExtractedOpen, setIsExtractedOpen] = useState(false)
  const [isQuizModalOpen, setIsQuizModalOpen] = useState(false)
  const [quizSubmitted, setQuizSubmitted] = useState(false)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isQuizLoading, setIsQuizLoading] = useState(false)
  const [isNotesLoading, setIsNotesLoading] = useState(false)
  const [history, setHistory] = useState(readHistory)
  const [activeHistoryId, setActiveHistoryId] = useState(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null
    setSelectedFile(file)
    setExtractedText('')
    setNotes('')
    setGeneratedNotes('')
    setQuizQuestions([])
    setSelectedAnswers({})
    setQuizScore(null)
    setQuizSubmitted(false)
    setIsQuizModalOpen(false)
    setActiveHistoryId(null)
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
    setGeneratedNotes('')
    setQuizQuestions([])
    setSelectedAnswers({})
    setQuizScore(null)
    setQuizSubmitted(false)
    setIsQuizModalOpen(false)

    try {
      const response = await fetch(`${BACKEND_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await readApiResponse(response)

      const nextText = data.text || ''
      const nextNotes = data.notes || ''

      setExtractedText(nextText)
      setNotes(nextNotes)

      if (nextNotes.trim()) {
        const historyEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          filename: selectedFile.name,
          summary: nextNotes,
          notes: '',
          extractedText: nextText,
          createdAt: new Date().toISOString(),
        }
        const nextHistory = [historyEntry, ...history]
        setHistory(nextHistory)
        writeHistory(nextHistory)
        setActiveHistoryId(historyEntry.id)
      }

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

  const handleGenerateNotes = async () => {
    if (!extractedText.trim()) {
      setError('Upload a PDF with extracted text before generating notes.')
      return
    }

    setIsNotesLoading(true)
    setError('')
    setGeneratedNotes('')

    try {
      const response = await fetch(`${BACKEND_URL}/api/documents/generate-notes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: extractedText }),
      })
      const data = await readApiResponse(response)

      if (typeof data.notes !== 'string' || !data.notes.trim()) {
        throw new Error('The server did not return any generated notes.')
      }

      setGeneratedNotes(data.notes)

      const currentHistory = history.find((entry) => entry.id === activeHistoryId)
      const nextHistory = currentHistory
        ? history.map((entry) => (
          entry.id === activeHistoryId ? { ...entry, notes: data.notes } : entry
        ))
        : [{
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          filename: selectedFile?.name || 'Untitled PDF',
          summary: notes,
          notes: data.notes,
          extractedText,
          createdAt: new Date().toISOString(),
        }, ...history]

      setHistory(nextHistory)
      writeHistory(nextHistory)
      if (!currentHistory) setActiveHistoryId(nextHistory[0].id)
    } catch (notesError) {
      setError(notesError.message || 'Something went wrong while generating notes.')
    } finally {
      setIsNotesLoading(false)
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
      const response = await fetch(`${BACKEND_URL}/api/documents/quiz`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: extractedText }),
      })
      const data = await readApiResponse(response)

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

        const rawAnswer = question.answer ?? question.correctAnswer ?? question.correct ?? question.correctAnswerText ?? null

        return {
          question: question.question || question.prompt || question.text || `Question ${index + 1}`,
          options,
          correctAnswer: normalizeChoice(rawAnswer, options),
          explanation: question.explanation || '',
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

  const openHistoryEntry = (entry) => {
    setActiveHistoryId(entry.id)
    setNotes(entry.summary || '')
    setGeneratedNotes(entry.notes || '')
    setExtractedText(entry.extractedText || '')
    setQuizQuestions([])
    setSelectedAnswers({})
    setQuizScore(null)
    setQuizSubmitted(false)
    setIsQuizModalOpen(false)
    setError('')
    setIsExtractedOpen(false)
    setIsSidebarOpen(false)
  }

  const deleteHistoryEntry = (entryId) => {
    const entry = history.find((item) => item.id === entryId)
    if (!entry || !window.confirm(`Delete history for ${entry.filename}?`)) return

    const nextHistory = history.filter((entry) => entry.id !== entryId)
    setHistory(nextHistory)
    writeHistory(nextHistory)
    if (activeHistoryId === entryId) {
      setActiveHistoryId(null)
      setNotes('')
      setGeneratedNotes('')
    }
  }

  return (
    <div className="app-layout">
      {isSidebarOpen && (
        <button
          type="button"
          className="history-backdrop"
          aria-label="Close history"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <aside className={`history-sidebar${isSidebarOpen ? '' : ' collapsed'}`} aria-hidden={!isSidebarOpen}>
        <p className="sidebar-brand">MyNotes</p>
        <div className="history-sidebar-header">
          <div>
            <p className="eyebrow">Saved workspace</p>
            <h2>History</h2>
          </div>
          <span>{history.length}</span>
        </div>

        {!history.length ? (
          <p className="history-sidebar-empty">No saved history yet.</p>
        ) : (
          <div className="history-list">
            {history.map((entry) => (
              <div className="history-item" key={entry.id}>
                <button
                  type="button"
                  className={`history-entry${activeHistoryId === entry.id ? ' active' : ''}`}
                  onClick={() => openHistoryEntry(entry)}
                >
                  <strong className="history-item-title">{entry.filename}</strong>
                  <span>{new Date(entry.createdAt).toLocaleString()}</span>
                </button>
                <button
                  type="button"
                  className="history-delete"
                  aria-label={`Delete ${entry.filename} from history`}
                  title="Delete history item"
                  onClick={() => deleteHistoryEntry(entry.id)}
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}
      </aside>

      <main className={`app-shell main-content${isSidebarOpen ? '' : ' sidebar-collapsed'}`}>
      <button
        type="button"
        className="history-toggle"
        aria-label={isSidebarOpen ? 'Close history' : 'Open history'}
        aria-expanded={isSidebarOpen}
        onClick={() => setIsSidebarOpen((current) => !current)}
      >
        <span />
        <span />
        <span />
      </button>

      <section className="intro">
        <p className="eyebrow">PDF workspace</p>
        <h1>Smart Notes Generator</h1>
        <p className="subtitle">Turn your documents into clear notes, concise summaries, and interactive quizzes.</p>
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

        <div className="quiz-actions">
          <button type="button" onClick={() => downloadText(notes, 'summary.txt')} disabled={!notes.trim()}>
            Download Summary
          </button>
        </div>

        {extractedText && (
          <div className="quiz-actions">
            <button type="button" onClick={handleGenerateNotes} disabled={isNotesLoading}>
              {isNotesLoading ? 'Generating Notes...' : 'Generate Notes'}
            </button>
          </div>
        )}

        {notes && (
          <div className="quiz-actions">
            <button type="button" onClick={handleAttemptQuiz} disabled={isQuizLoading}>
              {isQuizLoading ? 'Generating Quiz...' : 'Attempt Quiz'}
            </button>
            </div>
        )}
      </section>

      <section className="text-panel" aria-live="polite">
        <div className="panel-heading">
          <h2>Notes</h2>
          {generatedNotes && <span>{generatedNotes.length.toLocaleString()} characters</span>}
        </div>
        <div className={`text-output notes-output${generatedNotes ? ' has-content' : ''}`}>
          {isNotesLoading
            ? <p className="notes-placeholder">Generating notes...</p>
            : generatedNotes
              ? <div className="notes-content">{renderNotes(generatedNotes)}</div>
              : <p className="notes-placeholder">Your generated notes will appear here.</p>}
        </div>

        <div className="quiz-actions">
          <button type="button" onClick={() => downloadText(generatedNotes, 'notes.txt')} disabled={!generatedNotes.trim()}>
            Download Notes
          </button>
        </div>
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
                          value={optionIndex}
                          checked={selectedValue === optionIndex}
                          onChange={() => handleAnswerChange(index, optionIndex)}
                        />
                        <span>{option}</span>
                      </label>
                    ))}
                  </div>

                  {quizSubmitted && (
                    <>
                      {!isCorrect && (
                        <p className="quiz-feedback">
                          Correct answer: <strong>{question.options[question.correctAnswer] || 'Unavailable'}</strong>
                        </p>
                      )}
                      {question.explanation && (
                        <p className="quiz-feedback">Explanation: {question.explanation}</p>
                      )}
                    </>
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
    </div>
  )
}

export default App
