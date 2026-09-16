import unittest

from main import _quiz_payload_is_grounded

SOURCE_TEXT = (
    "Photosynthesis is the process plants use to convert sunlight into chemical energy. "
    "It occurs in chloroplasts and requires water, carbon dioxide, and light. "
    "The main products are glucose and oxygen."
)

VALID_QUIZ = {
    "questions": [
        {
            "question": "Which organelle is the site of photosynthesis?",
            "options": ["Chloroplast", "Nucleus", "Ribosome", "Mitochondrion"],
            "answer": 0,
            "explanation": "Photosynthesis occurs in chloroplasts, where light energy is converted into chemical energy."
        },
        {
            "question": "What are the reactants required for photosynthesis?",
            "options": ["Water and carbon dioxide", "Oxygen and glucose", "Nitrogen and sunlight", "ATP and starch"],
            "answer": 0,
            "explanation": "The process uses water, carbon dioxide, and light energy to make glucose."
        },
        {
            "question": "Which product is released during photosynthesis?",
            "options": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
            "answer": 0,
            "explanation": "Oxygen is released as a product when plants convert light energy into stored chemical energy."
        },
        {
            "question": "What is the main purpose of photosynthesis?",
            "options": ["To convert light energy into chemical energy", "To absorb oxygen from the air", "To break down glucose", "To release carbon dioxide"],
            "answer": 0,
            "explanation": "Photosynthesis stores energy from sunlight in glucose, which is the main purpose of the process."
        },
        {
            "question": "Which statement best describes the process?",
            "options": ["It stores solar energy in glucose", "It creates heat from sunlight", "It moves water into the nucleus", "It breaks down chlorophyll"],
            "answer": 0,
            "explanation": "The source states that photosynthesis converts sunlight into chemical energy, stored as glucose."
        },
    ]
}


class QuizValidationTest(unittest.TestCase):
    def test_valid_quiz_payload_passes_grounding_check(self):
        self.assertTrue(_quiz_payload_is_grounded(VALID_QUIZ, SOURCE_TEXT))


if __name__ == '__main__':
    unittest.main()
