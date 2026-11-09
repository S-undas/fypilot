import json

from src.panel.categories import CATEGORIES, GENERIC_QUESTIONS
from src.panel.generate import generate_questions, parse_questions, similar


def test_parse_questions_handles_fences_junk_and_missing_marks():
    text = '```json\n["Why did you choose this design for the project", "Too short", 7, "How does the system handle failures in the field?"]\n```'
    assert parse_questions(text) == ["Why did you choose this design for the project?", "How does the system handle failures in the field?"]
    assert parse_questions("no list here") == [] and parse_questions("[not json]") == []
    long_q = " ".join(["word"] * 50) + "?"
    assert parse_questions(json.dumps([long_q, "Why did you choose this design for the project?"])) == ["Why did you choose this design for the project?"]


def test_similar_detects_near_duplicates():
    assert similar("What are the main limitations of your project?", "What are the main limitations of this project?")
    assert not similar("Why did you pick a Raspberry Pi board?", "How is user privacy protected in the application?")


DISTINCT = [
    "Why did you choose a Raspberry Pi instead of a cloud server for processing?",
    "Who exactly are the visually impaired users you interviewed?",
    "Which existing smart glasses products did you compare against?",
    "What is the single biggest contribution beyond prior published systems?",
    "How does the object detector decide when a detection is reliable enough to announce?",
    "Which training dataset shaped your final model weights?",
    "What accuracy figure supports your claim about real-time performance?",
    "How many participants took part in your usability sessions?",
    "Could the budget have covered a better camera module?",
    "What fallback exists when the wireless link drops?",
    "Where does the system fail in crowded outdoor scenes?",
    "Which feature did you drop because of time pressure?",
    "How is footage of bystanders stored and who can access it?",
    "Might the audio alerts distract a user crossing a road?",
]


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return json.dumps(DISTINCT[2 * (self.calls - 1): 2 * self.calls])


def test_generate_questions_one_call_per_category_with_n_each(index):
    out = generate_questions(index, CountingLLM(), None, "A", n=2)
    assert len(out) == 14 and [c for c, _ in out[:2]] == ["motivation", "motivation"]
    assert {c for c, _ in out} == set(CATEGORIES)


def test_generate_questions_drops_duplicates_across_categories(index):
    class SameLLM:
        def complete(self, messages):
            return json.dumps(["Why did you choose this particular approach for the system?"])

    out = generate_questions(index, SameLLM(), None, "A", n=1)
    assert len(out) == 1


def test_generic_questions_cover_every_category_three_times():
    assert set(GENERIC_QUESTIONS) == set(CATEGORIES) and all(len(v) == 3 for v in GENERIC_QUESTIONS.values())
