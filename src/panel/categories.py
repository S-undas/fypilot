from __future__ import annotations

CATEGORIES: dict[str, dict[str, str]] = {
    "motivation": {
        "definition": "why the project exists: the problem, who has it, and why it matters",
        "query": "motivation problem statement why this project is needed goals objectives",
    },
    "novelty": {
        "definition": "what is new or different compared with existing work and what the contribution is",
        "query": "novelty contribution compared with related work existing solutions differences literature review",
    },
    "methodology": {
        "definition": "how the project was done: approach, design choices, algorithms and architecture",
        "query": "methodology approach design architecture algorithm how the system works",
    },
    "evaluation": {
        "definition": "how success was measured: metrics, experiments, tests and results",
        "query": "evaluation results experiments metrics accuracy testing validation",
    },
    "feasibility": {
        "definition": "whether the scope was achievable: constraints, resources, time and risks",
        "query": "feasibility scope constraints resources timeline risks hardware requirements",
    },
    "limitations": {
        "definition": "what does not work well, what was left out, and what could be improved",
        "query": "limitations challenges weaknesses future work assumptions",
    },
    "ethics": {
        "definition": "privacy, security, safety, misuse and fairness concerns",
        "query": "ethics privacy security safety data protection user consent",
    },
}

GENERIC_QUESTIONS: dict[str, list[str]] = {
    "motivation": [
        "Why did you choose this project, and what problem does it solve?",
        "Who are the intended users, and why do they need this?",
        "What happens if this problem is left unsolved?",
    ],
    "novelty": [
        "What is new in your approach compared with existing solutions?",
        "How is your project different from the related work you reviewed?",
        "What is your main contribution?",
    ],
    "methodology": [
        "Why did you choose this methodology over the alternatives?",
        "Walk us through how your system works from input to output.",
        "What are the key design decisions and why did you make them?",
    ],
    "evaluation": [
        "How did you evaluate your system, and which metrics did you use?",
        "What results did you obtain, and how do they compare with a baseline?",
        "How do you know your results are reliable?",
    ],
    "feasibility": [
        "How did you decide the scope was achievable in the available time?",
        "What technical, hardware or data constraints did you face, and how did you handle them?",
        "What risks threatened the project, and how were they managed?",
    ],
    "limitations": [
        "What are the main limitations of your project?",
        "In what situations does your system fail or perform poorly?",
        "What would you improve with more time?",
    ],
    "ethics": [
        "What ethical or privacy concerns does your project raise?",
        "How do you handle user data and security?",
        "Could your system cause harm or be misused, and how did you address that?",
    ],
}
