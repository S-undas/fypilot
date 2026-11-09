from __future__ import annotations

import re
from dataclasses import dataclass

DOC_TYPES = {"F25_044": "rd", "F25_182": "rd", "F25_051": "dev", "F25_181": "dev"}


@dataclass(frozen=True)
class Requirement:
    key: str
    name: str
    title: str
    what: str
    query: str
    types: tuple[str, ...] = ("rd", "dev")
    pdf_only: bool = False

    def matches(self, heading: str) -> bool:
        return re.search(self.title, heading, re.I) is not None


REQUIREMENTS = [
    Requirement("abstract", "Abstract", r"^abstract$", "a short abstract summarizing the project", "abstract summary of the project"),
    Requirement("executive_summary", "Executive Summary", r"^executive summary$", "an executive summary of the project", "executive summary of the project"),
    Requirement("references", "References", r"^(references|bibliography)$", "a list of references or a bibliography",
                "references bibliography", pdf_only=True),
    Requirement("introduction", "Introduction", r"^introduction$", "an introduction chapter", "introduction of the project"),
    Requirement("vision", "Project Vision", r"project vision", "a project vision chapter", "project vision problem statement scope and goals"),
    Requirement("related", "Related applications or literature review", r"related|literature",
                "a chapter reviewing related applications or related literature", "related applications and literature review of existing work"),
    Requirement("srs", "Software Requirement Specifications", r"requirement",
                "a software requirement specifications chapter", "software requirements specification"),
    Requirement("methodology", "Proposed approach and methodology", r"approach|methodology",
                "a chapter describing the proposed approach and methodology of the research", "proposed approach and methodology", types=("rd",)),
    Requirement("design", "High-level and low-level design", r"design",
                "a high-level and low-level design chapter", "high-level and low-level design system architecture"),
    Requirement("implementation", "Implementation details", r"implementation",
                "implementation details of the prototype or software component that was built", "implementation details of the developed prototype or component"),
    Requirement("test_cases", "Test case design and test metrics", r"test case",
                "test case design and description, and test metrics", "test case design and description test metrics"),
    Requirement("experiments", "Experimental results and discussion", r"experiment|results",
                "a chapter presenting the results of the experiments conducted, with discussion", "experimental results and discussion", types=("rd",)),
    Requirement("user_manual", "User manual", r"user manual", "a user manual explaining how to use the system", "user manual how to use the application"),
    Requirement("conclusion", "Conclusion and future work", r"conclusion|work completed|future work",
                "a final conclusion of the whole report, with work completed and future work (the closing section of one chapter does not count)", "conclusion work completed and future work"),
]


def applicable(doc_type: str) -> list[Requirement]:
    return [r for r in REQUIREMENTS if doc_type in r.types]
