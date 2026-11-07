"""Domain models demonstrating classes, inheritance, and encapsulation."""

from __future__ import annotations

from typing import Mapping


class Person:
    """Base class containing common person information."""

    def __init__(self, name: str, email: str = "") -> None:
        cleaned_name = str(name).strip()
        if not cleaned_name:
            raise ValueError("name is required")
        self._name = cleaned_name
        self._email = str(email).strip()

    @property
    def name(self) -> str:
        return self._name

    @property
    def email(self) -> str:
        return self._email


class Student(Person):
    """Student model. Private ID and validated scores demonstrate encapsulation."""

    SUBJECTS = ("python", "sql", "statistics")

    def __init__(
        self,
        student_id: str,
        name: str,
        email: str = "",
        department: str = "IT",
        year: int = 4,
        scores: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(name=name, email=email)
        normalized_id = str(student_id).strip()
        if not normalized_id:
            raise ValueError("student_id is required")
        self.__student_id = normalized_id
        self._department = str(department or "Unknown").strip() or "Unknown"
        self._year = int(year)
        if not 1 <= self._year <= 4:
            raise ValueError("year must be between 1 and 4")

        incoming = dict(scores or {})
        validated_scores: dict[str, float] = {}
        for subject in self.SUBJECTS:
            raw_score = incoming.get(subject, 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{subject} score must be numeric") from exc
            if not 0 <= score <= 100:
                raise ValueError(f"{subject} score must be between 0 and 100")
            validated_scores[subject] = score
        self._scores = validated_scores

    @property
    def student_id(self) -> str:
        return self.__student_id

    @property
    def department(self) -> str:
        return self._department

    @property
    def year(self) -> int:
        return self._year

    @property
    def scores(self) -> dict[str, float]:
        # Return a copy so callers cannot mutate the internal score dictionary.
        return self._scores.copy()

    @property
    def average(self) -> float:
        return sum(self._scores.values()) / len(self._scores)

    @property
    def grade(self) -> str:
        average = self.average
        if average >= 90:
            return "A+"
        if average >= 80:
            return "A"
        if average >= 70:
            return "B"
        if average >= 60:
            return "C"
        if average >= 50:
            return "D"
        return "F"

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "Student":
        """Create a Student from a CSV/JSON-style dictionary."""
        lowered = {str(key).strip().lower(): value for key, value in record.items()}
        scores = {subject: lowered.get(subject, 0) for subject in cls.SUBJECTS}
        return cls(
            student_id=str(lowered.get("student_id", "")),
            name=str(lowered.get("name", "")),
            email=str(lowered.get("email", "") or ""),
            department=str(lowered.get("department", "IT") or "IT"),
            year=int(float(lowered.get("year", 4) or 4)),
            scores=scores,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert the object to a serializable dictionary."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "department": self.department,
            "year": self.year,
            **self.scores,
            "average": round(self.average, 2),
            "grade": self.grade,
        }
