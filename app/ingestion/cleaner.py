"""Light text cleaning. We deliberately keep symbols like -> ^ = ( ) / so that
formulas and functional dependencies survive."""
import re

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "–": "-", "—": "-",
             "−": "-", "’": "'", "“": '"', "”": '"', " ": " "}


def clean_text(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", line, re.I):
            continue  # drop blank lines and bare page numbers
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
