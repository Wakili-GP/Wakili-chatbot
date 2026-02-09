# ============================================
# file: app/utils.py
# ============================================
from __future__ import annotations

import re
from typing import List, Set


_ARABIC_STOPWORDS: Set[str] = {
    "في", "من", "على", "إلى", "عن", "أن", "هذا", "هذه", "التي", "الذي",
    "ما", "لا", "أو", "و", "كل", "ذلك", "بين", "كان", "قد", "هو", "هي",
    "لم", "بل", "ثم", "إذا", "حتى", "لكن", "منه", "فيه", "عند", "له",
    "بها", "لها", "منها", "فيها", "التى", "الذى", "ولا", "وفى", "كما",
    "تلك", "هنا", "أي", "دون", "ليس", "إلا", "أما", "مع", "عليه",
}


def arabic_tokenize(text: str) -> List[str]:
    """
    Arabic tokenization used by BM25 + metadata index.
    Removes diacritics, keeps Arabic letters/spaces, removes stopwords.
    """
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)      # strip tashkeel
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)        # keep Arabic only
    tokens = text.split()
    return [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) > 1]


def convert_to_eastern_arabic(text: str) -> str:
    """Converts 0123456789 to ٠١٢٣٤٥٦٧٨٩"""
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    return text.translate(str.maketrans(western, eastern))

