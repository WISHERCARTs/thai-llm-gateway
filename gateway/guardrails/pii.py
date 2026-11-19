"""Find and redact personal data in Thai and English text.

Detected values are swapped for placeholders such as [THAI_ID_1] before the prompt leaves the
gateway, so the model provider and the traces never see them. The mapping stays in memory for
the one request, so the caller's own values can be put back into the answer.

Detectors are rules, not a model, so their limits are known:
- Thai ID numbers must pass the official mod-11 checksum, which keeps random 13-digit numbers out.
- Card numbers must pass the Luhn check.
- Bank account numbers are only taken in the dashed 3-1-5-1 form or next to a word like "บัญชี",
  because a bare 10-digit number is usually a phone number or an order number.
- Thai names are only found after a title (นาย, นาง, นางสาว, ด.ช., ด.ญ., น.ส.). "คุณ" is left out
  on purpose: it is also the everyday word for "you".
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

THAI_TITLES = ("นางสาว", "นาย", "นาง", "น.ส.", "ด.ช.", "ด.ญ.", "เด็กชาย", "เด็กหญิง")
EN_TITLES = r"(?:Mr|Mrs|Ms|Miss|Dr)\.?"
# Words that end a name even though they are not in PyThaiNLP's stop word list.
NAME_STOP = {"โทร", "เบอร์", "อีเมล", "อายุ", "บ้าน", "เลข", "บัญชี", "ที่อยู่", "เกิด", "ชื่อ", "นามสกุล"}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
THAI_ID = re.compile(r"(?<![\d-])(\d)[- ]?(\d{4})[- ]?(\d{5})[- ]?(\d{2})[- ]?(\d)(?![\d-])")
PHONE = re.compile(
    r"(?<![\d+])(?:\+66[- ]?|0)(?:[689]\d[- ]?\d{3}[- ]?\d{4}|2[- ]?\d{3}[- ]?\d{4}|[3-7]\d[- ]?\d{3}[- ]?\d{3})(?!\d)"
)
CARD = re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)")
BANK_DASHED = re.compile(r"(?<![\d-])\d{3}-\d-\d{5}-\d(?![\d-])")
BANK_CONTEXT = re.compile(r"(?:บัญชี|account(?: number| no\.?)?|acct)\D{0,20}?(\d[\d -]{8,15}\d)", re.IGNORECASE)
EN_NAME = re.compile(rf"\b{EN_TITLES}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
THAI_CHAR = re.compile("[\u0e00-\u0e7f]")


@dataclass
class Finding:
    kind: str
    start: int
    end: int
    value: str


@dataclass
class Redaction:
    text: str
    findings: list[Finding] = field(default_factory=list)


def thai_id_valid(digits: str) -> bool:
    if len(digits) != 13 or not digits.isdigit() or digits[0] == "0":
        return False
    total = sum(int(d) * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - total % 11) % 10 == int(digits[12])


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    from pythainlp.corpus import thai_stopwords

    return frozenset(thai_stopwords()) | NAME_STOP


def _thai_word_run(text: str, start: int) -> int:
    """End of the Thai name that starts at `start`: stop at a space or at a stop word."""
    from pythainlp.tokenize import word_tokenize

    end = start
    while end < len(text) and THAI_CHAR.match(text[end]):
        end += 1
    if end == start:
        return start
    pos = start
    for tok in word_tokenize(text[start:end], engine="newmm", keep_whitespace=False):
        if tok in _stopwords() and pos > start:
            break
        pos += len(tok)
    return pos


def _surname_end(text: str, start: int) -> int:
    """Surnames are written as one unspaced run ("วงศ์ใหญ่"), so take the run up to the next space.

    A very long run means the surname is glued to the rest of the sentence; then cut at a stop word.
    """
    end = start
    while end < len(text) and THAI_CHAR.match(text[end]):
        end += 1
    if end - start <= 20:
        return end
    return _thai_word_run(text, start)


def _is_title_word(text: str, start: int, title: str) -> bool:
    """True when the title stands as its own word: "นาย" in "นายสมชาย", not in "นายกรัฐมนตรี"."""
    from pythainlp.tokenize import word_tokenize

    if "." in title:
        return True
    toks = word_tokenize(text[start : start + 40], engine="newmm", keep_whitespace=False)
    return bool(toks) and toks[0] == title


def _thai_names(text: str) -> list[Finding]:
    out = []
    pattern = "|".join(re.escape(t) for t in THAI_TITLES)
    for m in re.finditer(pattern, text):
        if not _is_title_word(text, m.start(), m.group()):
            continue
        first_end = _thai_word_run(text, m.end())
        if first_end == m.end():
            continue
        end = first_end
        # A surname follows after a single space.
        if end + 1 < len(text) and text[end] == " " and THAI_CHAR.match(text[end + 1]):
            surname_end = _surname_end(text, end + 1)
            word = text[end + 1 : surname_end]
            if surname_end > end + 1 and word not in _stopwords():
                end = surname_end
        out.append(Finding("PERSON", m.end(), end, text[m.end() : end]))
    return out


def find_pii(text: str) -> list[Finding]:
    found: list[Finding] = []
    for m in EMAIL.finditer(text):
        found.append(Finding("EMAIL", m.start(), m.end(), m.group()))
    for m in THAI_ID.finditer(text):
        if thai_id_valid("".join(m.groups())):
            found.append(Finding("THAI_ID", m.start(), m.end(), m.group()))
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if luhn_valid(digits):
            found.append(Finding("CARD", m.start(), m.end(), m.group()))
    for m in BANK_DASHED.finditer(text):
        found.append(Finding("BANK_ACCOUNT", m.start(), m.end(), m.group()))
    for m in BANK_CONTEXT.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) in (10, 12):
            found.append(Finding("BANK_ACCOUNT", m.start(1), m.end(1), m.group(1)))
    for m in PHONE.finditer(text):
        found.append(Finding("PHONE", m.start(), m.end(), m.group()))
    for m in EN_NAME.finditer(text):
        found.append(Finding("PERSON", m.start(1), m.end(1), m.group(1)))
    if THAI_CHAR.search(text):
        found.extend(_thai_names(text))
    return _drop_overlaps(found)


def _drop_overlaps(found: list[Finding]) -> list[Finding]:
    """Keep the earliest, then longest, finding where spans overlap (a Thai ID beats a phone)."""
    priority = {"EMAIL": 0, "THAI_ID": 1, "CARD": 2, "BANK_ACCOUNT": 3, "PHONE": 4, "PERSON": 5}
    found.sort(key=lambda f: (f.start, priority[f.kind], -(f.end - f.start)))
    kept: list[Finding] = []
    for f in found:
        if kept and f.start < kept[-1].end:
            continue
        kept.append(f)
    return kept


class Vault:
    """Placeholder <-> value mapping for one request."""

    def __init__(self):
        self.by_value: dict[tuple[str, str], str] = {}
        self.by_token: dict[str, str] = {}
        self.counts: dict[str, int] = {}

    def token_for(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self.by_value:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            token = f"[{kind}_{self.counts[kind]}]"
            self.by_value[key] = token
            self.by_token[token] = value
        return self.by_value[key]

    def redact(self, text: str) -> Redaction:
        findings = find_pii(text)
        if not findings:
            return Redaction(text)
        parts, pos = [], 0
        for f in findings:
            parts.append(text[pos : f.start])
            parts.append(self.token_for(f.kind, f.value))
            pos = f.end
        parts.append(text[pos:])
        return Redaction("".join(parts), findings)

    def restore(self, text: str) -> str:
        if not self.by_token:
            return text
        return re.sub(r"\[[A-Z_]+_\d+\]", lambda m: self.by_token.get(m.group(), m.group()), text)

    @property
    def total(self) -> int:
        return len(self.by_token)


def mask_pii(text: str) -> tuple[str, int]:
    """Replace personal data with [REDACTED_<KIND>]. Used on answers; these are never restored."""
    findings = find_pii(text)
    if not findings:
        return text, 0
    parts, pos = [], 0
    for f in findings:
        parts.append(text[pos : f.start])
        parts.append(f"[REDACTED_{f.kind}]")
        pos = f.end
    parts.append(text[pos:])
    return "".join(parts), len(findings)
