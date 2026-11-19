"""Score text for prompt-injection attempts, in Thai and English.

A rule-based first line: fast, no model call, explainable (each hit names the rule). It catches
the common direct attacks; a paraphrased or novel attack can get through, so the gateway also
checks answers on the way out (see output.py). The eval set in evals/guardrails measures both
the catch rate and the false alarms on normal questions.
"""

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
# Thai rules allow some words in between: Thai has no spaces between words.
GAP = r".{0,25}?"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    weight: float


def _r(name: str, pattern: str, weight: float) -> Rule:
    return Rule(name, re.compile(pattern, re.IGNORECASE | re.DOTALL), weight)


RULES = [
    # English
    _r(
        "ignore_instructions",
        r"\b(ignore|disregard|forget|skip|override)\b.{0,30}\b(previous|prior|above|earlier|preceding|all|any|your)\b"
        r".{0,20}\b(instructions?|rules|prompts?|directions|guidelines|messages)\b",
        0.9,
    ),
    _r(
        "forget_everything",
        r"\b(forget|ignore|disregard)\s+(everything|all of that|all that)\s+(above|before|so far|you were told)",
        0.85,
    ),
    _r(
        "reveal_system_prompt",
        r"\b(reveal|show|print|repeat|output|tell me|what (is|are)|display|leak|dump)\b.{0,30}"
        r"\b(system|initial|hidden|original|secret|developer)\s+(prompt|instructions?|message|rules)\b",
        0.85,
    ),
    _r(
        "persona_jailbreak",
        r"\b(you are now|from now on,? you are|act as|pretend to be|roleplay as)\b.{0,40}"
        r"\b(dan|jailbr\w*|unrestricted|unfiltered|uncensored|evil|no (rules|limits|restrictions))\b",
        0.85,
    ),
    _r("do_anything_now", r"\b(do anything now|developer mode|god mode|jailbreak mode)\b", 0.8),
    _r(
        "no_restrictions",
        r"\b(pretend|imagine|assume)\b.{0,30}\b(no|without)\b.{0,15}\b(rules|restrictions|filters|guidelines|limits)\b",
        0.75,
    ),
    _r(
        "bypass_safety",
        r"\b(bypass|disable|turn off|circumvent|evade)\b.{0,25}\b(safety|guardrails?|filters?|restrictions|moderation|content policy)\b",
        0.6,
    ),
    _r(
        "fake_role_marker",
        r"(<\|im_start\|>|<\|system\|>|\[/?INST\]|</?system>|^\s*(system|assistant)\s*:|###\s*(system|instruction))",
        0.7,
    ),
    _r("new_instructions", r"\b(new|updated|real|actual)\s+(instructions?|task|system prompt)\s*:", 0.6),
    # Thai
    _r(
        "th_ignore_instructions",
        rf"(ลืม|ละเว้น|เพิกเฉย|ไม่ต้องสนใจ|ไม่ต้องทำตาม|อย่าทำตาม|ข้าม|ยกเลิก){GAP}"
        rf"(คำสั่ง|กฎ|ข้อกำหนด|คำแนะนำ|ข้อความ)(ทั้งหมด|เดิม|ก่อนหน้า|ข้างบน|ด้านบน|ที่ผ่านมา|ที่ได้รับ|ของระบบ)",
        0.9,
    ),
    _r(
        "th_reveal_system_prompt",
        rf"(บอก|แสดง|เปิดเผย|พิมพ์|ขอดู|ส่ง|เล่า|ทวน){GAP}"
        rf"(system ?prompt|ซิสเต็ม ?พรอมต์|พรอมต์ระบบ|คำสั่งระบบ|คำสั่งตั้งต้น|คำสั่งลับ|คำสั่งแรก|คำสั่งที่ได้รับ)",
        0.85,
    ),
    _r(
        "th_persona_jailbreak",
        rf"(ตอนนี้คุณคือ|ต่อจากนี้คุณคือ|จากนี้ไปคุณ|สวมบทบาท|แกล้งทำเป็น|สมมติว่าคุณ){GAP}"
        rf"(ไม่มีข้อจำกัด|ไร้ข้อจำกัด|ไม่มีกฎ|ไร้กฎ|ไม่ต้องทำตามกฎ|ไม่มีการเซ็นเซอร์|ทำได้ทุกอย่าง)",
        0.85,
    ),
    _r("th_developer_mode", r"(โหมดนักพัฒนา|โหมดไร้ข้อจำกัด|โหมดเจลเบรก|เจลเบรก)", 0.75),
    _r("th_bypass_safety", rf"(ข้าม|หลบ|ปิด|เลี่ยง|ทะลุ){GAP}(ระบบ)?(ความปลอดภัย|ตัวกรอง|ข้อจำกัด|การตรวจสอบ|การป้องกัน)", 0.6),
    # Encoded payloads
    _r("encoded_payload", r"\b(decode|base64|rot13)\b.{0,40}[A-Za-z0-9+/]{40,}={0,2}", 0.5),
]


@dataclass
class InjectionResult:
    score: float
    rules: list[str]

    def blocked(self, threshold: float) -> bool:
        return self.score >= threshold


def normalize(text: str) -> str:
    # NFKC folds full-width and look-alike letters, but it also splits Thai sara am (ำ) into two
    # code points; put it back so Thai patterns still match.
    text = unicodedata.normalize("NFKC", text).replace("\u0e4d\u0e32", "\u0e33")
    text = ZERO_WIDTH.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def _decoded_payloads(text: str) -> list[str]:
    """Decode base64 blobs so an encoded attack is scored like a plain one."""
    out = []
    for blob in BASE64_BLOB.findall(text)[:5]:
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded.isprintable():
            out.append(decoded)
    return out


def score_injection(text: str) -> InjectionResult:
    norm = normalize(text)
    texts = [norm, *(normalize(d) for d in _decoded_payloads(norm))]
    hits = [r for r in RULES if any(r.pattern.search(t) for t in texts)]
    keep = 1.0
    for r in hits:
        keep *= 1 - r.weight
    return InjectionResult(round(1 - keep, 3), [r.name for r in hits])
