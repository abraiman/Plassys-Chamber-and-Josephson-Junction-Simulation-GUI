"""
resistance_parser.py -- turns a raw speech-to-text transcript into a
committable measurement value, or None if the utterance doesn't look
like a measurement at all (so ambient chatter gets silently dropped
rather than corrupting a cell).

Two engines will feed this differently: Whisper normally RENDERS
spoken numbers as digits already ("forty one point eight" -> "41.8"),
while Vosk's default English models typically leave numbers spelled
out as words. This parser accepts BOTH forms so it works with either
engine, or a mix (e.g. Vosk on the live path, Whisper as a slower
high-accuracy re-check).

No dependencies beyond the standard library -- fully testable without
an audio backend or a downloaded model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --- word-number vocabulary -------------------------------------------------

_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULTIPLIERS = {"hundred": 100, "thousand": 1000}

# Words that separate the whole part from the fractional part when a
# decimal is spoken aloud (e.g. "twelve point four").
_DECIMAL_WORDS = {"point", "decimal"}

# Special (non-numeric) measurement outcomes a multimeter can show.
# "OL" / "overload" is the multimeter's own way of displaying an open
# circuit -- folded into the same "open" outcome used elsewhere in this
# workflow, rather than a third distinct category.
_SPECIAL_OUTCOMES = {
    "open": "open", "shorted": "short", "short": "short",
    "overload": "open", "o l": "open", "ol": "open",
}

# Command words recognized alongside measurements so the user can
# correct a mistake or skip a cell without ever touching the keyboard.
COMMAND_WORDS = {
    "undo": "undo", "redo": "redo", "skip": "skip", "next": "advance",
    "repeat": "repeat", "cancel": "cancel", "stop": "stop", "pause": "pause",
}

_UNIT_TO_KOHM = {
    # normalize everything to kOhm, matching the unit convention used
    # throughout typical resistance-measurement reference spreadsheets
    # ("(kOhm)").
    "ohm": 0.001, "ohms": 0.001,
    "kilohm": 1.0, "kilohms": 1.0, "kohm": 1.0, "kohms": 1.0, "k": 1.0,
    "megaohm": 1000.0, "megaohms": 1000.0, "meg": 1000.0, "megs": 1000.0,
    "mohm": 1000.0, "mohms": 1000.0,  # "M-ohm" (mega), NOT milliohm --
    # milliohms essentially never come up for room-temp JJ resistance
    # measurements in this range, so "m" alone is deliberately NOT
    # mapped here (too ambiguous with "meg" heard as "m") -- always
    # spell out "milliohm(s)" if that's ever actually needed.
    "milliohm": 0.000001, "milliohms": 0.000001,
}

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


@dataclass
class ParsedMeasurement:
    kind: str                       # "value" | "special" | "command"
    value_kohm: Optional[float] = None   # set when kind == "value"
    special: Optional[str] = None        # set when kind == "special" ("open" | "short")
    command: Optional[str] = None        # set when kind == "command"
    raw_text: str = ""


def _words_to_number(tokens: list) -> Optional[float]:
    """Converts a run of number-word tokens (already split on the
    decimal word, if any) into a float. Handles "five thousand one
    hundred twenty five" -> 5125, "forty one" -> 41, "twelve" -> 12.
    Returns None if the tokens don't form a recognizable number."""
    if not tokens:
        return None
    total = 0
    current = 0
    saw_any = False
    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]
            saw_any = True
        elif tok in _TENS:
            current += _TENS[tok]
            saw_any = True
        elif tok in _MULTIPLIERS:
            mult = _MULTIPLIERS[tok]
            current = (current or 1) * mult
            if mult == 1000:
                total += current
                current = 0
            saw_any = True
        elif tok == "and":
            continue
        else:
            return None  # unrecognized token -- not a clean number-word run
    if not saw_any:
        return None
    return float(total + current)


def _parse_decimal_from_words(text: str) -> Optional[float]:
    """Spelled-out numbers, with an optional spelled-out decimal point
    ("twelve point four" -> 12.4). Digit-form numbers are handled
    separately in parse_resistance_utterance via _NUMBER_RE first --
    this is the fallback for engines (Vosk, by default) that leave
    numbers as words instead of normalizing them to digits."""
    words = text.split()
    for i, w in enumerate(words):
        if w in _DECIMAL_WORDS:
            whole = _words_to_number(words[:i])
            frac_tokens = words[i + 1:]
            frac_digits = [str(_ONES[t]) for t in frac_tokens if t in _ONES]
            if whole is None or len(frac_digits) != len(frac_tokens) or not frac_digits:
                return None
            return float(f"{int(whole)}.{''.join(frac_digits)}")
    return _words_to_number(words)


def parse_resistance_utterance(text: str) -> Optional[ParsedMeasurement]:
    """The single entry point: takes one already-transcribed utterance
    (a segmenter/VAD has already decided this was one spoken phrase)
    and returns a ParsedMeasurement, or None if it doesn't look like a
    measurement, a special outcome, or a recognized command at all --
    the caller's cue to silently discard it (background chatter, a
    mis-triggered segment) rather than writing anything into the table."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    lowered = raw.lower().strip(" .!?")
    lowered = re.sub(r"[^\w\s.,]", "", lowered)  # drop stray punctuation Whisper sometimes adds

    if lowered in COMMAND_WORDS:
        return ParsedMeasurement(kind="command", command=COMMAND_WORDS[lowered], raw_text=raw)

    if lowered in _SPECIAL_OUTCOMES:
        return ParsedMeasurement(kind="special", special=_SPECIAL_OUTCOMES[lowered], raw_text=raw)

    # unit word, if any -- searched for anywhere in the utterance so
    # "twelve point four kilohms" and "kilohms twelve point four" both work
    unit_scale = 1.0
    for unit_word, scale in _UNIT_TO_KOHM.items():
        if re.search(rf"\b{re.escape(unit_word)}\b", lowered):
            unit_scale = scale
            break

    # 1. digit form first (what Whisper normally produces: "41.8", "5,125")
    m = _NUMBER_RE.search(lowered)
    if m:
        digits = m.group(0).replace(",", "")
        try:
            value = float(digits)
            return ParsedMeasurement(kind="value", value_kohm=value * unit_scale, raw_text=raw)
        except ValueError:
            pass

    # 2. spelled-out word form fallback (what Vosk's default models normally produce)
    # strip the unit words out first so they don't confuse the number-word parser
    word_only = lowered
    for unit_word in _UNIT_TO_KOHM:
        word_only = re.sub(rf"\b{re.escape(unit_word)}\b", "", word_only)
    value = _parse_decimal_from_words(word_only.strip())
    if value is not None:
        return ParsedMeasurement(kind="value", value_kohm=value * unit_scale, raw_text=raw)

    return None
