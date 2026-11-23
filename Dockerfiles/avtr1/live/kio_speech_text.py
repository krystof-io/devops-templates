# Speech preparation: turn text written to be READ (markdown, as an assistant
# streams it) into pieces of text to be SPOKEN, one sentence at a time.
#
# Not upstream code. Copied to /opt/avtr-1/src/ at image build. Standard library
# only, so it can be unit-tested anywhere: python3 live/test_speech_text.py
#
# Why it exists: when another program supplies the words (the Hermes HUD sends
# Hermes' reply text to /speak-stream), the transcript keeps the original text
# and only the copy that goes to the voice is cleaned. The rules mirror the
# HUD's own speakable() (hermes-hud src/data/hermes/mapping.ts) so a reply
# sounds the same whichever side cleans it:
#
#   dropped entirely   MEDIA: lines, [[audio_as_voice]] and other [[...]] directives,
#                      fenced code blocks, table rule lines, images, emoji
#   unwrapped          links (the label is kept), inline code, **bold**, _italic_,
#                      headings, list bullets and numbers, table cells
#   replaced           bare URLs -> "a link"
#
# Text arrives in arbitrary pieces. Markdown is line-structured, so complete
# lines are classified first; the unfinished last line is only read ahead when
# it is plainly prose, so speech can start before a long paragraph has finished
# arriving without ever speaking half of a code fence or a MEDIA: line.

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")

# A short fragment ("Yes.", "Hi!") waits and is spoken with what follows.
_MIN_PIECE = 12


def split_sentences(text: str, *, final: bool) -> tuple[list[str], str]:
    """Return (complete sentences, remainder)."""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end(1)
        piece = text[start:end].strip()
        if len(piece) < _MIN_PIECE:
            continue  # rides with what follows (or is returned as the remainder at the end)
        if piece:
            out.append(piece)
        start = m.end()
    rest = text[start:]
    if final and rest.strip():
        out.append(rest.strip())
        rest = ""
    return out, rest


_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_HIDDEN_LINE = re.compile(r"^\s*(MEDIA:|\[\[[^\]]*\]\]\s*$)")
_TABLE_RULE = re.compile(r"^\s*\|?[-:\s|]+\|?\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s*>+\s?")
# Starts that mean "do not read ahead in this unfinished line yet".
_HOLD_STARTS = ("```", "~~~", "MEDIA:", "[[", "|", "#", ">")

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS = re.compile(r"(?<!\w)[*_~]{1,3}([^*_~]+)[*_~]{1,3}(?!\w)")
_STRAY_MARKS = re.compile(r"(?<!\w)[*_~`]+|[*_~`]+(?!\w)")
_URL = re.compile(r"https?://\S+")
_DIRECTIVE = re.compile(r"\[\[[^\]]*\]\]")
_SPACE = re.compile(r"\s+")
# Emoji and pictographs, with their joiners and variation selectors.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D\U000020E3\U00002190-\U000021FF\U00002300-\U000023FF]+"
)


def clean_inline(text: str) -> str:
    """Remove inline markup from one line or sentence of prose."""
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _DIRECTIVE.sub("", text)
    text = text.replace("|", ", ")
    text = _EMPHASIS.sub(r"\1", text)
    text = _STRAY_MARKS.sub("", text)  # an opening ** whose closing half has not arrived yet
    text = text.replace("_", " ")  # snake_case_name reads as three words
    text = _URL.sub("a link", text)
    text = _EMOJI.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    text = re.sub(r"\s+([.,!?;:…])", r"\1", text)  # a removed emoji or tag leaves "word ."
    return re.sub(r"^[,;:\s]+", "", text)


def _speakable(text: str) -> bool:
    """Something a voice can say: at least one letter or digit."""
    return any(c.isalnum() for c in text)


class SpeechText:
    """Feed it text as it arrives; it hands back sentences ready for the voice."""

    def __init__(self) -> None:
        self._tail = ""  # the unfinished last line, raw
        self._tail_taken = 0  # how much of _tail has already been moved into _prose
        self._prose = ""  # cleaned text waiting for a sentence end
        self._in_code = False

    def feed(self, piece: str) -> list[str]:
        self._tail += piece
        while "\n" in self._tail:
            line, self._tail = self._tail.split("\n", 1)
            self._line(line, taken=self._tail_taken)
            self._tail_taken = 0
        self._read_ahead()
        return self._take(final=False)

    def flush(self) -> list[str]:
        """The text is complete: return whatever is left."""
        if self._tail:
            self._line(self._tail, taken=self._tail_taken)
        self._tail, self._tail_taken = "", 0
        return self._take(final=True)

    # ---- internals ----
    def _line(self, whole: str, *, taken: int) -> None:
        """Classify one finished line and add the part of it not already read ahead to the prose."""
        if _FENCE.match(whole):
            self._in_code = not self._in_code
            self._break()
            return
        is_rule = bool(whole.strip()) and bool(_TABLE_RULE.match(whole))
        if self._in_code or _HIDDEN_LINE.match(whole) or is_rule:
            self._break()
            return
        text = whole[taken:]
        if whole.lstrip().startswith("|"):  # a table row: its cells, read as a list
            text = ", ".join(c.strip() for c in whole.strip().strip("|").split("|") if c.strip())
        elif taken == 0:  # nothing of this line was read ahead: strip its line markup
            text = _BULLET.sub("", _HEADING.sub("", _QUOTE.sub("", text)))
        self._prose += text
        self._break()

    def _break(self) -> None:
        """A line ended: whatever prose is pending is a sentence of its own."""
        pending = self._prose.rstrip()
        if not pending:
            self._prose = ""
        elif pending[-1] == ":":
            self._prose = pending[:-1] + ". "  # "Three things today:" introduces a list or a block
        elif pending[-1] in ".!?…;,":
            self._prose = pending + " "
        else:
            self._prose = pending + ". "

    def _read_ahead(self) -> None:
        """Move finished sentences out of the unfinished last line, if it is plainly prose."""
        if self._in_code:
            return
        tail = self._tail
        head = tail.lstrip()
        if len(head) < 8 or head.startswith(_HOLD_STARTS):
            return
        # Only up to the last sentence end that is followed by whitespace.
        last = None
        for m in _SENTENCE_END.finditer(tail, self._tail_taken):
            if m.group(2):
                last = m
        if last is None:
            return
        chunk = tail[self._tail_taken:last.end()]
        if self._tail_taken == 0:
            chunk = _BULLET.sub("", _HEADING.sub("", _QUOTE.sub("", chunk)))
        self._prose += chunk
        self._tail_taken = last.end()

    def _take(self, *, final: bool) -> list[str]:
        sentences, self._prose = split_sentences(self._prose, final=final)
        out = [clean_inline(s) for s in sentences]
        return [s for s in out if _speakable(s)]


def prepare(text: str) -> list[str]:
    """All of a finished text at once."""
    st = SpeechText()
    return st.feed(text) + st.flush()
