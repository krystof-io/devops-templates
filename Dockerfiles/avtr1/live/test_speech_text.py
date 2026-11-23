#!/usr/bin/env python3
"""Tests for kio_speech_text. Standard library only:  python3 live/test_speech_text.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kio_speech_text import SpeechText, clean_inline, prepare  # noqa: E402

REPLY = """Morning. The **overnight backup** finished clean, and three emails are flagged urgent 🙂

## Today
1. 12:30 inbox triage
2. 15:00 dentist
- call the [landlord](https://example.com/landlord) about Thursday

Here is the command:
```bash
kubectl get pods -A | grep -v Running
```
MEDIA:/tmp/whiteboard.jpg
[[audio_as_voice]]

| Job | State |
|-----|-------|
| tls-cert-watch | failed |

See https://nasa.gov/artemis for the `launch window`. Done!
"""

def spoken(pieces):
    st = SpeechText()
    out = []
    for p in pieces:
        out += st.feed(p)
    return out + st.flush()


def norm(sentences):
    # Where the cuts fall depends on how the text arrived; what is spoken must not.
    return " ".join(" ".join(sentences).split())


def main():
    whole = prepare(REPLY)
    joined = norm(whole)
    for bad in ("kubectl", "grep", "MEDIA", "whiteboard", "audio_as_voice", "**", "##", "```", "https://", "🙂", "---", "[", "]("):
        assert bad not in joined, (bad, joined)
    for good in ("overnight backup finished clean", "12:30 inbox triage", "landlord about Thursday", "a link for the launch window", "Done!"):
        assert good in joined, (good, joined)

    # However the text is cut up on arrival, the same words are spoken.
    rng = random.Random(7)
    for trial in range(300):
        pieces, i = [], 0
        while i < len(REPLY):
            n = rng.choice((1, 2, 3, 5, 8, 13, 40))
            pieces.append(REPLY[i:i + n])
            i += n
        got = norm(spoken(pieces))
        assert got == joined, f"trial {trial}:\n{got}\n!=\n{joined}"

    # Speech starts before a long paragraph has finished arriving.
    st = SpeechText()
    early = st.feed("The first sentence is complete here. The second one is still being wri")
    assert early == ["The first sentence is complete here."], early
    assert st.feed("tten.") == [] and st.flush() == ["The second one is still being written."]

    # Nothing of a code block or a MEDIA line is spoken, even one character at a time.
    assert spoken(list("```\nrm -rf /\n```\nSafe to say this sentence.\n")) == ["Safe to say this sentence."]
    assert spoken(list("MEDIA:/x/y.png\nA picture is attached for you.\n")) == ["A picture is attached for you."]

    # Short fragments ride with what follows; nothing is lost at the end.
    assert prepare("Yes. That works for me.") == ["Yes. That works for me."]
    assert prepare("Hi") == ["Hi."]  # a line end closes the sentence
    assert prepare("   \n\n") == [] and prepare("🙂") == [] and prepare("***") == []

    assert clean_inline("**bold** and _it_ and `code` and ~~gone~~") == "bold and it and code and gone"
    assert clean_inline("an unfinished **bold start") == "an unfinished bold start"
    assert clean_inline("snake_case_name stays") == "snake case name stays"
    print("kio_speech_text: all tests passed")
    print("  sample ->", whole)


if __name__ == "__main__":
    main()
