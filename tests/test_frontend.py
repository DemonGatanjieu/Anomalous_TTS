"""Front-end regression test against the verified baseline (tests/fixtures/frontend.json).

The baseline matches upstream GPT-SoVITS text processing (see UPSTREAM.md), except
Japanese English-words / ASCII "%" which differ on purpose.
"""

import json
from pathlib import Path

import pytest

from Anomalous_TTS.core import text_frontend

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "frontend.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang", ["ja", "zh", "en"])
def test_phonemes_match_baseline(engine, lang):
    wrong = []
    for case in FIXTURE[lang]:
        ids, bert = text_frontend.get_phones_and_bert(case["text"], lang, engine, final=True)
        assert bert.shape == (1024, len(ids))
        if ids != case["ids"] or abs(float(bert.sum()) - case["bert_sum"]) > 0.05:
            wrong.append(case["text"])
    assert not wrong, f"{len(wrong)} {lang} sentences changed: {wrong[:5]}"


def test_same_characters_across_two_words_keep_their_tones(engine):  # engine: g2pW loaded as in use
    """Our one intended difference from upstream (UPSTREAM.md, tone_sandhi patch)."""
    from Anomalous_TTS.vendor.gpt_sovits.text import chinese2

    def tones(text):
        phones, word2ph = chinese2.g2p(chinese2.text_normalize(text))
        out, i = [], 0
        for n in word2ph:
            out.append(phones[i + n - 1][-1])
            i += n
        return "".join(out)

    assert tones("银行行长") == "2223"  # upstream: 2253
    assert tones("人民民主") == "2223"
    assert tones("奶奶") == "35" and tones("好好学习") == "3522" and tones("高高兴兴") == "1545"
