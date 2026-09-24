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
