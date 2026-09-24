"""Regenerate tests/fixtures/frontend.json from the current front end.

Only run this after checking the new output against upstream GPT-SoVITS
(see UPSTREAM.md "对拍"); the fixture is the verified baseline the tests guard.

    ANOMALOUS_TTS_ASSETS=... python tools/make_frontend_fixtures.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import conftest  # noqa: E402  (stubs ComfyUI, imports the package)

from Anomalous_TTS.core import text_frontend  # noqa: E402
from Anomalous_TTS.core.engine import Engine  # noqa: E402

import torch  # noqa: E402

corpus = json.loads((ROOT / "tests" / "fixtures" / "corpus.json").read_text(encoding="utf-8"))
engine = Engine(conftest.AssetResources(Path(os.environ["ANOMALOUS_TTS_ASSETS"])), torch.device("cpu"), torch.float32)
out = {}
for lang, lines in corpus.items():
    out[lang] = []
    for text in lines:
        ids, bert = text_frontend.get_phones_and_bert(text, lang, engine, final=True)
        out[lang].append({"text": text, "ids": ids, "bert_sum": round(float(bert.sum()), 3)})
(ROOT / "tests" / "fixtures" / "frontend.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: len(v) for k, v in out.items()})
