"""Model-level tests (need ANOMALOUS_TTS_ASSETS; see conftest.py)."""

import numpy as np
import torch

from Anomalous_TTS.core import characters, t2s_batch, text_frontend
from Anomalous_TTS.core.engine import SynthesisParams
from Anomalous_TTS.core.planner import NodeOptions, build_plan

SENTENCES = ["先生、おはようございます！", "今日も一緒に頑張りましょうね。", "えへへ、先生に褒められちゃいました！"]


def _voice_inputs(engine, assets):
    char = characters.load_character("test", str(assets / "character"))
    plan = build_plan({"test": char}, NodeOptions(character="test", reference_audio="ref.wav"), "テスト。")
    voice = plan.lines[0].voice
    gpt = engine.gpt(voice.gpt_path)
    ref = engine._reference(voice, engine.sovits(voice.sovits_path))
    rows = []
    for s in SENTENCES:
        p, b = text_frontend.get_phones_and_bert(text_frontend.ensure_sentence_end(s, "ja"), "ja", engine)
        rows.append((torch.LongTensor(ref.phones + p), torch.cat([ref.bert, b], 1)))
    return gpt, ref, rows


def test_batch_decoder_matches_upstream_naive(engine, assets):
    gpt, ref, rows = _voice_inputs(engine, assets)
    model, stop = gpt.model, 50 * gpt.max_sec
    with torch.inference_mode():
        for i, (x, b) in enumerate(rows):
            torch.manual_seed(100 + i)
            pred, idx = model.infer_panel(x.unsqueeze(0), torch.tensor([x.shape[0]]), ref.prompt.unsqueeze(0),
                                          b.unsqueeze(0), top_k=15, top_p=1.0, temperature=1.0,
                                          early_stop_num=stop, repetition_penalty=1.35)
            ours = t2s_batch.decode(model, [x], [b], ref.prompt, [100 + i], 15, 1.0, 1.0, 1.35, stop)[0]
            assert torch.equal(pred[0, -idx:].long(), ours.long())


def test_batch_result_independent_of_batch(engine, assets):
    gpt, ref, rows = _voice_inputs(engine, assets)
    stop = 50 * gpt.max_sec
    single = [t2s_batch.decode(gpt.model, [x], [b], ref.prompt, [7 + i], 15, 1.0, 1.0, 1.35, stop)[0]
              for i, (x, b) in enumerate(rows)]
    batch = t2s_batch.decode(gpt.model, [r[0] for r in rows], [r[1] for r in rows], ref.prompt,
                             [7 + i for i in range(len(rows))], 15, 1.0, 1.0, 1.35, stop)
    assert all(torch.equal(a, b) for a, b in zip(single, batch))


def test_synthesize_cache_and_determinism(engine, assets):
    char = characters.load_character("test", str(assets / "character"))
    chars = {"test": char}
    opts = NodeOptions(character="test", reference_audio="ref.wav", seed=3)
    params = SynthesisParams(batch_size=4)
    text = "".join(SENTENCES)
    wav1, sr, rep1 = engine.synthesize(build_plan(chars, opts, text), params)
    assert sr == 32000 and wav1.size > sr and np.isfinite(wav1).all() and rep1.generated == 3
    wav2, _, rep2 = engine.synthesize(build_plan(chars, opts, text), params)
    assert rep2.from_cache == 3 and np.array_equal(wav1, wav2)
    edited = text.replace("今日も一緒に頑張りましょうね。", "今日はお休みですね。")
    _, _, rep3 = engine.synthesize(build_plan(chars, opts, edited), params)
    assert rep3.generated == 1 and rep3.from_cache == 2
