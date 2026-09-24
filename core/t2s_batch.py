"""Batched GPT (text-to-semantic) decoding with one random stream per sentence.

Adapted from GPT-SoVITS ``Text2SemanticDecoder.infer_panel_batch_infer`` and
``infer_panel_naive`` (AR/models/t2s_model.py, MIT, RVC-Boss). Differences:

- Each row samples with its own ``torch.Generator``, so a sentence's result
  depends only on its own seed, not on which other sentences share the batch.
  That keeps results reproducible and makes per-sentence caching exact.
- Works with and without a reference prompt (upstream batch mode needs one).
- Stop rule per row is the one of ``infer_panel_naive``: stop when the sampled
  token or the arg-max token is EOS (EOS dropped); EOS is blocked for the first
  11 steps; hard cap ``early_stop_num`` and 1500 steps.

With one row and no padding it performs the same operations as
``infer_panel_naive`` (tested in tests/test_t2s_batch.py).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

MAX_STEPS = 1500
MIN_STEPS = 11


def _row_noise(shape_v: int, gens: List[torch.Generator], like: torch.Tensor) -> torch.Tensor:
    rows = [torch.empty((1, shape_v), device=like.device, dtype=like.dtype).exponential_(1, generator=g) for g in gens]
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def decode(
    model,
    phones: List[torch.Tensor],  # each (T_i,) long
    berts: List[torch.Tensor],  # each (1024, T_i)
    prompt: Optional[torch.Tensor],  # (P,) long, shared by all rows; None = no reference text
    seeds: List[int],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    early_stop_num: int,
) -> List[torch.Tensor]:
    """Return the generated semantic tokens (without prompt and EOS) for each row."""
    from ..vendor.gpt_sovits.AR.models.utils import logits_to_probs, make_pad_mask_left

    device = phones[0].device
    bsz = len(phones)
    lens = torch.tensor([p.shape[0] for p in phones], device=device)
    max_len = int(lens.max())
    padded = bool((lens != max_len).any())

    xs = []
    for p, b in zip(phones, berts):
        x = model.ar_text_embedding(p.unsqueeze(0))
        x = x + model.bert_proj(b.transpose(0, 1).unsqueeze(0))
        x = model.ar_text_position(x).squeeze(0)
        if x.shape[0] < max_len:
            x = F.pad(x, (0, 0, max_len - x.shape[0], 0), value=0)  # pad left
        xs.append(x)
    x = torch.stack(xs, dim=0)

    if prompt is not None:
        y = prompt.unsqueeze(0).expand(bsz, -1).contiguous()
        y_emb = model.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        xy_pos = torch.cat([x, model.ar_audio_position(y_emb)], dim=1)
    else:
        y = torch.zeros(bsz, 0, dtype=torch.long, device=device)
        y_len = 0
        xy_pos = x
    prefix_len = y.shape[1]
    x_len = max_len
    src_len = x_len + y_len

    x_mask = F.pad(torch.zeros(x_len, x_len, dtype=torch.bool, device=device), (0, y_len), value=True)
    y_mask = F.pad(
        torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=device), diagonal=1), (x_len, 0), value=False
    )
    attn_mask = torch.cat([x_mask, y_mask], dim=0).view(1, src_len, src_len).repeat(bsz, 1, 1)
    if padded:
        pad = make_pad_mask_left(lens, max_len)
        pad = torch.cat([pad, torch.zeros(bsz, y_len, dtype=torch.bool, device=device)], dim=1)
        attn_mask = attn_mask.logical_or(pad.view(bsz, 1, src_len).expand(-1, src_len, -1))
    attn_mask = attn_mask.unsqueeze(1).expand(-1, model.num_head, -1, -1)

    gens = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        gens.append(g)

    alive = list(range(bsz))  # row -> original index
    results: List[Optional[torch.Tensor]] = [None] * bsz
    k_cache = v_cache = None
    step_mask: Optional[torch.Tensor] = None
    for idx in range(MAX_STEPS):
        if idx == 0:
            xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(xy_pos, attn_mask, None)
            if padded:
                step_mask = F.pad(attn_mask[:, :, -1:], (0, 1), value=False)
        else:
            xy_dec, k_cache, v_cache = model.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache, step_mask)
            if step_mask is not None:
                step_mask = F.pad(step_mask, (0, 1), value=False)
        logits = model.ar_predict_layer(xy_dec[:, -1])
        if idx < MIN_STEPS:
            logits = logits[:, :-1]

        probs = logits_to_probs(
            logits, previous_tokens=y, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, temperature=temperature,
        )
        q = _row_noise(probs.shape[-1], gens, probs)
        samples = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=y.dtype)
        y = torch.cat([y, samples], dim=1)

        eos = (samples[:, 0] == model.EOS) | (torch.argmax(logits, dim=-1) == model.EOS)
        too_long = early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num
        last = idx == MAX_STEPS - 1
        done = [i for i in range(len(alive)) if bool(eos[i]) or too_long or last]
        for i in done:
            row = y[i, prefix_len:]
            if bool(eos[i]):
                row = row[:-1]
            if row.numel() == 0:
                row = torch.zeros(1, dtype=y.dtype, device=device)  # upstream: "bad zero prediction"
            results[alive[i]] = row
        if done:
            keep = [i for i in range(len(alive)) if i not in set(done)]
            if not keep:
                break
            keep_t = torch.tensor(keep, device=device)
            y = y.index_select(0, keep_t)
            k_cache = [k.index_select(0, keep_t) for k in k_cache]
            v_cache = [v.index_select(0, keep_t) for v in v_cache]
            if step_mask is not None:
                step_mask = step_mask.index_select(0, keep_t)
            alive = [alive[i] for i in keep]
            gens = [gens[i] for i in keep]

        y_emb = model.ar_audio_embedding(y[:, -1:])
        xy_pos = y_emb * model.ar_audio_position.x_scale + model.ar_audio_position.alpha * model.ar_audio_position.pe[
            :, y_len + idx
        ].to(dtype=y_emb.dtype, device=y_emb.device)
    return results  # type: ignore[return-value]
