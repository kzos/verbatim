# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
# SPDX-License-Identifier: Apache-2.0
"""Reproducers for three defects in NeMo's per-stream context-biasing registry.

Read from source first, then reproduced here so an upstream report carries evidence rather than an
argument. Nothing in this file is specific to any downstream project: it builds boosting trees from raw
token id lists, so it needs no checkpoint, no tokenizer and no audio.

R1  remove_model() does not fire the reallocation callback that add_model() fires.
R2  removing the model that sits at offset 0 leaves that model's own offset at -num_states.
R3  after a removal the surviving models must score identically to before. This is the one that decides
    whether R1 and R2 are cosmetic or a correctness bug.
"""
import warnings
warnings.filterwarnings("ignore")
import torch
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import GPUBiasingMultiModel
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import GPUBoostingTreeModel
from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph
import nemo

V = 32
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"nemo {nemo.__version__}  torch {torch.__version__}  device {dev}")
print(f"biasing_multi_model: {GPUBiasingMultiModel.__module__}")


def tree(token_id_lists, score=2.0):
    g = ContextGraph(context_score=score, depth_scaling=1.0)
    g.build(token_ids=token_id_lists,
            phrases=[f"p{i}" for i in range(len(token_id_lists))],
            scores=[score] * len(token_id_lists),
            uniform_weights=False)
    return GPUBoostingTreeModel.from_context_graph(
        context_graph=g, vocab_size=V, unk_score=0.0, final_eos_score=0.0,
        use_triton=True, uniform_weights=False)


fired = []
mm = GPUBiasingMultiModel(vocab_size=V, reallocation_callback_fn=lambda: fired.append("realloc"))
mm = mm.to(dev)

trees = [tree([[3, 4, 5], [6, 7]]), tree([[8, 9, 10, 11]]), tree([[12, 13], [14, 15, 16]])]
ids = []
for i, t in enumerate(trees):
    before = len(fired)
    ids.append(mm.add_model(t.to(dev), alpha=1.0))
    print(f"  add_model -> id {ids[-1]}, callback fired {len(fired) - before} time(s)")

print(f"\nregistered ids {ids}")
print(f"states offsets {mm.model2states_offset[:5].tolist()}")
print(f"arcs   offsets {mm.model2arcs_offset[:5].tolist()}")


def score_of(model_id, state_batch):
    """Scores for one model over a batch of states, as a plain tensor."""
    states = torch.tensor(state_batch, dtype=torch.long, device=dev)
    mids = torch.full((len(state_batch),), model_id, dtype=torch.long, device=dev)
    s, _ = mm.advance(states=states, model_ids=mids)
    return s.detach().clone()


# Walk model 2 one step from its start state so the comparison is over a real traversal, not just BOS.
init = mm.get_init_states(batch_size=1, bos=True)
start = int(init[0].item())
probe_states = [start] * 4
before_removal = score_of(ids[2], probe_states)
print(f"\nmodel {ids[2]} scores before removal: nonzero entries "
      f"{int((before_removal[0] != 0).sum())} / {V}")

# ---------------------------------------------------------------- R1 and R2
n_before = len(fired)
off_before = mm.model2states_offset[: len(ids) + 1].tolist()
mm.remove_model(ids[0])
off_after = mm.model2states_offset[: len(ids) + 1].tolist()

print("\n=== R1: does remove_model fire the reallocation callback? ===")
print(f"  callback fired during remove_model: {len(fired) - n_before} time(s)")
print(f"  R1 {'REPRODUCED (it does not fire)' if len(fired) == n_before else 'not reproduced'}")

print("\n=== R2: the removed model's own offset ===")
print(f"  offsets before removal: {off_before}")
print(f"  offsets after  removal: {off_after}")
neg = [i for i, v in enumerate(off_after) if v < 0]
print(f"  negative offsets at model ids {neg}")
print(f"  R2 {'REPRODUCED' if neg else 'not reproduced'}")

# ---------------------------------------------------------------- R3
after_removal = score_of(ids[2], probe_states)
same = torch.equal(before_removal, after_removal)
print("\n=== R3: does a surviving model still score identically? ===")
print(f"  bit-identical: {same}")
if not same:
    d = (before_removal - after_removal).abs()
    print(f"  max abs difference {d.max().item():.6g} over {int((d != 0).sum())} entries")
print(f"  R3 {'PASSES - compaction is correct for live models' if same else 'FAILS - REAL CORRECTNESS BUG'}")

print("\n=== summary ===")
print(f"  R1 no-callback-on-remove : {'REPRODUCED' if len(fired) == n_before else 'no'}")
print(f"  R2 negative offset       : {'REPRODUCED' if neg else 'no'}")
print(f"  R3 surviving-model scores: {'unchanged' if same else 'CHANGED'}")


# ---------------------------------------------------------------- R4 and R5
print("\n" + "=" * 70)
print("R4: every RESERVED-but-unused slot also went negative. Does that bite?")
print("=" * 70)
n_slots = mm.model2states_offset.shape[0]
neg_slots = int((mm.model2states_offset < 0).sum())
print(f"  reserved slots: {n_slots}, now holding a negative offset: {neg_slots}")
print(f"  offset of the slot that a model_id of -1 indexes: "
      f"{int(mm.model2states_offset[-1].item())}")

mixed = torch.tensor([start, start, start, start], dtype=torch.long, device=dev)
mids_mixed = torch.tensor([ids[2], -1, ids[1], -1], dtype=torch.long, device=dev)
s_mixed, _ = mm.advance(states=mixed, model_ids=mids_mixed)
unbiased_rows_zero = bool((s_mixed[1] == 0).all() and (s_mixed[3] == 0).all())
print(f"  triton path: rows with model_id -1 are all-zero: {unbiased_rows_zero}")
print(f"  triton path: biased row still nonzero: {int((s_mixed[0] != 0).sum())} entries")

print("\n  same batch through the non-Triton fallback:")
try:
    mm.use_triton = False
    s_py, _ = mm.advance(states=mixed, model_ids=mids_mixed)
    agree = torch.equal(s_py, s_mixed)
    print(f"    completed without error. matches triton path: {agree}")
    if not agree:
        d = (s_py - s_mixed).abs()
        print(f"    max abs difference {d.max().item():.6g} over {int((d != 0).sum())} entries")
        print("    R4 REPRODUCED: the two paths disagree after a removal")
    else:
        print("    R4 does not bite here: both paths agree")
finally:
    mm.use_triton = True

print("\n" + "=" * 70)
print("R5: register a new session AFTER a removal. Does the reused slot work?")
print("=" * 70)
new_id = mm.add_model(tree([[20, 21], [22, 23, 24]]).to(dev), alpha=1.0)
print(f"  add_model after remove -> id {new_id}")
print(f"  offsets now: {mm.model2states_offset[:6].tolist()}")
s_new = score_of(new_id, probe_states)
print(f"  new model scores: nonzero entries {int((s_new[0] != 0).sum())} / {V}")
still = score_of(ids[2], probe_states)
print(f"  older surviving model still bit-identical to its pre-removal scores: "
      f"{torch.equal(still, before_removal)}")
