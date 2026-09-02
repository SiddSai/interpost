"""Phase 1b tripwire: extraction + generation against a real HF model (gpt2, cached
from the DPO smoke test)."""

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

MODEL_ID = "gpt2"


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tok


def test_extract_pooled_shapes_and_finiteness(gpt2):
    from examples._shared.extract import extract_pooled

    model, tok = gpt2
    prompts = ["The weather today is", "My favorite food is", "In the year 2000"]
    responses = [" sunny and warm.", " pizza with extra cheese.", " computers got faster."]
    layers = [2, 5, -1]

    acts = extract_pooled(model, tok, prompts, responses, layers, batch_size=2)

    assert set(acts) == set(layers)
    d = model.config.n_embd
    for li in layers:
        assert acts[li].shape == (3, d)
        assert np.isfinite(acts[li]).all()


def test_generate_responses_returns_k_per_prompt(gpt2):
    from examples._shared.extract import generate_responses

    model, tok = gpt2
    prompts = ["Once upon a time", "The best way to learn is"]
    out = generate_responses(model, tok, prompts, k=3, max_new_tokens=8, batch_size=2)

    assert len(out) == 2
    assert all(len(cands) == 3 for cands in out)
    assert all(isinstance(c, str) for cands in out for c in cands)


def test_extract_then_fit_probe_end_to_end(gpt2):
    """toy label = 'does the response mention food' — should be linearly probeable."""
    from examples._shared.extract import extract_pooled
    from examples._shared.probe import fit_probe

    model, tok = gpt2
    food = [" pizza and pasta.", " a burger and fries.", " sushi tonight.", " tacos again."]
    other = [" a long walk outside.", " reading a book.", " fixing the car.", " loud music."]
    prompts = ["I want"] * 8
    responses = food + other
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    acts = extract_pooled(model, tok, prompts, responses, [4, 8], batch_size=4)
    probe = fit_probe(acts, labels, val_frac=0.5)
    assert probe.layer in (4, 8)
    assert 0.0 <= probe.val_auc <= 1.0
