"""Phase 0 tripwire: stock TRL DPO training runs in our pinned environment.

interpost's whole strategy is subclassing ``trl.DPOTrainer`` / ``trl.GRPOTrainer``.
When TRL changes an internal method we rely on, this test (plus the ``trl==`` pin in
pyproject.toml) is how we find out immediately. Runs on CPU with gpt2 in ~1-2 min;
downloads gpt2 (~550 MB) on first run.
"""

import pytest

pytestmark = pytest.mark.smoke

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")

MODEL_ID = "gpt2"


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tok


def _preference_rows(n: int = 48):
    return [
        {
            "prompt": f"Question {i}: what is {i} plus {i}?",
            "chosen": f" It is {2 * i}.",
            "rejected": " I would rather not say.",
        }
        for i in range(n)
    ]


def test_resolve_layers_finds_gpt2_blocks(gpt2):
    """Phase 0 code: HookManager locates the decoder-layer ModuleList."""
    from interpost.activations import HookManager

    model, _ = gpt2
    hm = HookManager(model)
    assert len(hm.layers) == model.config.n_layer


def test_dpo_trainer_runs_five_steps(gpt2, tmp_path):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = gpt2
    train_dataset = Dataset.from_list(_preference_rows())

    args = DPOConfig(
        output_dir=str(tmp_path),
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        beta=0.1,
        max_length=128,
        gradient_checkpointing=False,
        bf16=False,
        fp16=False,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        seed=0,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tok,
    )
    result = trainer.train()

    loss = result.training_loss
    assert loss is not None
    assert loss == loss  # not NaN
    assert abs(loss) < 1e4  # not diverged
