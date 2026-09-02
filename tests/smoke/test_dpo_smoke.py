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


def test_dpo_config_kwargs_used_by_run_dpo_are_valid(gpt2, tmp_path):
    """The exact DPOConfig kwargs run_dpo.py passes must be accepted by the pinned
    TRL, and checkpoints must actually get written (the probe-transfer seam)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = gpt2
    ds = Dataset.from_list(_preference_rows()).train_test_split(test_size=0.2, seed=0)

    args = DPOConfig(
        output_dir=str(tmp_path),
        max_steps=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        warmup_steps=2,
        beta=0.1,
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, 0.1],
        max_length=128,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="steps",
        save_steps=2,
        report_to="none",
        seed=0,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tok,
    )
    trainer.train()

    checkpoints = list(tmp_path.glob("checkpoint-*"))
    assert checkpoints, "no checkpoint dirs written"


def test_dpo_lora_path_runs_and_merges(gpt2, tmp_path):
    """run_dpo.py's LoRA path: DPOTrainer(peft_config=...) trains, merge_and_unload
    produces a plain model, save_model writes it."""
    peft = pytest.importorskip("peft")
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = gpt2
    ds = Dataset.from_list(_preference_rows())

    args = DPOConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        beta=0.1,
        max_length=128,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        report_to="none",
        save_strategy="no",
        seed=0,
    )
    lora = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    assert not hasattr(merged, "peft_config")
    merged.save_pretrained(tmp_path / "merged")
    assert (tmp_path / "merged" / "config.json").exists()
