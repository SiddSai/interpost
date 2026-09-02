"""Phase 2 tripwire: OfflineDPOTrainer adds a differentiable activation term to the
DPO loss, on gpt2 (+ LoRA, the real path)."""

import pytest

pytestmark = pytest.mark.smoke

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

MODEL_ID = "gpt2"


def _rows(n=48):
    return [
        {"prompt": f"Q{i}: 2+{i}?", "chosen": f" it is {2 + i}.", "rejected": " no comment."}
        for i in range(n)
    ]


@pytest.fixture
def gpt2():
    """Fresh model per test — PEFT-wrapping the same object twice warns and stacks adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return AutoModelForCausalLM.from_pretrained(MODEL_ID), tok


def _lora():
    from peft import LoraConfig

    return LoraConfig(r=4, target_modules=["c_attn"], task_type="CAUSAL_LM")


def _config(tmp_path):
    from trl import DPOConfig

    return DPOConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        beta=0.1,
        max_length=64,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        bf16=False,
        fp16=False,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        seed=0,
    )


def test_hookmanager_resolves_lora_wrapped_gpt2(gpt2):
    from peft import get_peft_model

    from interpost.activations import HookManager

    model, _ = gpt2
    hm = HookManager(get_peft_model(model, _lora()))
    assert len(hm.layers) == model.config.n_layer


def test_signal_term_changes_the_loss(gpt2, tmp_path):
    from datasets import Dataset

    from interpost.trainers import OfflineDPOTrainer

    model, tok = gpt2
    ds = Dataset.from_list(_rows())

    def make(weight):
        return OfflineDPOTrainer(
            model=model,
            args=_config(tmp_path),
            train_dataset=ds,
            processing_class=tok,
            peft_config=_lora(),
            signal_layer=4,
            signal_term_fn=lambda c, r: torch.relu(r.mean(dim=-1)),
            signal_weight=weight,
        )

    t0, t1 = make(0.0), make(5.0)
    batch = t0._prepare_inputs(next(iter(t0.get_train_dataloader())))
    l0 = t0.compute_loss(t0.model, batch)
    l1 = t1.compute_loss(t1.model, t1._prepare_inputs(next(iter(t1.get_train_dataloader()))))

    assert torch.isfinite(l0) and torch.isfinite(l1)
    assert not torch.allclose(l0, l1), "nonzero signal_weight should move the loss"
    assert t1._metrics["train"]["signal/term"], "signal/term not recorded"


def test_train_loop_runs_and_logs_signal(gpt2, tmp_path):
    from datasets import Dataset
    from transformers import TrainerCallback

    from interpost.trainers import OfflineDPOTrainer

    seen = []

    class Grab(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "signal/term" in logs:
                seen.append(logs["signal/term"])

    model, tok = gpt2
    trainer = OfflineDPOTrainer(
        model=model,
        args=_config(tmp_path),
        train_dataset=Dataset.from_list(_rows()),
        processing_class=tok,
        peft_config=_lora(),
        signal_layer=4,
        signal_term_fn=lambda c, r: torch.relu(r.mean(dim=-1)),
        signal_weight=0.1,
        callbacks=[Grab()],
    )
    trainer.train()
    assert seen and all(v == v for v in seen), "signal/term never logged"
