"""An MLX-backed generation client conforming to the harness `SupportsGenerate`.

Lets the evaluation harness score a local MLX model (a fused LoRA checkpoint, or a
base MLX model) with no Ollama / GGUF-conversion step in the path — useful on
Apple Silicon where `ollama create` cannot import MLX-quantised weights.

Determinism: MLX generation is greedy at ``temp=0`` (argmax), so `seed` is
accepted for interface parity but does not change a temp-0 result. Heavy `mlx`
imports are deferred to construction so importing this module stays cheap.
"""

from __future__ import annotations


class MlxClient:
    """Minimal client: ``.model`` + ``.generate(prompt, system, ...)``."""

    def __init__(self, model_path: str, *, max_tokens_default: int = 512) -> None:
        from mlx_lm import load

        self.model = model_path
        self._max_tokens_default = max_tokens_default
        # load() returns (model, tokenizer) — or (model, tokenizer, config) with
        # return_config=True, which we do not use. Index rather than tuple-unpack
        # so the type checker does not object to the union return.
        loaded = load(model_path)
        self._m, self._tok = loaded[0], loaded[1]

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int | None = None,
    ) -> GenText:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = self._tok.apply_chat_template(messages, add_generation_prompt=True)

        sampler = make_sampler(temp=float(temperature))
        text = generate(
            self._m,
            self._tok,
            prompt=rendered,
            max_tokens=num_predict or self._max_tokens_default,
            sampler=sampler,
            verbose=False,
        )
        return GenText(text=text, model=self.model)


class GenText:
    """Duck-typed stand-in for the Ollama client's generation result."""

    __slots__ = ("text", "model")

    def __init__(self, text: str, model: str) -> None:
        self.text = text
        self.model = model
