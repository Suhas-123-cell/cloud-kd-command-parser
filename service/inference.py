"""Prompt format, generation and schema validation shared by training, evaluation and the service.

Training example:   "Command: <command>\nJSON: <teacher JSON reply><eos>"
At inference the model is given "Command: <command>\nJSON:" and decodes greedily until it
emits <eos>, a newline, or MAX_NEW_TOKENS tokens.
"""
import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
if not SCHEMA_PATH.exists():  # running from the repository: use the kit's schema
    SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
KEYS = tuple(SCHEMA["keys"])
INTENTS = frozenset(SCHEMA["intents"])


def prompt_for(command):
    return f"Command: {command.strip()}\nJSON:"


def training_text(command, teacher_json):
    """Return (prompt, target). Loss is computed on the target only."""
    return prompt_for(command), " " + teacher_json.strip()


def validate(raw):
    """Return the parsed 4-key object if `raw` satisfies schema.json, else None."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError, RecursionError):  # JSONDecodeError is a ValueError
        return None
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        return None
    if not isinstance(obj["intent"], str) or obj["intent"] not in INTENTS:
        return None
    for k in ("date", "time", "place"):
        if obj[k] is not None and not isinstance(obj[k], str):
            return None
    return {k: obj[k] for k in KEYS}


def pretranspose_linears(model):
    """Replace every nn.Linear forward with x @ W^T where W^T is stored contiguous.

    Same maths, different memory layout. Measured inside the linux/arm64 container on the
    laptop: torch's F.linear path (OpenBLAS, transposed weight) ran the 512->50304 lm_head in
    ~26 ms, while a matmul with a pre-transposed contiguous weight took ~5 ms. Costs one extra
    copy of the weights (~+150 MB for pythia-70m), so the original weights are released.
    """
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and not hasattr(module, "_wt"):
            wt = module.weight.detach().t().contiguous()
            module._wt = wt
            module._b = None if module.bias is None else module.bias.detach()

            def fwd(x, m=module):
                y = x.matmul(m._wt)
                return y if m._b is None else y.add_(m._b)

            module.forward = fwd
    # embed_out (lm_head) shares no storage with the input embedding in pythia, so the
    # original [out, in] weights can be dropped to keep memory flat.
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "_wt"):
            module.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    return model


class Parser:
    """Loads a causal LM + tokenizer from a local directory and generates raw JSON text."""

    def __init__(self, model_dir, max_new_tokens=64, threads=None, fast_linear=True, stop_at_newline=True,
                 constrained_fallback=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        self.model.eval()
        if fast_linear:
            pretranspose_linears(self.model)
        self.max_new_tokens = int(max_new_tokens)
        newline_ids = self.tok.encode("\n") if stop_at_newline else []
        self.stop_ids = {self.tok.eos_token_id, *newline_ids}

        self.constrained_fallback = constrained_fallback
        # " {"intent": "<intent>", "date":" for each allowed intent; the tokenisation of this
        # prefix is a prefix of the tokenised training target (checked in tests).
        self.intent_prefixes = {i: self.tok(f' {{"intent": "{i}", "date":').input_ids for i in sorted(INTENTS)}
        self.fallbacks = 0

    def _greedy(self, ids, budget):
        """Greedy decoding from token ids; returns the new token ids (stop token excluded)."""
        torch = self.torch
        out, past, cur = [], None, torch.tensor([ids])
        with torch.inference_mode():
            for _ in range(budget):
                res = self.model(input_ids=cur, past_key_values=past, use_cache=True)
                past = res.past_key_values
                nxt = int(res.logits[0, -1].argmax())
                if nxt in self.stop_ids:
                    break
                out.append(nxt)
                cur = torch.tensor([[nxt]])
        return out

    def _best_intent_prefix(self, prompt_ids):
        """Score every allowed intent prefix by the model's own log-probability; return the best ids."""
        torch = self.torch
        names = list(self.intent_prefixes)
        seqs = [prompt_ids + self.intent_prefixes[n] for n in names]
        width = max(map(len, seqs))
        batch = torch.full((len(seqs), width), self.tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(batch)
        for k, s in enumerate(seqs):
            batch[k, :len(s)] = torch.tensor(s)
            mask[k, :len(s)] = 1
        with torch.inference_mode():
            logp = torch.log_softmax(self.model(input_ids=batch, attention_mask=mask).logits, dim=-1)
        scores = []
        for k, n in enumerate(names):
            pre = self.intent_prefixes[n]
            pos = range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(pre))
            scores.append(sum(float(logp[k, p, t]) for p, t in zip(pos, pre)))
        return self.intent_prefixes[names[max(range(len(names)), key=scores.__getitem__)]]

    def generate(self, command):
        prompt_ids = self.tok(prompt_for(command)).input_ids
        text = self.tok.decode(self._greedy(prompt_ids, self.max_new_tokens), skip_special_tokens=True).strip()
        if not self.constrained_fallback or validate(text) is not None:
            return text
        # Schema-constrained fallback: the free greedy output broke schema.json (e.g. an intent
        # outside the eight). Force the best-scoring allowed intent prefix, then let the model
        # finish the slots greedily. Outputs that were already valid are never changed.
        self.fallbacks += 1
        prefix = self._best_intent_prefix(prompt_ids)
        rest = self._greedy(prompt_ids + prefix, max(self.max_new_tokens - len(prefix), 1))
        return self.tok.decode(prefix + rest, skip_special_tokens=True).strip()
