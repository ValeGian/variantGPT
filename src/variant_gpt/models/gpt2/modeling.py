import inspect
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .configuration import GPT2Config
from ...activations import ACT2FN
from ...attention import AttentionConfig, build_attention


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = ACT2FN[config.activation_function]
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # Explicit forward avoids nn.Sequential overhead and lets torch.compile
        # fuse the activation into the linear kernel more easily.
        return self.drop(self.proj(self.act(self.fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon)

        attn_cfg = AttentionConfig(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            flash=config.flash,
            n_kv_head=getattr(config, "n_kv_head", None),
            # Optional on GPT2Config; only required when attention_type == "local".
            window_size=getattr(config, "window_size", None),
        )
        self.attn = build_attention(config.attention_type, attn_cfg)

        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # pre-register position indices to avoid re-allocation every forward pass
        self.register_buffer(
            "_pos",
            torch.arange(config.block_size, dtype=torch.long),
            persistent=False,
        )

        self.to(config.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # Embedding — use pre-registered buffer, slice to avoid copy
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(self._pos[:t])  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks — plain loop is compile-friendly; nn.Sequential
        # wrapping a ModuleList is kept for checkpointing flexibility
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness (higher = more random).
            top_k: Limits generation to the top-k most likely tokens.
            top_p: Limits generation to tokens with cumulative probability <= top_p.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            # crop context to block_size if needed
            ctx = (
                input_tokens
                if input_tokens.size(1) <= self.config.block_size
                else input_tokens[:, -self.config.block_size:]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(ctx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # (B, vocab)

            # --- top-k gate (applied to logits, before softmax) ---
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < threshold[:, [-1]], float("-inf"))

            # --- single softmax ---
            probs = F.softmax(logits, dim=-1)

            # --- top-p (nucleus) gate (applied to probs) ---
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = sorted_probs.cumsum(dim=-1)

                # shift right so the first token above the threshold is kept
                sorted_mask = (cumulative - sorted_probs) > top_p
                sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

                # scatter back to original vocab ordering and renormalize
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)

        return input_tokens


__all__ = [
    "GPT2Model",
]
