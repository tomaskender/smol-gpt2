# GPT-2 from Scratch — Learning Plan

## Stage 1 — Foundations (you already have most of this)
**Goal:** Make sure your NumPy/matrix intuition is solid before transformers.

1. Implement a single linear layer `y = xW + b` and verify shapes manually.
2. Implement `softmax`, `GELU` (the GPT-2 activation — not ReLU), and `LayerNorm` from the formula.
3. Write a tiny 2-layer MLP and do a forward pass on dummy data. Print every intermediate shape.

---

## Stage 2 — Attention (the core idea)
**Goal:** Understand what attention is actually computing, not just the formula.

4. Implement **scaled dot-product attention** for a single head: `softmax(QKᵀ / √dₖ) V`. Test on a (seq, d) matrix.
5. Add a **causal mask** — an upper-triangular matrix of `-inf` applied before softmax so token `i` can only attend to tokens ≤ `i`. This is what makes it *autoregressive*.
6. Extend to **multi-head attention**: split `d_model` into `n_heads` chunks, run attention on each, concatenate results, project with `W_o`.

---

## Stage 3 — Transformer Block
**Goal:** Build one complete GPT-2 block.

7. Combine multi-head causal attention + a 2-layer feedforward (with GELU) into a single block.
8. Use **pre-norm**: apply `LayerNorm` *before* each sublayer, not after. Add residual connections around each sublayer: `x = x + sublayer(norm(x))`.
9. Instantiate the block and do a forward pass. Verify input and output shapes are identical `(batch, seq, d_model)`.

---

## Stage 4 — Full GPT-2 Model
**Goal:** Stack everything into a complete model.

10. Add a **token embedding table** `(vocab_size, d_model)` and a **learned positional embedding table** `(max_seq_len, d_model)`. GPT-2 uses learned positionals, not sinusoidal.
11. Stack `N` blocks (GPT-2 small uses 12).
12. Add a final `LayerNorm` after the last block, then a linear projection back to `(vocab_size,)` — this produces logits.
13. **Weight tying**: the output projection matrix is the *same matrix* as the token embedding table (transposed). GPT-2 does this.

---

## Stage 5 — Inference
**Goal:** Generate text to verify correctness before worrying about training.

14. Load the [official GPT-2 weights](https://huggingface.co/gpt2) from HuggingFace into your NumPy arrays. The weight names map directly to your layer structure.
15. Implement greedy decoding: take logits from the last position, `argmax`, append to sequence, repeat.
16. Use the `tiktoken` library for tokenization — don't implement BPE yourself at this stage.
17. Feed in a prompt and verify your output matches HuggingFace's `generate()`.

---

## Stage 6 — Training (optional, resource-intensive)
**Goal:** Understand the training loop.

18. Implement cross-entropy loss over the vocabulary at each position.
19. Switch from NumPy to PyTorch (autograd is essentially required for training — reimplementing backprop through attention by hand is a separate project).
20. Train on a small text corpus (e.g. Shakespeare) with AdamW.

---

## Key things to look up as you go
- Why `/ √dₖ`? (variance control — worth deriving yourself)
- Why causal masking instead of bidirectional? (GPT is generative, BERT is not)
- What GELU does vs ReLU (smooth, probabilistic gating)
- What weight tying saves and why it works

---

> **Recommended checkpoints after each stage:** print every tensor shape at every step. Shape errors will be your main enemy — catching them early saves hours.
