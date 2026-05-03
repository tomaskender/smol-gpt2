import numpy as np
from transformers import GPT2Tokenizer
from safetensors import safe_open


GEN = np.random.default_rng(0)

def softmax(X: np.array):
    # return np.exp(x)/np.sum(np.exp(x,axis=-1, keepdims=True))
    exp = np.exp(X - np.max(X, axis=-1, keepdims=True)) # stabilize to prevent overflow
    return exp / np.sum(exp, axis=-1, keepdims=True)

def gelu(X: np.array):
    return 0.5 * X * (1 + np.tanh(np.sqrt(2/np.pi) * (X + 0.044715 * (X ** 3))))

def layernorm(X: np.array):
    u = np.mean(X, axis=-1, keepdims=True)
    var = np.mean((X-u) ** 2, axis=-1, keepdims=True)
    return (X-u) / np.sqrt(var + 1e-5)

def run_mlp():
    IN_DIM = 16
    HIDDEN_DIM = 8
    OUT_DIM = 4

    X = GEN.normal(0, 1, IN_DIM)
    W_h = GEN.normal(0, 1, (IN_DIM, HIDDEN_DIM))
    b_h = GEN.normal(0, 1, HIDDEN_DIM)
    W_o = GEN.normal(0, 1, (HIDDEN_DIM, OUT_DIM))
    b_o = GEN.normal(0, 1, OUT_DIM)

    X = X @ W_h + b_h
    X = gelu(X)
    X = X @ W_o + b_o
    return X

# print(run_mlp())

def attention(Q: np.array, K: np.array, V: np.array):
    qk_norm = Q @ K.T / np.sqrt(K.shape[-1])
    causal_mask = np.triu(np.full(qk_norm.shape, -np.inf), k=1) # upper triangular matrix to mask out future tokens during training
    return softmax(causal_mask + qk_norm) @ V

def multihead_attention(tokens: np.array, n_heads: int, W_allblocks: dict, block_id: int):
    d_model = tokens.shape[-1]
    seq_len = tokens.shape[0]
    d_head = d_model // n_heads

    W_qkv_all = W_allblocks[f"h.{block_id}.attn.c_attn.weight"].reshape((d_model, 3, n_heads, d_head))
    W_q_all = W_qkv_all[:, 0, :, :]
    W_k_all = W_qkv_all[:, 1, :, :]
    W_v_all = W_qkv_all[:, 2, :, :]

    b_qkv_all = W_allblocks[f"h.{block_id}.attn.c_attn.bias"].reshape((3, n_heads, d_head))
    b_q_all, b_k_all, b_v_all = b_qkv_all

    out = np.empty((seq_len,0))
    for i in range(n_heads):
        W_q = W_q_all[:, i, :]
        # W_q = GEN.normal(0, 1, (d_model, d_head))
        b_q = b_q_all[i]
        # b_q = GEN.normal(0, 1, (d_head))
        W_k = W_k_all[:, i, :]
        # W_k = GEN.normal(0, 1, (d_model, d_head))
        b_k = b_k_all[i]
        # b_k = GEN.normal(0, 1, (d_head))
        W_v = W_v_all[:, i, :]
        # W_v = GEN.normal(0, 1, (d_model, d_head))
        b_v = b_v_all[i]
        # b_v = GEN.normal(0, 1, (d_head))
        out = np.hstack((out, attention(tokens @ W_q + b_q, tokens @ W_k + b_k, tokens @ W_v + b_v)))

    W_o = W_allblocks[f"h.{block_id}.attn.c_proj.weight"]
    # W_o = GEN.normal(0, 1, (d_model, d_model))
    b_o = W_allblocks[f"h.{block_id}.attn.c_proj.bias"]
    # b_o = GEN.normal(0, 1, (d_model,))
    return out @ W_o + b_o

def transformer_block(X: np.array, W_allblocks: dict, block_id: int):
    # d_model = X.shape[-1]
    # d_hidden = d_model * 4

    W_ln1 = W_allblocks[f"h.{block_id}.ln_1.weight"]
    # W_ln1 = GEN.normal(0, 1, (d_model,))
    b_ln1 = W_allblocks[f"h.{block_id}.ln_1.bias"]
    # b_ln1 = GEN.normal(0, 1, (d_model,))
    W_ln2 = W_allblocks[f"h.{block_id}.ln_2.weight"]
    # W_ln2 = GEN.normal(0, 1, (d_model,))
    b_ln2 = W_allblocks[f"h.{block_id}.ln_2.bias"]
    # b_ln2 = GEN.normal(0, 1, (d_model,))
    W_h = W_allblocks[f"h.{block_id}.mlp.c_fc.weight"]
    # W_h = GEN.normal(0, 1, (d_model, d_hidden))
    b_h = W_allblocks[f"h.{block_id}.mlp.c_fc.bias"]
    # b_h = GEN.normal(0, 1, d_hidden)
    W_o = W_allblocks[f"h.{block_id}.mlp.c_proj.weight"]
    # W_o = GEN.normal(0, 1, (d_hidden, d_model))
    b_o = W_allblocks[f"h.{block_id}.mlp.c_proj.bias"]
    # b_o = GEN.normal(0, 1, d_model)

    # Pre-Norm + residual connection
    X_orig = X.copy()
    X = layernorm(X)
    X = X * W_ln1 + b_ln1

    X = multihead_attention(X, 12, W_allblocks, block_id)

    # TODO add dropout for training
    X += X_orig
    X_orig = X.copy()

    # Post-Norm
    X = layernorm(X)
    X = X * W_ln2 + b_ln2

    # 2-layer MLP with residual connectionp2 
    X = X @ W_h + b_h
    X = gelu(X)
    X = X @ W_o + b_o
    # TODO add dropout for training
    X += X_orig

    return X

def gpt2_infere(text: str):
    MAX_SENTENCE_LEN = 1024

    tokenizer = GPT2Tokenizer(
        vocab="./vocab.json",
        merges="./merges.txt"
    )

    weights = {}
    with safe_open("model.safetensors", framework="np", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    print("User query:", text)
    tokens = tokenizer.encode(text)

    wpe = weights["wpe.weight"]
    position_embeddings = wpe[np.arange(len(tokens))]
    wte = weights["wte.weight"]
    token_embeddings = wte[tokens]
    X_in = token_embeddings + position_embeddings

    print("\033[95mGPT:", end="", flush=True)
    for _ in range(20):
        assert(len(tokens) <= MAX_SENTENCE_LEN)
        X = X_in

        for i in range(12):
            X = transformer_block(X, weights, i)

        X = layernorm(X)
        X = X * weights["ln_f.weight"] + weights["ln_f.bias"]

        X = X @ wte.T # d_model -> vocab_size

        # greedy decoding; or we could use top-k/top-p/temperature instead
        selected_token_id = np.argmax(X[-1])
        
        word = tokenizer.decode([int(selected_token_id)])
        print(word, end="", flush=True)
        if word.endswith("."):
            break

        tokens.append(selected_token_id)
        position_embeddings = wpe[len(tokens)]
        token_embeddings = wte[selected_token_id]
        X_in = np.vstack((X_in, token_embeddings + position_embeddings))
    print("\033[0m")

def run_gpt2():
    gpt2_infere("Chelsea want to win the Premier League")
    gpt2_infere("Tottenham want to win the Premier League")
    gpt2_infere("Liverpool want to win the Premier League")
    gpt2_infere("Arsenal want to win the Premier League")
    gpt2_infere("Brighton want to win the Premier League")

run_gpt2()