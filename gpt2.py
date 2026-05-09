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

class MultiHeadAttention:
    def __init__(self, n_heads: int, d_model: int, W_allblocks: dict, block_id: int, use_kv_cache: bool):
        self.n_heads = n_heads
        self.d_head = d_model // self.n_heads

        self.W_o = W_allblocks[f"h.{block_id}.attn.c_proj.weight"]
        # self.W_o = GEN.normal(0, 1, (d_model, d_model))
        self.b_o = W_allblocks[f"h.{block_id}.attn.c_proj.bias"]
        # self.b_o = GEN.normal(0, 1, (d_model,))

        W_qkv_all = W_allblocks[f"h.{block_id}.attn.c_attn.weight"].reshape((d_model, 3, n_heads, self.d_head))
        self.W_q_all = W_qkv_all[:, 0, :, :]
        self.W_k_all = W_qkv_all[:, 1, :, :]
        self.W_v_all = W_qkv_all[:, 2, :, :]

        b_qkv_all = W_allblocks[f"h.{block_id}.attn.c_attn.bias"].reshape((3, n_heads, self.d_head))
        self.b_q_all, self.b_k_all, self.b_v_all = b_qkv_all

        self.use_kv_cache = use_kv_cache
        self.kv_cache = [None] * n_heads

    def forward(self, tokens: np.array):
        seq_len = tokens.shape[0]
        out = np.empty((seq_len,0))
        for i in range(self.n_heads):
            W_q = self.W_q_all[:, i, :]
            b_q = self.b_q_all[i]
            W_k = self.W_k_all[:, i, :]
            b_k = self.b_k_all[i]
            W_v = self.W_v_all[:, i, :]
            b_v = self.b_v_all[i]

            if self.use_kv_cache:
                if self.kv_cache[i] is None:
                    # prefill
                    K, V = tokens @ W_k + b_k, tokens @ W_v + b_v
                else:
                    # decode
                    K, V = self.kv_cache[i]
                    K, V = np.vstack((K, tokens[-1:] @ W_k + b_k)), np.vstack((V, np.array(tokens[-1]) @ W_v + b_v))
                self.kv_cache[i] = K, V
            else:
                K, V = tokens @ W_k + b_k, tokens @ W_v + b_v
            out = np.hstack((out, attention(tokens @ W_q + b_q, K, V)))

        return out @ self.W_o + self.b_o

class TransformerBlock:
    def __init__(self, W_allblocks: dict, block_id: int, d_model: int, use_kv_cache: bool):
        # d_model = X.shape[-1]
        # d_hidden = d_model * 4
        
        self.W_ln1 = W_allblocks[f"h.{block_id}.ln_1.weight"]
        # self.W_ln1 = GEN.normal(0, 1, (d_model,))
        self.b_ln1 = W_allblocks[f"h.{block_id}.ln_1.bias"]
        # self.b_ln1 = GEN.normal(0, 1, (d_model,))
        self.W_ln2 = W_allblocks[f"h.{block_id}.ln_2.weight"]
        # self.W_ln2 = GEN.normal(0, 1, (d_model,))
        self.b_ln2 = W_allblocks[f"h.{block_id}.ln_2.bias"]
        # self.b_ln2 = GEN.normal(0, 1, (d_model,))
        self.W_h = W_allblocks[f"h.{block_id}.mlp.c_fc.weight"]
        # self.W_h = GEN.normal(0, 1, (d_model, d_hidden))
        self.b_h = W_allblocks[f"h.{block_id}.mlp.c_fc.bias"]
        # self.b_h = GEN.normal(0, 1, d_hidden)
        self.W_o = W_allblocks[f"h.{block_id}.mlp.c_proj.weight"]
        # self.W_o = GEN.normal(0, 1, (d_hidden, d_model))
        self.b_o = W_allblocks[f"h.{block_id}.mlp.c_proj.bias"]
        # self.b_o = GEN.normal(0, 1, d_model)

        self.multihead_attention = MultiHeadAttention(12, d_model, W_allblocks, block_id, use_kv_cache=use_kv_cache)

    def forward(self, X: np.array):
        # Pre-Norm + residual connection
        X_orig = X.copy()
        X = layernorm(X)
        X = X * self.W_ln1 + self.b_ln1

        X = self.multihead_attention.forward(X)

        # TODO add dropout for training
        X += X_orig
        X_orig = X.copy()

        # Post-Norm
        X = layernorm(X)
        X = X * self.W_ln2 + self.b_ln2

        # 2-layer MLP with residual connectionp2 
        X = X @ self.W_h + self.b_h
        X = gelu(X)
        X = X @ self.W_o + self.b_o
        # TODO add dropout for training
        X += X_orig

        return X

def gpt2_infere(text: str, use_cache: bool = True):
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

    transformer_blocks = [TransformerBlock(weights, i, X_in.shape[-1], use_cache) for i in range(12)]

    print("\033[95mGPT:", end="", flush=True)
    for _ in range(20):
        assert(len(tokens) <= MAX_SENTENCE_LEN)
        X = X_in

        for block in transformer_blocks:
            X = block.forward(X)

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
    gpt2_infere("Arsenal want to win the Premier League", use_cache=False)
    gpt2_infere("Brighton want to win the Premier League", use_cache=False)

run_gpt2()