"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
from tinygrad import Tensor, TinyJit, nn # tensors, autograd, optimizers, kernel cache
random.seed(int(os.environ.get('MICROGPT_SEED', 42))) # Let there be order among chaos
Tensor.manual_seed(int(os.environ.get('MICROGPT_SEED', 42)))

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
def load_dataset(input_url=os.environ.get('MICROGPT_INPUT_URL', 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt')):
    fname = input_url.rsplit('/', 1)[-1]
    if not os.path.exists(fname):
        import urllib.request
        urllib.request.urlretrieve(input_url, fname)
    return [line.strip() for line in open(fname) if line.strip()]
docs = load_dataset()
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
# tinygrad provides this: Tensors track ops; .backward() walks the graph; the optimizer flips on requires_grad.

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: Tensor.randn(nout, nin) * std
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(state_dict.values()) # collect the trainable Tensors
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping a sequence of tokens to logits over what comes next.
# Follow GPT-2 with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU.
# Idiomatic tinygrad: the forward processes the WHOLE sequence (shape (T, n_embd)) in one call with a
# causal attention mask, instead of the per-token KV-cache loop the other two files use. This keeps the
# graph shape stable across positions, which is how tinygrad expects to be used.
def linear(x, w):
    return x @ w.T # batched: x of shape (..., n_in) @ (n_in, n_out) -> (..., n_out)

def softmax(logits):
    return logits.softmax() # along last dim by default

def rmsnorm(x):
    ms = (x * x).mean(axis=-1, keepdim=True)
    scale = (ms + 1e-5) ** -0.5
    return x * scale

def gpt(tokens):
    T = tokens.shape[0]
    tok_emb = state_dict['wte'][tokens] # (T, n_embd) embedding lookup via fancy indexing
    pos_emb = state_dict['wpe'][:T]     # (T, n_embd) first T position embeddings
    x = tok_emb + pos_emb               # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block, causal
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']) # (T, n_embd)
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        # split into heads: (T, n_embd) -> (n_head, T, head_dim)
        q = q.reshape(T, n_head, head_dim).transpose(0, 1)
        k = k.reshape(T, n_head, head_dim).transpose(0, 1)
        v = v.reshape(T, n_head, head_dim).transpose(0, 1)
        attn_logits = (q @ k.transpose(-2, -1)) / head_dim**0.5 # (n_head, T, T)
        mask = Tensor.ones(T, T).tril()                         # 1 on/below diagonal, 0 above (causal)
        attn_logits = mask.where(attn_logits, -float('inf'))    # mask out future positions
        attn_weights = attn_logits.softmax(axis=-1)
        x_attn = attn_weights @ v                               # (n_head, T, head_dim)
        x_attn = x_attn.transpose(0, 1).reshape(T, n_embd)      # merge heads back
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head']) # (T, vocab_size)
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = nn.optim.Adam(params, lr=learning_rate, b1=beta1, b2=beta2, eps=eps_adam)

# Repeat in sequence
def train(num_steps=1000): # number of training steps
    Tensor.training = True # tinygrad: required for optimizer.step()

    # The forward+backward+update for ONE step, with shape-stable inputs (block_size).
    # @TinyJit traces the graph on the first call and replays the compiled kernels on every subsequent
    # call: one compile, then constant-time replay. Padding every doc to block_size is what makes this
    # safe -- the graph shape is identical every step.
    @TinyJit
    def step_fn(input_tokens, target_tokens, mask):
        optimizer.zero_grad()
        logits = gpt(input_tokens)                                                # (block_size, vocab_size)
        probs = softmax(logits)
        per_pos_loss = -probs[Tensor.arange(block_size), target_tokens].log()     # (block_size,)
        loss = (per_pos_loss * mask).sum() / mask.sum()                           # average over VALID positions only
        loss.backward()
        optimizer.step()
        return loss

    for step in range(num_steps):

        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        valid_n = min(block_size, len(tokens) - 1)
        # Pad with BOS to a fixed length so step_fn sees the same shape every call.
        tokens = (tokens + [BOS] * (block_size + 1 - len(tokens)))[:block_size + 1]

        input_tokens = Tensor(tokens[:-1])                                          # (block_size,)
        target_tokens = Tensor(tokens[1:])                                          # (block_size,)
        mask = Tensor([1.0] * valid_n + [0.0] * (block_size - valid_n))             # only score valid positions

        # Adam optimizer update: linear learning rate decay; .assign() mutates the lr tensor in place
        # so the JIT'd step_fn picks up the new value without re-tracing.
        optimizer.lr.assign(Tensor([learning_rate * (1 - step / num_steps)]))
        loss = step_fn(input_tokens, target_tokens, mask)

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference: may the model babble back to us
def infer(temperature=0.5, num_samples=20): # in (0, 1], control the "creativity" of generated text, low to high
    Tensor.training = False
    print("\n--- inference (new, hallucinated names) ---")

    # Same shape-stability trick as train(): pad to block_size on every call so step_fn sees one shape
    # and JIT compiles once. The causal mask in gpt() guarantees the logits at position i depend only on
    # positions 0..i, so the padded tail can be filled with anything (we use BOS) without affecting the
    # logits we actually sample from.
    @TinyJit
    def step_fn(input_tokens):
        return gpt(input_tokens) # (block_size, vocab_size)

    for sample_idx in range(num_samples):
        tokens = [BOS]
        for pos_id in range(block_size):
            padded = tokens + [BOS] * (block_size - len(tokens))   # pad to block_size with BOS
            all_logits = step_fn(Tensor(padded))                   # (block_size, vocab_size), one cached graph
            probs = softmax(all_logits[len(tokens) - 1] / temperature) # sample from the last REAL position
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            tokens.append(token_id)
        print(f"sample {sample_idx+1:2d}: {''.join(uchars[t] for t in tokens[1:])}")
train()
infer()