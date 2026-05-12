"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
from tinygrad import Tensor, TinyJit # tensors, autograd, kernel cache
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
for p in params: p.requires_grad = True # the (... * std) above strips requires_grad from the leaf
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
# tinygrad: the forward processes the WHOLE sequence in one call with a causal mask (vs the per-token KV-cache loop)
def linear(x, w):
    return x @ w.T

def softmax(logits):
    return logits.softmax()

def rmsnorm(x):
    ms = (x * x).mean(axis=-1, keepdim=True)
    scale = (ms + 1e-5) ** -0.5
    return x * scale

def gpt(tokens):
    T = tokens.shape[0]
    tok_emb = state_dict['wte'][tokens] # token embedding
    pos_emb = state_dict['wpe'][:T] # position embedding
    x = tok_emb + pos_emb # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        q = q.reshape(T, n_head, head_dim).transpose(0, 1)
        k = k.reshape(T, n_head, head_dim).transpose(0, 1)
        v = v.reshape(T, n_head, head_dim).transpose(0, 1)
        attn_logits = (q @ k.transpose(-2, -1)) / head_dim**0.5
        mask = Tensor.ones(T, T).tril() # causal mask
        attn_logits = mask.where(attn_logits, -float('inf'))
        attn_weights = attn_logits.softmax(axis=-1)
        x_attn = attn_weights @ v
        x_attn = x_attn.transpose(0, 1).reshape(T, n_embd)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
# tinygrad: buffers are Tensors (not Python scalars) so the whole optimizer step compiles into the JIT graph
class Adam:
    def __init__(self, params, lr, b1, b2, eps):
        self.params, self.b1, self.b2, self.eps = params, b1, b2, eps
        self.lr = Tensor([lr], requires_grad=False).contiguous()
        self.m = [Tensor.zeros(*p.shape, requires_grad=False).contiguous() for p in params] # first moment buffer
        self.v = [Tensor.zeros(*p.shape, requires_grad=False).contiguous() for p in params] # second moment buffer
        self.b1_t = Tensor.ones(1, requires_grad=False).contiguous()
        self.b2_t = Tensor.ones(1, requires_grad=False).contiguous()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.b1_t.assign(self.b1_t * self.b1)
        self.b2_t.assign(self.b2_t * self.b2)
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i].assign(self.b1 * self.m[i] + (1 - self.b1) * g)
            self.v[i].assign(self.b2 * self.v[i] + (1 - self.b2) * g * g)
            m_hat = self.m[i] / (1 - self.b1_t)
            v_hat = self.v[i] / (1 - self.b2_t)
            p.assign(p.detach() - self.lr * m_hat / (v_hat.sqrt() + self.eps))

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = Adam(params, lr=learning_rate, b1=beta1, b2=beta2, eps=eps_adam)

# Repeat in sequence
def train(num_steps=1000): # number of training steps
    Tensor.training = True # tinygrad: required for backward()

    # tinygrad: @TinyJit traces this once and replays the compiled kernels on every later call. Padding
    # every doc to block_size (below) keeps the graph shape identical every step.
    @TinyJit
    def step_fn(input_tokens, target_tokens, mask):
        optimizer.zero_grad()
        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        logits = gpt(input_tokens)
        probs = softmax(logits)
        per_pos_loss = -probs[Tensor.arange(block_size), target_tokens].log()
        loss = (per_pos_loss * mask).sum() / mask.sum() # final average loss over the document sequence. May yours be low.
        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()
        # Adam optimizer update: update the model parameters based on the corresponding gradients
        optimizer.step()
        # tinygrad: the JIT only traces ops whose outputs are realized; without this realize() the
        # side-effect .assign()s on m/v/b1_t/b2_t/params get dropped on JIT replay.
        Tensor.realize(loss, *params, *optimizer.m, *optimizer.v, optimizer.b1_t, optimizer.b2_t)
        return loss

    for step in range(num_steps):

        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        valid_n = min(block_size, len(tokens) - 1)
        tokens = (tokens + [BOS] * (block_size + 1 - len(tokens)))[:block_size + 1] # pad to fixed shape for JIT

        input_tokens = Tensor(tokens[:-1])
        target_tokens = Tensor(tokens[1:])
        mask = Tensor([1.0] * valid_n + [0.0] * (block_size - valid_n)) # only score valid positions

        optimizer.lr.assign(Tensor([learning_rate * (1 - step / num_steps)])) # linear learning rate decay
        loss = step_fn(input_tokens, target_tokens, mask)

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference: may the model babble back to us
def infer(temperature=0.5, num_samples=20): # in (0, 1], control the "creativity" of generated text, low to high
    Tensor.training = False
    print("\n--- inference (new, hallucinated names) ---")

    # tinygrad: same shape-stability trick as train() -- pad to block_size so the JIT compiles once. The
    # causal mask in gpt() makes the padded tail invisible to the position we actually sample from.
    @TinyJit
    def step_fn(input_tokens):
        return gpt(input_tokens)

    for sample_idx in range(num_samples):
        tokens = [BOS]
        for pos_id in range(block_size):
            padded = tokens + [BOS] * (block_size - len(tokens))
            all_logits = step_fn(Tensor(padded))
            probs = softmax(all_logits[len(tokens) - 1] / temperature) # sample from the last real position
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            tokens.append(token_id)
        print(f"sample {sample_idx+1:2d}: {''.join(uchars[t] for t in tokens[1:])}")
train()
infer()