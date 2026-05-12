"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
import torch    # tensors, autograd, optimizer base class
import torch.nn as nn # nn.Parameter
random.seed(int(os.environ.get('MICROGPT_SEED', 42))) # Let there be order among chaos
torch.manual_seed(int(os.environ.get('MICROGPT_SEED', 42)))

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
# PyTorch provides this: tensors with requires_grad=True track ops; .backward() walks the graph.

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: nn.Parameter(torch.randn(nout, nin) * std)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(state_dict.values()) # collect the trainable Parameter tensors
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return w @ x

def softmax(logits):
    exps = (logits - logits.max()).exp()
    return exps / exps.sum()

def rmsnorm(x):
    ms = (x * x).mean()
    scale = (ms + 1e-5) ** -0.5
    return x * scale

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = tok_emb + pos_emb # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = torch.stack([ki[hs:hs+head_dim] for ki in keys[li]])
            v_h = torch.stack([vi[hs:hs+head_dim] for vi in values[li]])
            attn_logits = (k_h @ q_h) / head_dim**0.5
            attn_weights = softmax(attn_logits)
            head_out = attn_weights @ v_h
            x_attn.append(head_out)
        x_attn = torch.cat(x_attn)
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
class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.85, 0.99), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p) # first moment buffer
                    state['v'] = torch.zeros_like(p) # second moment buffer
                state['step'] += 1
                t = state['step']
                state['m'].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = Adam(params, lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

# Repeat in sequence
def train(num_steps=1000): # number of training steps
    for step in range(num_steps):

        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = sum(losses) / n # final average loss over the document sequence. May yours be low.

        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()

        # Adam optimizer update: update the model parameters based on the corresponding gradients
        for g in optimizer.param_groups:
            g['lr'] = learning_rate * (1 - step / num_steps) # linear learning rate decay
        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference: may the model babble back to us
@torch.no_grad()
def infer(temperature=0.5, num_samples=20): # in (0, 1], control the "creativity" of generated text, low to high
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits / temperature)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
train()
infer()