from flax import linen as nn
import flax
import math
import jax
import jax.numpy as jnp

@flax.struct.dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int
    n_blocks: int
    
    
class CausalSelfAttention(nn.Module):
    n_embd: int
    n_head: int
    block_size: int

    def setup(self):
        self.query = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)
        self.key = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)
        self.value = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)
        self.c_proj1 = nn.DenseGeneral(features=self.n_embd, axis=(-2,-1), use_bias=False)
        self.c_proj2 =  nn.Dense(self.n_embd , use_bias=False)
        
    def __call__(self, x):
       head_dim = self.n_embd // self.n_head
       # Project the input to the query, key, and value tensors for each head
       query = self.query(x) # (B,T,H,D)
       key = self.key(x) # (B,T,H,D)
       value = self.value(x) # (B,T,H,D)

       # Compute the attention scores ("affinities")
       wei = jnp.einsum('...qhd,...khd->...hqk', query, key)*head_dim**-0.5 # (B,H,T,T)
       trill = jnp.tril(jnp.ones((self.block_size,self.block_size))) # (T,T)
       wei = jnp.where(trill == 0, -jnp.inf, wei) # (B,T,T)
       wei = jax.nn.softmax(wei, axis=-1) # (B,H,T,T)

       
      # Perform weighted aggregation of values
       wei = jnp.einsum('...hqk,...khd->...qhd', wei, value)
       wei = self.c_proj1(wei) # (B,T,C)
       out = self.c_proj2(wei) # (B,T,C)
       return out

class MLP(nn.Module):
    n_embd: int

    def setup(self):
        self.c_fc    = nn.Dense(4 * self.n_embd, use_bias=False)
        self.relu    = nn.relu 
        self.c_proj  = nn.Dense(self.n_embd, use_bias=False)
        # self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        # x = self.dropout(x)
        return x
    
class Block(nn.Module):
    n_embd: int
    n_head: int
    block_size: int


    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=False)
        self.attn = CausalSelfAttention(self.n_embd, self.n_head, self.block_size)
        self.ln_2 = nn.LayerNorm(use_bias=False)
        self.mlp = MLP(self.n_embd)

    def __call__(self, x):
        res = x
        x = res + self.attn(self.ln_1(x))
        x = res + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    vocab_size: int
    block_size: int
    n_embd: int
    n_heads: int
    n_blocks: int

    def setup(self):

        self.token_emb = nn.Embed(self.vocab_size, self.n_embd)# (B,T,C) with C: embedding size/channels
        self.position_emb = nn.Embed(self.block_size, self.n_embd) # (T,C)
        self.blocks = [Block(self.n_embd, self.n_heads, self.block_size) for _ in range(self.n_blocks)]
        self.ln_f = nn.LayerNorm(use_bias=False)
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx):
        b, t = idx.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32) # shape (t)

        # forward the GPT model itself
        tok_emb = self.token_emb(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.position_emb(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        return logits

def generate(key, train_state, idx, block_size, max_new_tokens, temperature=1.0):
    """
    Take a conditioning sequence of indices idx (jax.Array of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits = train_state.apply_fn(train_state.params, idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature

        # Split random key
        state, key = jax.random.split(key)

        # sample from the distribution
        idx_next = jax.random.categorical(state, logits, shape=(logits.shape[0],1))

        # append sampled index to the running sequence and continue
        idx = jnp.concatenate([idx,idx_next], axis=1)

    return idx


