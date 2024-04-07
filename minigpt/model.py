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
    dropout_rate: float
    
class MultiHeadAttention(nn.Module):
    n_embd: int
    n_head: int
    block_size: int
    train: bool
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x):
       head_dim = self.n_embd // self.n_head
       # Project the input to the query, key, and value tensors for each head
       query = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)(x) # (B,T,H,D)
       key = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)(x) # (B,T,H,D)
       value = nn.DenseGeneral(features=(self.n_head, self.n_embd // self.n_head), axis=-1, use_bias=False)(x) # (B,T,H,D)

       # Compute the attention scores ("affinities")
       wei = jnp.einsum('...qhd,...khd->...hqk', query, key)*head_dim**-0.5 # (B,H,T,T)
       trill = jnp.tril(jnp.ones((x.shape[1],x.shape[1]))) # (T,T)
       wei = jnp.where(trill == 0, -jnp.inf, wei) # (B,T,T)
       wei = jax.nn.softmax(wei, axis=-1) # (B,H,T,T)
       wei =  nn.Dropout(rate=self.dropout_rate,  deterministic=not self.train)(wei) # (B,H,T,T)
       
      # Perform weighted aggregation of values
       wei = jnp.einsum('...hqk,...khd->...qhd', wei, value)
       wei = nn.DenseGeneral(features=self.n_embd, axis=(-2,-1), use_bias=False)(wei) # (B,T,C)
       wei = nn.Dense(self.n_embd , use_bias=False)(wei) # (B,T,C)
       out =  nn.Dropout(rate=self.dropout_rate, deterministic=not self.train)(wei) # (B,T,C)
       return out

class MLP(nn.Module):
    n_embd: int
    train: bool
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.n_embd, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_embd, use_bias=False)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.train)(x)
        return x
    
class Block(nn.Module):
    n_embd: int
    n_head: int
    block_size: int
    train: bool
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x):
        res = x
        x = nn.LayerNorm(use_bias=False)(x)
        x = res + MultiHeadAttention(self.n_embd, self.n_head, self.block_size, self.train, self.dropout_rate)(x)
        x = nn.LayerNorm(use_bias=False)(x)
        x = res + MLP(self.n_embd, self.train, self.dropout_rate)(x)
        return x
    
class Gpt(nn.Module):
    vocab_size: int
    block_size: int
    n_embd: int
    n_heads: int
    n_blocks: int
    train: bool
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, idx):
        b, t = idx.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32) # shape (t)

        # forward the GPT model itself
        tok_emb =  nn.Embed(self.vocab_size, self.n_embd)(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = nn.Embed(self.block_size, self.n_embd)(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for _ in range(self.n_blocks):
            x = Block(self.n_embd, self.n_heads, self.block_size, self.train, self.dropout_rate)(x)        
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits =  nn.Dense(self.vocab_size, use_bias=False)(x) 
        return logits

def generate(key, model, params, idx, block_size, max_new_tokens, temperature=1.0):
    """
    Take a conditioning sequence of indices idx (jax.Array of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits =  model.apply(params, idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature

        # Split random key
        state, key = jax.random.split(key)

        # sample from the distribution
        idx_next = jax.random.categorical(state, logits, shape=(logits.shape[0],1))

        # append sampled index to the running sequence and continue
        idx = jnp.concatenate([idx,idx_next], axis=1)

    return idx


