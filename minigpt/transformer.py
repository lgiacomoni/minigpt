from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform) # Check GPU is visible
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import functools

class Block(nn.Module):
    """Transformer block: communication followed by computation """
    num_heads: int
    n_embd: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(self.n_embd)(x)
        x = x + nn.MultiHeadAttention(num_heads=self.num_heads, 
                                      use_bias=False, 
                                      attention_fn=functools.partial(nn.dot_product_attention,mask=nn.make_causal_mask(x))
                                      )(x)
        x = nn.Dense(self.n_embd)(x)
        x = nn.LayerNorm(self.n_embd)(x)
        x = x + nn.Sequential([nn.Dense(4*self.n_embd), nn.relu, nn.Dense(self.n_embd)])(x)

        return x
    
class FeedForward(nn.Module):
    """Position-wise feed-forward layer"""
    n_embd: int

    @nn.compact
    def __call__(self, x):
        x = nn.Sequential([nn.Dense(4*self.n_embd), nn.relu, nn.Dense(self.n_embd)])(x)
        return x

class SingleHeadAttention(nn.Module):
    """Single head attention"""
    head_size: int

    @nn.compact
    def __call__(self, x):
       # Project the input to the query, key, and value tensors
        query = nn.Dense(self.head_size, use_bias=False)(x) # (B,T,head_size)
        key = nn.Dense(self.head_size, use_bias=False)(x) # (B,T,head_size)
        value = nn.Dense(self.head_size, use_bias=False)(x) # (B,T,head_size)

        # Compute the attention scores ("affinities")
        wei = query @ key.transpose((0,2,1)) * x.shape[-1]**-0.5 # (B,T,T)
        trill = jnp.tril(jnp.ones((x.shape[1],x.shape[1]))) # (T,T)
        wei = jnp.where(trill == 0, -jnp.inf, wei) # (B,T,T)
        wei = jax.nn.softmax(wei, axis=-1) # (B,T,T)

        # Perform weighted aggregation of values
        out = wei @ value # (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    num_heads: int

    @nn.compact
    def __call__(self, x):
       head_dim = x.shape[-1] // self.num_heads
       # Project the input to the query, key, and value tensors for each head
       query = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, use_bias=False)(x) # (B,T,H,D)
       key = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, use_bias=False)(x) # (B,T,H,D)
       value = nn.DenseGeneral(features=(self.num_heads, head_dim), axis=-1, use_bias=False)(x) # (B,T,H,D)

       # Compute the attention scores ("affinities")
       wei = jnp.einsum('...qhd,...khd->...hqk', query, key)*head_dim**-0.5 # (B,H,T,T)
       trill = jnp.tril(jnp.ones((x.shape[1],x.shape[1]))) # (T,T)
       wei = jnp.where(trill == 0, -jnp.inf, wei) # (B,T,T)
       wei = jax.nn.softmax(wei, axis=-1) # (B,H,T,T)

       
      # Perform weighted aggregation of values
       wei = jnp.einsum('...hqk,...khd->...qhd', wei, value)
       wei = nn.DenseGeneral(features=x.shape[-1], axis=(-2,-1), use_bias=False)(wei) # (B,T,C)
       out = nn.Dense(x.shape[-1], use_bias=False)(wei) # (B,T,C)
       return out



class Block(nn.Module):
   n_embd: int
   n_head: int

   @nn.compact
   def __call__(self, x):
      x = nn.LayerNorm(self.n_embd)(x)
      x = x + MultiHeadAttention(self.n_head)(x)
      x = nn.LayerNorm(self.n_embd)(x)
      x = x + FeedForward(self.n_embd)(x)
      return x

class BigramLanguageModel(nn.Module):
  vocab_size: int
  n_embd: int
  block_size: int
  num_heads: int  

  @nn.compact
  def __call__(self, idx):
    B,T = idx.shape

    token_emb = nn.Embed(self.vocab_size, self.n_embd)(idx) # (B,T,C) with C: embedding size/channels
    # TODO: improve position embedding
    position_emb = nn.Embed(self.block_size, self.n_embd)(jnp.arange(T)) # (T,C)
    x = token_emb + position_emb # (B,T,C)
    x = Block(self.n_embd, self.num_heads)(x)
    x = Block(self.n_embd, self.num_heads)(x)
    x = Block(self.n_embd, self.num_heads)(x)
    x = Block(self.n_embd, self.num_heads)(x)
    x = Block(self.n_embd, self.num_heads)(x)
    x = Block(self.n_embd, self.num_heads)(x)
    x = nn.LayerNorm(self.n_embd)(x) # (B,T,C)
    logits = nn.Dense(self.vocab_size, use_bias=False)(x) # (B,T,C)

    return logits

def load_data(path):
    with open(path, 'r') as fin:
        text = fin.read()
    return text

def create_encoder_decoder_fns(chars):
    stoi = {char:code for code,char in enumerate(chars)}
    itos = {code:char for code,char in enumerate(chars)}

    encode = lambda text: [stoi[c] for c in text]
    decode = lambda code: ''.join([itos[c] for c in code])

    return encode, decode

def train_val_split(data, split=0.9):
    n = int(split*len(data))
    train_data = data[:n]
    val_data = data[n:]

    data = {'train': train_data, 'val': val_data}
    return data

# TODO: Can't be jitted right now. Can we modify
def get_batch(split, data):
  data = data[split]
  ix = np.random.randint(0, len(data) - block_size, (batch_size,),  dtype=np.int32)
  x = np.stack([data[i:i+block_size] for i in ix])
  y =  np.stack([data[i+1:i+block_size+1] for i in ix])
  return jax.device_put(x), jax.device_put(y)

#TODO: Optimise this function
def generate(key, model, params, idx, max_new_tokens):
  # idx is (B,T) array of indices in the current context
  for _ in range(max_new_tokens):
    #Crop idx to the last block_size tokens
    idx_cond = idx[:,-block_size:]

    # get the predictions
    logits = model.apply(params, idx_cond)

    # focus only on the last timestamp
    logits = logits[:,-1,:]

    # Split random key
    key1, key = jax.random.split(key)
    #sample from the distribution
    idx_next = jax.random.categorical(key1, logits, shape=(logits.shape[0],1))

    # append sampled index to returning sequence
    idx = jnp.concatenate([idx,idx_next], axis=1)

  return idx

def create_loss_fn(model):
    def cross_entropy_loss(params,inputs,targets):
      logits = model.apply(params, inputs)
      n_classes = logits.shape[-1]
      loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, n_classes)).mean()
      return loss

    return cross_entropy_loss

def estimate_loss(model, params, data, eval_iters=200):
    out = {}
    loss_fn = jax.jit(create_loss_fn(model), device=jax.devices('gpu')[0])
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for i in range(eval_iters):
           xs, ys = get_batch(split, data)
           losses[i] = loss_fn(params, xs, ys)
    
        out[split] = losses.mean()
    return out
   
def masked_fill(mask,a,fill_value):
   return jax.lax.select(mask, a, jax.lax.broadcast(fill_value, a.shape))

   
if __name__ == '__main__':
    # Initialise optimiser params
    start_learning_rate = 1e-3
    batch_size = 32 # Number of independent sequences we will process in parallel
    block_size = 256 # Maximum context length of predictions
    seed = 1337
    max_iters = 5000
    eval_interval = 500
    vocab_size = 65
    n_embd = 384
    n_head = 6

    np.random.seed(seed)
   
    text = load_data('input.txt')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print(f"length of dataset is: {len(text)}")
    print(''.join(chars))
    print(vocab_size)

    encode, decode = create_encoder_decoder_fns(chars)

    print(encode('hi there'))
    print(decode(encode('hi there')))


    # Encoded dataset
    data = np.array(encode(text), dtype=np.int32) #TODO: setting jnp.int64 returns a warning about downsizing and requiring an env variable to be set in order to use int64
    print(data.shape, data.dtype)

    data = train_val_split(data, split=0.9)

    init_key, training_key = jax.random.split(jax.random.PRNGKey(seed))
    
    xb, yb = get_batch('train', data)

    model = BigramLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, num_heads=n_head)

    # Initialise model parameters
    params = jax.device_put(model.init(init_key, xb))

    optimizer = optax.adamw(learning_rate=start_learning_rate)
    opt_state = jax.device_put(optimizer.init(params))

    loss_fn = jax.jit(create_loss_fn(model),  device=jax.devices('gpu')[0])

    for i in range(max_iters):
        training_key, key = jax.random.split(training_key)
        
        if i % eval_interval == 0:
           losses = estimate_loss(model, params, data)
           print(f"Step {i}, Train loss: {losses['train']}, Val loss: {losses['val']}")

        xs, ys = get_batch('train', data)
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    idx = jnp.zeros((1,1), dtype=jnp.int32, device=jax.devices('gpu')[0])
    print(decode(generate(key,model,params, idx, max_new_tokens=500)[0].tolist())) 