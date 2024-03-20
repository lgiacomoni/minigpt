import jax
import jax.numpy as jnp
import model as gpt
import optax
import numpy as np
from flax.training import train_state  # a useful dataclass to keep train state
from functools import partial


def get_batch(key, split, data, block_size, batch_size, device):
  data = data[split]
  ix = jax.random.randint(key,  (batch_size,), 0,  len(data) - block_size, dtype=jnp.int32)
  x = jnp.stack([data[i:i+block_size] for i in ix])
  y =  jnp.stack([data[i+1:i+block_size+1] for i in ix])
  return jax.device_put(x,device=device), jax.device_put(y, device=device)


@jax.jit
@jax.vmap
def compute_loss(logits, targets): # (T,C),(T,)
      n_classes = logits.shape[-1]
      loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, n_classes))
      return loss

@jax.jit
def train_step(state, xs, ys): # [TrainState, (B,T,C), (B,T)] -> TrainState, float 
    # TODO: When the model is relatively small, a pmap of train step would allow to train in parallel over multiple batches. However, this requires
    # replication of the model params across devices. This is not possible with very large models. Instead, a single very large model should be trained
    # by splitting a single model into multiple logical parts. 
    
    def loss_fn(params):
        logits = state.apply_fn(params, xs)
        loss = jnp.mean(compute_loss(logits, ys))
        return loss, logits
  
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    return state, loss

@jax.jit
def eval_step(state, xs, ys): # state, (B,T,C), (B,T)
    logits = state.apply_fn(state.params, xs)
    loss = jnp.mean(compute_loss(logits, ys))
    return loss


def train_one_epoch(key, state, data, get_batch):
    """Train for 1 epoch on the training set."""
    xs,ys = get_batch(key, 'train', data)
    state, loss = train_step(state, xs, ys)

    # Aggregate the metrics
    loss = jax.device_get(loss)  # pull from the accelerator onto host (CPU)
    return state, loss

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

# This one will keep things nice and tidy compared to our previous examples
def create_train_state(key, model, input, learning_rate):
    params = model.init(key, input)
    sgd_opt = optax.adam(learning_rate)
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=jax.jit(model.apply), params=params, tx=sgd_opt)

def estimate_loss(key, state, data, get_batch, eval_iters=200):
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for i in range(eval_iters):
           xs, ys = get_batch(key, split, data)
           losses[i] = eval_step(state, xs, ys)
    
        out[split] = losses.mean()
    return out

def run():
    # Initialise optimiser params
    start_learning_rate = 1e-3
    batch_size = 16 # Number of independent sequences we will process in parallel
    block_size = 32 # Maximum context length of predictions
    seed = 1337
    max_iters = 1000
    eval_interval = 100
    vocab_size = 65
    n_embd = 64
    n_head = 4
    n_blocks = 4


    text = load_data('input.txt')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encode, decode = create_encoder_decoder_fns(chars)

     # Encoded dataset
    data = np.array(encode(text), dtype=np.int32) #TODO: setting jnp.int64 returns a warning about downsizing and requiring an env variable to be set in order to use int64
    data = train_val_split(data, split=0.9)

    init_key, train_key = jax.random.split(jax.random.PRNGKey(seed))

    
    xb, _ = get_batch(init_key, 'train', data, block_size, batch_size, jax.devices('gpu')[0])


    model = gpt.GPT(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_heads=n_head, n_blocks=n_blocks)
    train_state = create_train_state(init_key, model, xb, start_learning_rate)
    
    for epoch in range(1, max_iters + 1):
        train_key, key1, key2 = jax.random.split(train_key, num=3)

        train_state, _ = train_one_epoch(key1, train_state, data,partial(get_batch, block_size=block_size, batch_size=batch_size, device= jax.devices('gpu')[0]))
        if epoch % eval_interval == 0:
            losses = estimate_loss(key2, train_state, data, partial(get_batch, block_size=block_size, batch_size=batch_size, device= jax.devices('gpu')[0]))
            print(f"Train epoch: {epoch}, loss: {losses['train']}")
            print(f"Test epoch: {epoch}, loss: {losses['val']}")

    idx = jnp.zeros((1,1), dtype=jnp.int32, device=jax.devices('gpu')[0])
    print(decode(gpt.generate(train_key, train_state, idx, block_size, max_new_tokens=500)[0].tolist())) 

if __name__ == '__main__':
    run()