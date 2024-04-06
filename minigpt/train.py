from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from model import Gpt, generate, ModelConfig
import optax
import numpy as np
from flax.training import train_state  # a useful dataclass to keep train state
from functools import partial
from flax import linen as nn
import orbax.checkpoint as ocp
import flax

TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'

@flax.struct.dataclass
class TrainConfig:
    start_learning_rate: float
    batch_size: int
    seed: int
    max_iters: int
    eval_interval: int

def get_batch(key: jax.random.PRNGKey, split: str, data: jnp.ndarray, block_size: int, batch_size: int, device: jax.Device) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get a batch of data.

    Args:
        key: the random key
        split: the split of the data to get the batch from
        data: the data
        block_size: the block size
        batch_size: the batch size
        device: the device to put the data on

    Returns:
        A tuple of the input and target data.
    """
    data = data[split]
    ix = jax.random.randint(key,  (batch_size,), 0,  len(data) - block_size, dtype=jnp.int32)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y =  jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return jax.device_put(x,device=device), jax.device_put(y, device=device)


@jax.jit
@jax.vmap
def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> float: # (T,C),(T,)
    """Computes the loss between the logits and the targets.

    Args:
        logits: the model predictions
        targets: the target values

    Returns:
        The loss value.
    """
    n_classes = logits.shape[-1]
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, n_classes))
    return loss

@jax.jit
def eval_step(state: train_state.TrainState, xs: jnp.ndarray, ys: jnp.ndarray) -> float: # state, (B,T,C), (B,T)
    """Computes the loss.

    Args:
        state: the TrainState
        xs: the input data
        ys: the target data

    Returns:
        The loss value.
    """
    logits = state.apply_fn(state.params, xs)
    loss = jnp.mean(compute_loss(logits, ys))
    return loss

@jax.jit
def train_step(state: train_state.TrainState, xs: jnp.ndarray, ys: jnp.ndarray): # [TrainState, (B,T,C), (B,T)] -> TrainState, float 
    """Performs a single training step.
    Args:
        state: the TrainState
        xs: the input data
        ys: the target data
    
    Returns:
        A tuple of the new TrainState and the loss value.
    """
    def loss_fn(params):
        logits = state.apply_fn(params, xs)
        loss = jnp.mean(compute_loss(logits, ys))
        return loss
  
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    return state, loss

def train_one_epoch(key, state, data, get_batch):
    """Trains the model for one epoch.

    Args:
        key: the random key
        state: the TrainState
        data: the data
        get_batch: the function to get a batch of data

    Returns:
        The new TrainState and the average loss value.
    """
    xs,ys = get_batch(key, TRAIN_SPLIT, data)
    state, loss = train_step(state, xs, ys)

    # Aggregate the metrics
    loss = jax.device_get(loss)  # pull from the accelerator onto host (CPU)
    return state, loss

def load_data(path):
    with open(path, 'r') as fin:
        text = fin.read()
    return text

def create_encoder_decoder_fns(chars: List[str]) -> Tuple[Callable[[str],List[int]], Callable[[List[int]], str]]:
    """Create encoder and decoder functions.

    Args:
        chars: the characters to encode

    Returns:
        A tuple of the encode and decode functions.
    """
    stoi = {char:code for code,char in enumerate(chars)}
    itos = {code:char for code,char in enumerate(chars)}

    encode = lambda text: [stoi[c] for c in text]
    decode = lambda code: ''.join([itos[c] for c in code])

    return encode, decode

def train_val_split(data: np.array, split: float =0.9):
    """Split the data into training and validation sets.

    Args:
        data: the data
        split: the split ratio

    Returns:
        A dictionary with the training and validation data.
    """
    n = int(split*len(data))
    train_data = data[:n]
    val_data = data[n:]

    data = {TRAIN_SPLIT: train_data, VAL_SPLIT: val_data}
    return data

def create_train_state(key: jax.random.PRNGKey, model: nn.Module, input: jnp.ndarray, learning_rate:float):
    """Create the TrainState.

    Args:
        key: the random key
        model: the model
        input: the input data
        learning_rate: the learning rate
    Returns:
        The TrainState.
    """
    params = model.init(key, input)
    sgd_opt = optax.adam(learning_rate)
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=jax.jit(model.apply), params=params, tx=sgd_opt)

def estimate_loss(key: jax.random.PRNGKey, state: train_state.TrainState, data: np.array, get_batch: Callable, eval_iters: int=200) -> dict:
    """Estimate the loss.

    Args:
        key: the random key
        state: the TrainState
        data: the data
        get_batch: the function to get a batch

    Returns:
        A dictionary with the training and validation loss.
    """
    out = {}
    for split in [TRAIN_SPLIT, VAL_SPLIT]:
        losses = np.zeros(eval_iters)
        for i in range(eval_iters):
           xs, ys = get_batch(key, split, data)
           losses[i] = eval_step(state, xs, ys)
    
        out[split] = losses.mean()
    return out

def run():
    train_config = TrainConfig(start_learning_rate=1e-4, 
                               batch_size=64, 
                               seed=1337, 
                               max_iters=5000, 
                               eval_interval=500)
    
    model_config = ModelConfig(vocab_size=65, 
                               n_embd=384, 
                               block_size=128, 
                               n_head=8, 
                               n_blocks=6)
    
    text = load_data('data/input.txt')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encode, decode = create_encoder_decoder_fns(chars)

     # Encoded dataset
    data = np.array(encode(text), dtype=np.int32) #TODO: setting jnp.int64 returns a warning about downsizing and requiring an env variable to be set in order to use int64
    data = train_val_split(data, split=0.9)

    init_key, train_key = jax.random.split(jax.random.PRNGKey(train_config.seed))

    
    xb, _ = get_batch(init_key, TRAIN_SPLIT, data, model_config.block_size, train_config.batch_size, jax.devices('gpu')[0])


    model = Gpt(vocab_size=vocab_size, block_size=model_config.block_size, n_embd=model_config.n_embd, num_heads=model_config.n_head, n_blocks=model_config.n_blocks)
    train_state = create_train_state(init_key, model, xb, train_config.start_learning_rate)
    
    # Set up checkpointing
    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mngr = ocp.CheckpointManager( options=options)

    for epoch in range(1, train_config.max_iters + 1):
        train_key, key1, key2 = jax.random.split(train_key, num=3)

        train_state, _ = train_one_epoch(key1, train_state, data,partial(get_batch, block_size=model_config.block_size, batch_size=train_config.batch_size, device= jax.devices('gpu')[0]))
        if epoch % train_config.eval_interval == 0:
            losses = estimate_loss(key2, train_state, data, partial(get_batch, block_size=model_config.block_size, batch_size=train_config.batch_size, device= jax.devices('gpu')[0]))
            print(f"Epoch: {epoch}, Train loss: {losses[TRAIN_SPLIT]}, Val loss: {losses[VAL_SPLIT]}")
            ckpt = {"state": train_state, "config": {"train_config": train_config, "model_config": model_config}}
            mngr.save(epoch, args=ocp.args.StandardSave(ckpt))
                

    idx = jnp.zeros((1,1), dtype=jnp.int32, device=jax.devices('gpu')[0])
    print(decode(generate(train_key, train_state, idx, model_config.block_size, max_new_tokens=500)[0].tolist())) 

if __name__ == '__main__':
    run()