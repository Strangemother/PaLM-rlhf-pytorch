import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator

# constants

NUM_BATCHES = int(15)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 3
PRIME_LENGTH = 128
GENERATE_EVERY = 1
GENERATE_LENGTH = 150
SEQ_LEN = 200

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def encode_text(val):
    ff = np.fromstring(val, dtype=np.uint8)
    return torch.from_numpy(ff).long()
    # return tuple(map(ord, val))

# accelerator

accelerator = Accelerator()
device = accelerator.device

# instantiate palm

model = PaLM(
    num_tokens=128,
    dim=256,
    depth=8
).to(device)

# prepare enwik8 data

with gzip.open("./data/demo.gz") as file:
    X = np.fromstring(file.read(int(2000)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(1600)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Adam(model.palm_parameters(), lr=LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)

# training
def train():
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=2.0, desc="training"):
        model.train()

        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader), return_loss=True)
            accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

        accelerator.print(f"training loss: {loss.item()}")
        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader), return_loss=True)
                accelerator.print(f"validation loss: {loss.item()}")

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:PRIME_LENGTH]
            ask_ints(inp)

    loss = model(next(train_loader), return_loss=True)
    accelerator.print(f"training loss: {loss.item()}")
    return loss


## ff = encode_text('apples')
## decode_tokens(model.generate(10, torch.from_numpy(ff).long()[None,...])[0])

def ask(v):
    return ask_ints(encode_text(v))


def ask_ints(inp):
    prime = decode_tokens(inp)
    accelerator.print(f"{prime} \n\n")
    sample = model.generate(GENERATE_LENGTH, inp[None, ...])
    output_str = decode_tokens(sample[0])
    accelerator.print(output_str, "\n")
    return output_str


# train()

from palm_rlhf_pytorch.ppo import RLHFTrainer as T
from palm_rlhf_pytorch.palm_rlhf_pytorch import PaLM, RewardModel, ActorCritic

def tk(strs):
    # torch.from_numpy(np.array(encode_text('hello'),dtype=np.uint8)[None,...])
    bb= np.array([encode_text(x) for x in strs])
    return torch.from_numpy(bb).long()

def text_to_uints(val):
    return np.frombuffer(bytes(val, 'utf'), dtype=np.uint8)

PROMPTS = [
    'hello'
]


def tk3(strs):
    # torch.from_numpy(np.array(encode_text('hello'),dtype=np.uint8)[None,...])
    # return text_to_uints(x)
    # bb= np.array([text_to_uints(x) for x in strs])
    # return torch.from_numpy(bb).long()
    return torch.Tensor(encode_text(PROMPTS[0]))

tk2 = lambda x: torch.from_numpy(np.array(encode_text(x),dtype=np.uint8)[None,...]).long()

def tk4(vv):
    return np.array(vv)

def create_trainer(_prompts=None):
    prompts = _prompts or PROMPTS
    rm = RewardModel(palm=model)
    tr = T(palm=model, reward_model=rm, prompts=prompts, tokenizer=tk3)
    return tr


tk = create_trainer()
tk.load()

def save():
    tk.save()

def qu(text, resp_len=50):
    res = tk.generate(resp_len, prompt=encode_text(text))
    return decode_tokens(res)


def qu_train(num_episodes=2, max_timesteps=3, update_timesteps=1):
    return tk.train(
            max_seq_len=PRIME_LENGTH,
            max_batch_size=BATCH_SIZE,
            num_episodes=num_episodes,
            max_timesteps=max_timesteps,
            update_timesteps=update_timesteps)
