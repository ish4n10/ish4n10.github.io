---
layout: post
title: "Implementing GPT Architecture From Scratch: Training and Output"
tags:
  - Machine Learning
---

> This is a follow-up of my previous post "[Implementing GPT Architecture From Scratch: **A Deep Dive into Transformers and Attention**](https://ish4n10.hashnode.dev/gpt-from-scratch-part-1)**"**

This will be a very short post explaining how i trained the untrained gpt in my own PC just to get some valid output.

First, I'm using the OpenWebText that was used to train GPT-2 and currently just use a single file, as I am using my own PC for this and this takes some time.

### Parsing Dataset

```python
enc = tiktoken.get_encoding("gpt2")

def prepare_data(path: str):
    os.makedirs("data", exist_ok=True)

    df = pd.read_parquet(path)
    texts = df["text"].tolist()

    tokens = []
    for i, text in enumerate(texts):
        tokens.extend(enc.encode_ordinary(text))
        tokens.append(enc.eot_token)

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    torch.save(train, "data/train.pt")
    torch.save(val, "data/val.pt")
    return train, val

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)
```

Nothing fancy here, we prepare the data from the parquet file (you can download the dataset from hugging face), we make 90% training data and 10% validation data.

**get\_batch** - Picks random starting positions, slices out sequences of length 256. `y` is `x` shifted by one, at every position the model predicts the next token.

### Training model

```python
cfg = PocketConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_STEPS = 5000
EVAL_EVERY = 200
SAVE_EVERY = 1000
LR = 3e-4
MIN_LR = 3e-5
WARMUP = 200
GRAD_CLIP = 1.0
DATA_PATH = r""


def get_lr(step: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / (MAX_STEPS - WARMUP)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine


@torch.no_grad()
def eval_loss(model, val_data, eval_steps=50):
    model.eval()
    losses = [
        F.cross_entropy(
            model(x := get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)[0]).view(-1, cfg.vocab_size),
            get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)[1].view(-1)
        ).item()
        for _ in range(eval_steps)
    ]
    model.train()
    return sum(losses) / len(losses)

@torch.no_grad()
def eval_loss(model, val_data, eval_steps=50):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = get_batch(val_data, BATCH_SIZE, cfg.seq_len, DEVICE)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():

    train_data, val_data = prepare_data(DATA_PATH)

    model     = PocketTransformer(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    model.train()


    for step in range(MAX_STEPS):

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, cfg.seq_len, DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % EVAL_EVERY == 0 and step > 0:
            val = eval_loss(model, val_data)
```

The learning rate schedular, "get\_lr", will give us the learning rate based on the number of steps we are in currently,

Warmup is required as the weights are random at the start, so we should not push it to a very large difference.

For every step:

*   Compute what lr should be right now
    
*   Grab 16 random sequences of 256 tokens, run them through the model and get predictions
    
*   Measure how wrong predictions are (cross entropy) and compute gradients
    
*   Update weights.
    

### Text Generation

```python
enc = tiktoken.get_encoding("gpt2")
CHECKPOINT = "trained.pt"

def load_model():
    model = PocketTransformer(cfg).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def generate(
    model,
    prompt:      str,
    max_tokens:  int   = 200,
    temperature: float = 0.8,
    top_k:       int   = 40,
) -> str:

    tokens = enc.encode_ordinary(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, seq)

    for _i in range(max_tokens):
        x_crop = x[:, -cfg.seq_len:]

        logits = model(x_crop)          # (1, seq, vocab_size)
        logits = logits[:, -1, :]       # last token only (1, vocab_size)

        # temperature
        logits = logits / temperature

        # top-k zero out everything outside top k
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_val, float('-inf'))

        # sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # append
        x = torch.cat([x, next_token], dim=1)

        # stop at end of text token
        if next_token.item() == enc.eot_token:
            break

    return enc.decode(x[0].tolist())


if __name__ == "__main__":
    model = load_model()

    prompts = [
        "The meaning of life is",
        "In the beginning",
        "The best way to learn machine learning is",
    ]

    for prompt in prompts:
        print(f"\nprompt: {prompt}")
        print("-" * 50)
        print(generate(model, prompt))
        print("\\n")
```

### Training and Output

I set LR = 1e-3 based on something i read online,

```plaintext
step   0 | loss 11.02 | lr 1.00e-03 | grad 1.34
step  50 | loss 9.41  | lr 7.65e-03 | grad 4.21
step 100 | loss 9.87  | lr 1.51e-02 | grad 8.73
step 150 | loss NaN   | lr 2.26e-02 | grad inf
```

The loss went NaN and grad exploded, I dropped LR to 3e-4 and it worked fine.

Even though the data i trained on, almost 20M tokens, is *very* small to actually get a feasable, we do get something that we can say got from our dataset.

```plaintext
prompt: The meaning of life is

The meaning of life is to be the one that the country is not only that the American people can be involved in the world.

For every time I'm not happy with my child, no, I mean. But I'm still going to get my friends and I'm just going to be looking into me. So I'm making people who really know they're going to be a great deal," he said.

I never thought that this is the only point. I love it, but I don't have to mention the most amazing things that will be. I have already done to him, but I'll have a couple of dollars to go there, but I don't want to be a real, right at that time to do. I don't think they have a lot of people who think I've been trying to be very hard. I don't see them. I don't


prompt: In the beginning

In the beginning of the previous year, the State Department began to hold up the U.S. and be in place in August.

The Washington Post reported the report "the U.S. President Donald Trump to the United States, at the top of the EU's top election.

The U.S. Army and the Syrian government at the end of the country is the U.S. Army.

"We don't take the same situation that would be much more of the country this season. We can get more money to the party, but we should be able to pay a step at the right time in our business.

"The same way we need to be able to fight the next election of the European Union, which is a different person with our nation," said a director, who served as a director of the House, a "disappful" in the American government.
```

Well, that's it. I am not gonna train it in 200M tokens of data as it is completely useless and waste of time, but this was just something i wanted to make and write about so i can understand it well too.

The code is available at [https://github.com/ish4n10/pocket-transformer](https://github.com/ish4n10/pocket-transformer) if you wanna tinker and try something






