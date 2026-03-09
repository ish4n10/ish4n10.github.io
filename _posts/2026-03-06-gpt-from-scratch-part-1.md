---
layout: post
title: "Implementing GPT Architecture From Scratch: A Deep Dive into Transformers and Attention"
tags:
  - Machine Learning
---

I highly recommend to have a knowledge of machine learning models or atleast the basics

## The Core Idea: Transformers

Before transformers, the industry relied on RNNs and LSTMs.

![A Long short-term memory (LSTM) unit architecture.](/assets/images/lstm-architecture.png)

> The paper "Attention is all you need" says that in LSTMs, the fundamental in-efficiency wasn't just forgetting, but mathematical structure of the sequential link itself.

To compute the hidden state `h` at position `t`, we must first compute `h` at position `t-1`. This is linear dependency.

The part that allows Transformers to replace RNNs is, instead of processing word one-by-one, the model calculates a score of how much each word in a sentence should **attend** to every other word simultaneously.

In the paper, author defined 3 vectors for each word in a sequence :-

*   **Query(Q)** : What the current word is "asking" for,
    
*   **Key(K)** : The label that describes what a word contains,
    
*   **Value(V)** : The actual content of that word.
    

To measure how much a Query matches a specific Key, we use Dot Product.

$$Scores = QK^T$$

When we multiply these two vectors, say of length `d_k`, as this dimension grows the "spread" of scores becomes massive. What's the problem here?

When we pass these scores through a **Softmax** function, we get our percentages,

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$$

If the input `x` is very large number, the Softmax function becomes super "peaky". It will give almost all the probability (1) to a single word and 0 to everything else.

In calculus, when a function is pushed to exreme values(0 or 1), its slope becomes nearly zero, hence the model gets stuck and stops improving.

For this, we divide the dot product by

$$\sqrt{d_k}$$

we mathematically push the variance back down to 1, this squashes the scores back into a range where Softmax is still sensitive. Hence we get the final attention formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Remember this, we are going to use an optimized version of this formula.

## Architecture

Our architecture has 3 main stages:

### Embedding and Initial Normalization

Embedding is a vector that represents a word's meaning. If the dimension of our model we've chosen **d\_model** = 384 for example, every single token in our **X** word vocabulary is represented by 384 coordinates.

In the original "Attention is all you need" paper, there was Post-Layer Normalization.

$$x_{out} = \text{Norm}(x_{in} + \text{Sublayer}(x_{in}))$$

We can see, because of the Norm is on the outside, every path the signal takes, it is forced to pass through the derivative of the Norm function. If derivative of Norm is small, the whole signal dies.

In Pre-Layer Normalization,

$$x_{out} = x_{in} + \text{Sublayer}(\text{Norm}(x_{in}))$$

The Norm is on the inside, when the gradient travels backward -> it sees a plus sign. One of the piece goes through Norm function, while other stays at 100% strength, preventing gradient from vanishing.

*The mathematical "shortcut" was the key finding in the paper "On Layer Normalization in the Transformer Architecture". They proved that Pre-Layer Normalization prevents the gradient from vanishing*.

### The Decoder Block

In the original attention paper, it had both encoder and decoder blocks, but for text generation and what most LLMs use, we're going with a Decoder only block. Encoders are usually used to understand input and give us a more rich and context-aware representation.

Instead of **Multi-Head Attention**, we're gonna use **Grouped-Query Attention (GQA)**, we have 6 Query heads sharing just 2 Key/Value heads.

Let's say we have

$$neural\_heads=6,\ Q\ projections = 6$$

Seperate weight matrices,

$${W_1, W_2, W_3, W_4, W_5, W_6}$$

And same for Key and Value. Each head runs attention independently, and then we concat all the heads. This causes KV cache to become very large. Total projections = 6 + 6 + 6 = 18

In GQA we ask, do we actually need 6 independent K and V heads?

We know that Q head needs to be different, each head asks a different question, but Key and Value is just "whats available". In GQA, we group the heads, say if **n\_heads** = 6, then we form 2 groups, **n\_kv\_heads = 2**. Two groups, Two heads of Key and Value. Now total projections = 6 + 2 + 2 = 10 Much better in memory.

![Implementation of Rotary Position Embedding (RoPE).](https://cdn.hashnode.com/uploads/covers/67f29eb39a2e108cdae12adb/367fa97d-020f-4076-a914-a6ca964870b6.png)

**In Rotary Positional Embeddings**, say we have 2 words, *King* at position *m* = 1, and *Man* at position *n* = 2.

The RoPE paper suggests that instead of rotating the whole 384 dimensional vector as a single unit, we break it into a pair of coordinates. Now that we have the coordinates, the rotation formula for a pair **(x, y)** at position *m* is:

$$x' = x \cos(m \cdot \theta) - y \sin(m\cdot \theta)$$

$$y' = x \sin(m \cdot \theta) + y \cos(m \cdot \theta)$$

Lets say the *King*, its vector is rotated by 1 units of angle *$\theta$* and *Man* of 2 units of angle *$\theta$*.

Now when the model calculates attention, it takes the dot product of the Query(King) and the Key(Man), adjusted by their differences in angle.

$$\text{Score} = \text{Magnitude} \times \cos(\text{Angle}{\text{King}} - \text{Angle}{\text{Man}})$$

$$\text{Angle difference} = (2 \cdot \theta) - (1 \cdot \theta) = \mathbf{1 \cdot \theta}$$

This means the model "feels" that these 2 words are kind of identical, as they are exactly 1 step apart regardless of whichever position it is situated in.

One more thing is, instead of standard ReLU or GELU activations in FFN, we're using squared ReLU. In one of Google's primer paper, the research found that this sharpening of the activation allows the model to reach a lower error rate fast during training.

### The Exit head

Before hitting the final linear layer, the data goes through one last RMSNorm, then projects it into our **vocab\_size** vocabulary. but what really happens at the End of the Network?

After the decoder blocks, every token has a vector of 384 numbers, this vector now encodes deep contextual meaning, for example not just, ***what word am i***, but ***what word am i, given everything that came before me***. This is where logits come into play, A logit is just a raw score the model gives to each possible next token, the highest score wins.

The last token's vector is what we care about, we take that 384 dimensional vector and project it upto **vocab\_size** dimensions.

These vector of numbers are our logits, to turn this into actual probabilities, we apply softmax.

## Implementation

### Model Configuration

First let's see the config file of our GPT (I named it pocket transformer)

```python
class PocketConfig:
	vocab_size = 50257 # tiktoken
	seq_len = 256 # context window
	d_model = 384 # embedding dimension
	n_layers = 6 # no. stacked decoder blocks
	n_heads = 6 # Query heads
	n_kv_heads = 2 # kv heads
	dropout = 0.0 # regularization

	@property
	def d_k(self) -> int:
		# dimension per attention
		return  self.d_model  // self.n_heads  # 64
	
	@property
	def n_groups(self) -> int:
		# 3 query heads share each of 2 kv heads
		return  self.n_heads  //  self.n_kv_heads # 3

	@property 
	def ffn_hidden(self) -> int:
		# ffn hidden dimension, 4x d_model rounded to chunks
		multiple = 64
		return multiple * ((4 * self.d_model + multiple - 1) // multiple)
```

Why expand by ffn? Attention mixes information between tokens, the ffn processes each token independently after that mixing.

### Model specific concepts

We are going to use a 3D tensor, of shape

$$(batch,\ seq,\ d\_model)$$

and 4D tensor of shape

$$(batch,\ n\_heads,\ seq,\ d\_k)$$

*   **batch\_size** - how many sequences are processed simutaneously
    
*   **seq\_len** - how many tokens in each sequence
    
*   **d\_model** - features per token (384)
    
*   **d\_k** - features per attention head (384/6)
    

$$W_q,\ W_k,\ W_v\ and\ W_o$$

Above are learned weight matrices, query, key, value and output of dimension

$$(d\_model,\ d\_model)$$

### Normalization Layer

As discussed above, we're gonna use RMSNorm.

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

The Normalization is:

$$\bar{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})}$$

Hence the final RMSNorm:-

$$\text{RMSNorm}(\mathbf{x})i = \left( \frac{x_i}{\sqrt{\frac{1}{d} \sum{j=1}^{d} x_j^2 + \epsilon}} \right)$$

```python
class RMSNorm(nn.Module):
	def  __init__(self,  eps  =  1e-8):
		super().__init__()
		self.eps  =  eps

	def forward(self, tx: Tensor) -> Tensor:
		root_mean_square = tx.pow(2).mean(dim=-1,  keepdim=True).sqrt() + self.eps
		return  tx / root_mean_square
```

The dim=-1 means the last dimension, here it is d\_model. As we want to normalize each token "independently", normalizing across d\_model means each token's 384 features are scaled independently, token 3's normalization never affects token 7

### Rotary Positional Embeddings

We already covered what it is in the above section.

Now applying this to a vector of **d\_k** dimensions, we can't rotate all 64 at once as rotation is a 2D operation. In RoPE, we pair up dimensions and rotate each pair independently by its own angle.

When we look at the pattern

```plaintext
part 1 (cos terms):  [x1cos, x2cos, x3cos, x4cos, ...]
                   = x * cos

part 2 (sin terms):  [-x2sin, x1sin, -x4sin, x3sin, ...]
                   = rotate_half(x) * sin
```

Hence,

$$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\theta) + \text{rotate_half}(\mathbf{x}) \odot \sin(m\theta)$$

```python
class RoPE(nn.Module):
    def __init__(self, d_k, seq_len):
        super().__init__()
        theta = 1.0 / torch.pow(10000, torch.arange(0, d_k, 2).float() / d_k) # 1 frequencey per pair
        positions = torch.arange(seq_len).float()
        
        angles = torch.outer(positions, theta) # ex. (seq_len, d_k/2)
        angles = torch.cat([angles, angles], dim=-1) # duplicating for dimensions

        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def rotate_half(self, tx) -> Tensor:
        half = tx.shape[-1] // 2
        x1 = tx[..., :half]
        x2 = tx[..., half:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, tx: Tensor) -> Tensor:
        seq_len = tx.shape[2]
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        return tx * cos + self.rotate_half(tx) * sin
```

In the initialization, the angles is basically "how much to rotate the dimension **x** at position **y**, then we precompute the cosine and sine of every angle.

### **Grouped Query Attention**

```python
import math
class GQA(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.cfg = cfg 
        self.Wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.Wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope = RoPE(d_k=cfg.d_k, seq_len=cfg.seq_len)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor, mask = None) -> Tensor:
        batch, seq, _ = tx.shape 

        # first step: project
        Q: Tensor = self.Wq(tx) # (batch, seq, n_heads * d_k)
        K = self.Wk(tx)  # (batch, seq, n_kv_heads * d_k)
        V = self.Wv(tx)

        # second step: reshape into attention heads 
        Q = Q.view(batch, seq, self.cfg.n_heads, self.cfg.d_k).transpose(1, 2)  # (batch, 6, seq, 64)
        K = K.view(batch, seq, self.cfg.n_kv_heads, self.cfg.d_k).transpose(1, 2)
        V = V.view(batch, seq, self.cfg.n_kv_heads, self.cfg.d_k).transpose(1, 2)

        # apply rope 
        Q = self.rope(Q)  # (batch, 6, seq, 64)
        K: Tensor = self.rope(K) # (batch, 2, seq, 64)

        # expand K, V from n_kv_heads to n_heads
        K = K.repeat_interleave(self.cfg.n_groups, dim=1) # (batch, 6, seq, 64)
        V = V.repeat_interleave(self.cfg.n_groups, dim=1) # keeps groups together
        
        # now attentino scores 
        scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.cfg.d_k))

        # causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax functoin 
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted sum with value 
        output = torch.matmul(weights, V)

        # reshape back, should be contigous
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.cfg.d_model)

        # output projection 
        return self.Wo(output)
```

The most important thing here is the simple attention formula we discussed.

`scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.cfg.d_k))`

`transpose(-2, -1)` swaps the last 2 dimensions; ie **seq** and **d\_k**. We do this to match dimensions for matrix multiplication.

For example take

```json
Q:      (2, 6, 10, 64)   6 heads, each token is a 64-dim query vector
K.T:    (2, 6, 64, 10)   6 heads, transposed for matmul compatibility
```

Now matmul will result in the dimension **(2, 6, 10, 10).**

### Feed-Forward Network

It is just two linear layers with an activation in between them. It runs each token independently, kind of like no-communication between other tokens.

We expand the dimension first, giving the model more "space" for thinking, apply our activation functions, then compress back.

```python
import torch.nn.functional as F
class FFN(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__() 
        self.fc1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor) -> Tensor:
        x = self.fc1(tx) 
        x = F.relu(x).pow(2) 
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### Decoder Block

Now, we're gonna combine the RMSNorm, GQA and FFN with residualt connections.

```python
class DecoderBlock(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()
        self.norm1 = RMSNorm()
        self.gqa = GQA(cfg)
        self.norm2 = RMSNorm()
        self.ffn = FFN(cfg)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, tx: Tensor, mask = None) -> Tensor:
        temp = tx 
        tx = self.norm1(tx)
        tx = self.gqa(tx, mask)
        tx = temp + self.dropout(tx)

        # ffn layer 
        temp = tx
        tx = self.norm2(tx)
        tx = self.ffn(tx)
        tx = temp + self.dropout(tx)

        return tx
```

The + in for residual connections (temp variable) is exactly the same as the part where i explained Pre-Layer Normalization and Post-Layer Normalization. The gradient shortcut that keeps training stable across all layers.

### Wrapping it all up, PocketTransformer

The steps are pretty simple

1.  Embedding
    
2.  Normalization
    
3.  Causal Mask
    
4.  Loop through the blocks
    
5.  Normalization
    
6.  Projection to vocabulary and return logits
    

```python
class PocketTransformer(nn.Module):
    def __init__(self, cfg: PocketConfig):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm1 = RMSNorm()
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm2 = RMSNorm()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tx: Tensor) -> Tensor:
        # apply embedding
        x = self.embedding(tx)
        x = self.norm1(x) 

        seq = x.shape[1]
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
        # (1, 1, seq, seq) 

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm2(x)
        logits = self.head(x)

        return logits
```

`torch.tril` means triangle lower, it returns the lower triangular part of a matrix and zeros everywhere else.

```json
token 1: can see [1, 0, 0, 0, 0] - only itself
token 2: can see [1, 1, 0, 0, 0] - itself and token 1
token 3: can see [1, 1, 1, 0, 0] - itself and tokens 1,2
token 4: can see [1, 1, 1, 1, 0] - itself and tokens 1,2,3
token 5: can see [1, 1, 1, 1, 1] - everything
```

We need shape `(1, 1, seq, seq)` to broadcast against scores shape `(batch, heads, seq, seq)`

```json
scores: (2, 6, 10, 10) 
mask: (1, 1, 10, 10) = broadcasts across batch=2 and heads=6
```

We now have a model, untrained, weights completely random. If you asked it to generate text right now it would produce gibberish. Every weight is just noise.

But the architecture is complete. Every component we built, RMSNorm, RoPE, GQA, ReLU^2, the residual stream, is similar to modern LLMs like LLaMA. The only difference is zeros: more layers, wider dimensions, more parameters, more data.

In Part 2 we train it, and try to make something meaningful.

The github is [https://github.com/ish4n10/pocket-transformer](https://github.com/ish4n10/pocket-transformer)

## References

### Foundational Papers

**Attention Is All You Need (2017)** The original transformer paper. Everything we built is based on this. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**On Layer Normalization in the Transformer Architecture (2020)** The paper that proved pre-norm is better than post-norm. [https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)

* * *

### Modern Architecture Papers

**RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)** The RoPE paper. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

**GQA: Training Generalized Multi-Query Transformer Models (2023)** [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

**LLaMA: Open and Efficient Foundation Language Models (2023)** [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)

**LLaMA 2 (2023)** [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

**Root Mean Square Layer Normalization (2019)** [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)

**Primer: Searching for Efficient Transformers for Language Modeling (2021)** The paper that found ReLU^2 outperforms standard activations. [https://arxiv.org/abs/2109.08668](https://arxiv.org/abs/2109.08668)






