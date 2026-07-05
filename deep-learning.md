# Deep Learning Cheat Sheet

> Quick reference for writing pseudocode. Symbols, shapes, and update rules are stated explicitly so you can translate directly into code.

---

## Table of Contents

0. [Model Quick Reference](#model-quick-reference)
1. [Life Cycle of Deep Learning Models](#1-life-cycle-of-deep-learning-models)
   - [Evaluation & Convergence Metrics](#evaluation--convergence-metrics)
2. [Deep Learning Network Architectures](#2-deep-learning-network-architectures)
   - [MLP / Feedforward](#21-mlp--feedforward-neural-network)
   - [CNN](#22-convolutional-neural-network-cnn)
   - [Vision Transformer (ViT)](#23-vision-transformer-vit)
   - [RNN & GRU](#24-rnn--gru)
   - [LSTM](#25-lstm)
   - [Transformer](#26-transformer)
   - [GPT / BERT](#27-gpt--bert)
   - [VAE](#28-variational-autoencoder-vae)
   - [GAN](#29-generative-adversarial-network-gan)
   - [Diffusion Models](#210-diffusion-models)
3. [Dictionary](#3-dictionary) — [A–Z index](#activation-function)

---

## Model Quick Reference

> Pick an architecture by data modality, scale, and compute budget.

### Architecture Comparison

| Architecture | Param scale | Compute (inference) | Core operators | Best for |
|--------------|-------------|---------------------|----------------|----------|
| [MLP](#21-mlp--feedforward-neural-network) | $10^4$–$10^8$ | $O(n \cdot d \cdot h)$ | Linear, activation | Tabular, low-dim vectors |
| [CNN](#22-convolutional-neural-network-cnn) | $10^6$–$10^8$ | $O(HW \cdot k^2 \cdot C_{in} C_{out})$ | Conv, pool, BN | Images, spatial grids |
| [ViT](#23-vision-transformer-vit) | $10^7$–$10^9$ | $O(P^2 \cdot d)$ per block | Patch embed, MHA | Large image datasets |
| [RNN / GRU](#24-rnn--gru) | $10^5$–$10^7$ | $O(T \cdot h^2)$ sequential | Recurrent matmul, gates | Short sequences |
| [LSTM](#25-lstm) | $10^5$–$10^7$ | $O(T \cdot h^2)$ sequential | Gates, cell state | Medium sequences |
| [Transformer](#26-transformer) | $10^7$–$10^{12}$ | $O(T^2 \cdot d)$ per layer | Self-attention, FFN | Text, sequences, multimodal |
| [GPT / BERT](#27-gpt--bert) | $10^8$–$10^{12}$ | Same as Transformer | Causal / masked LM | NLP pretrain & finetune |
| [VAE](#28-variational-autoencoder-vae) | $10^6$–$10^8$ | Encode + decode pass | Reparameterize, KL | Generation, compression |
| [GAN](#29-generative-adversarial-network-gan) | $10^6$–$10^8$ | Generator forward | Conv/transpose conv | Image synthesis |
| [Diffusion](#210-diffusion-models) | $10^7$–$10^9$ | $O(T_{steps} \cdot \text{UNet})$ | Noise schedule, denoise | High-quality generation |

### By Data Modality

| Modality | Best choice | Alternatives |
|----------|-------------|--------------|
| **Tabular / vectors** | [MLP](#21-mlp--feedforward-neural-network) | [Transformer](#26-transformer) (overkill) |
| **Images (small data)** | [CNN](#22-convolutional-neural-network-cnn) | [ViT](#23-vision-transformer-vit) + heavy aug |
| **Images (large data)** | [ViT](#23-vision-transformer-vit), [CNN](#22-convolutional-neural-network-cnn) (ResNet) | [Diffusion](#210-diffusion-models) for gen |
| **Text / tokens** | [Transformer](#26-transformer), [GPT / BERT](#27-gpt--bert) | RNN family (legacy) |
| **Time series** | [LSTM](#25-lstm), [Transformer](#26-transformer) | [GRU](#24-rnn--gru) |
| **Image generation** | [Diffusion](#210-diffusion-models) | [GAN](#29-generative-adversarial-network-gan), [VAE](#28-variational-autoencoder-vae) |
| **Latent representation** | [VAE](#28-variational-autoencoder-vae) | [Autoencoder](#28-variational-autoencoder-vae) (deterministic) |

### Decision Flow

```text
START
├─ Input type?
│   ├─ Tabular vector → MLP
│   ├─ Image → CNN (default) → ViT (large data)
│   ├─ Sequence → LSTM/GRU (short) → Transformer (long / NLP)
│   └─ Generate?
│       ├─ Sharp images → GAN or Diffusion
│       ├─ Smooth latent → VAE
│       └─ Text → GPT / BERT / Diffusion (multimodal)
```

---

## 1. Life Cycle of Deep Learning Models

### Life Cycle

#### 1. Data Pipelines & Augmentation

High-dimensional data requires efficient loading, tensorization, and on-the-fly augmentation.

| Step | Pseudocode skeleton |
|------|---------------------|
| **Dataset** | `Dataset.__getitem__` → return tensor sample |
| **DataLoader** | `batch, shuffle, num_workers, pin_memory` |
| **Tensorization** | `torch.tensor`, `ToTensor`, normalize to $[0,1]$ or zero-mean |
| **Augmentation** | `RandomCrop, Flip, ColorJitter, Mixup, CutMix` |

```text
FUNCTION build_dataloader(paths, batch_size, train=True):
    dataset ← ImageDataset(paths, transform=train_transform IF train ELSE eval_transform)
    RETURN DataLoader(dataset, batch_size, shuffle=train, num_workers=4, pin_memory=True)

FUNCTION train_transform():
    RETURN Compose([
        RandomResizedCrop(224), RandomHorizontalFlip(),
        ColorJitter(0.2, 0.2, 0.2), ToTensor(), Normalize(mean, std)
    ])
```

| Augmentation | Use when |
|--------------|----------|
| RandomCrop / Flip | images; invariance to position |
| ColorJitter | lighting robustness |
| Mixup / CutMix | regularization; classification |
| Token masking | BERT-style pretraining |
| SpecAugment | speech / audio spectrograms |

---

#### 2. Optimization & Training Dynamics

| Topic | Key idea |
|-------|----------|
| **Vanishing gradient** | Deep nets: gradients $\to 0$ in early layers; use ReLU, skip connections, proper init |
| **Exploding gradient** | Gradients $\to \infty$; use <a href="#gradient-clipping">gradient clipping</a> |
| **Activation** | ReLU (default), GELU (Transformers), Sigmoid/Tanh (gates) |
| **Optimizer** | AdamW (default), SGD+momentum (CV fine-tune) |
| **LR Scheduler** | Cosine decay, warmup, ReduceLROnPlateau |

```text
optimizer ← AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler ← CosineAnnealingLR(optimizer, T_max=epochs)

FOR epoch IN 1..epochs:
    FOR batch IN train_loader:
        loss ← forward_backward(model, batch)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()
```

| Optimizer | When to use |
|-----------|-------------|
| SGD + momentum | CNN training from scratch; fine-tuning |
| Adam / AdamW | Transformers, default deep learning |
| AdamW | preferred over Adam (decoupled weight decay) |

| LR Schedule | Pattern |
|-------------|---------|
| Step decay | multiply lr by 0.1 every N epochs |
| Cosine | smooth decay to near zero |
| Warmup | linear increase for first W steps (Transformers) |
| OneCycle | rise then fall in one cycle |

---

#### 3. Regularization & Anti-Overfitting

| Technique | Mechanism |
|-----------|-----------|
| <a href="#dropout">Dropout</a> | Randomly zero neurons during training |
| <a href="#weight-decay">Weight decay</a> | L2 penalty on weights (in AdamW) |
| <a href="#batch-normalization">Batch Normalization</a> | Normalize activations per mini-batch (CNN) |
| <a href="#layer-normalization">Layer Normalization</a> | Normalize per sample (Transformers, RNN) |
| Early stopping | Halt when val loss stops improving |
| Data augmentation | Artificially expand training set |

```text
# Typical CNN block
x ← Conv2d(x)
x ← BatchNorm2d(x)
x ← ReLU(x)
x ← Dropout2d(p=0.2)(x)

# Typical Transformer block uses LayerNorm inside sublayers
```

---

#### 4. Evaluation & Convergence Metrics

```text
FUNCTION train_and_monitor(model, train_loader, val_loader, epochs):
    FOR epoch IN 1..epochs:
        train_loss ← train_one_epoch(model, train_loader)
        val_loss, val_acc ← evaluate(model, val_loader)
        log(train_loss, val_loss, gap=val_loss - train_loss)
        IF val_loss not improving for patience epochs: BREAK
```

##### Evaluation & Convergence Metrics

Monitor **training loss** vs. **validation loss** to diagnose fit:

| Pattern | Diagnosis | Remedy |
|---------|-----------|--------|
| Both high | <a href="#underfitting">Underfitting</a> | Bigger model, train longer, lower regularization |
| Train low, val high | <a href="#overfitting">Overfitting</a> | Dropout, weight decay, more data, augmentation |
| Both decreasing | Healthy training | Continue |
| Val loss increases | Overfitting begins | Early stopping, checkpoint best val |

**Common metrics**

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1, top-1 / top-5 error |
| Regression | MSE, MAE, $R^2$ |
| Generation | FID, IS (images), perplexity (language) |
| Segmentation | IoU, Dice coefficient |

**Interpretability**

| Tool | Purpose |
|------|---------|
| Feature maps | visualize CNN filters |
| Attention weights | see what Transformer focuses on |
| Grad-CAM | saliency heatmaps for classification |
| t-SNE / UMAP on embeddings | cluster structure |

---

## 2. Deep Learning Network Architectures

### 2.1 MLP / Feedforward Neural Network

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Tabular data; fixed-size vectors; baseline for any structured input |
| **Cons** | No spatial or sequential inductive bias; parameter count grows with input dim |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $X$ | input batch | (n, $d_{in}$) |
| $W^l$ | weight matrix layer $l$ | ($d_{l-1}$, $d_l$) |
| $b^l$ | bias | ($d_l$) |
| $a^l$ | activation | (n, $d_l$) |
| $L$ | number of layers | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Forward** ($a^0 = X$)  
   $z^l = a^{l-1} W^l + b^l$  
   $a^l = g(z^l)$

2. **Loss** (classification)  
   $\mathcal{L} = -\dfrac{1}{n}\sum_i \sum_k y_{ik}\log\hat{y}_{ik}$

3. **Loss** (regression)  
   $\mathcal{L} = \dfrac{1}{n}\sum_i \|y_i - \hat{y}_i\|^2$

4. **Objective**  
   $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$

</details>

<details>
<summary><strong>d. Update Rules</strong> · <a href="#backpropagation">backpropagation</a></summary>

1. Forward pass: compute $z^l, a^l$ for all layers; evaluate $\mathcal{L}$

2. Backward pass: $\delta^L = \dfrac{\partial \mathcal{L}}{\partial a^L} \odot g'(z^L)$

3. For $l = L-1, \ldots, 1$:  
   $\dfrac{\partial \mathcal{L}}{\partial W^l} = (a^{l-1})^\top \delta^l$  
   $\delta^{l-1} = \delta^l (W^l)^\top \odot g'(z^{l-1})$

4. Update: $W^l \leftarrow W^l - \eta \dfrac{\partial \mathcal{L}}{\partial W^l}$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| hidden dims | 64 – 512 | per layer |
| depth | 2 – 5 layers | |
| learning_rate | 1e-4 – 1e-2 | AdamW |
| dropout | 0.1 – 0.5 | |
| weight_decay | 1e-4 – 1e-2 | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Task loss, accuracy / MSE, train-val gap. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← Sequential([
    Linear(d_in, 256), ReLU(), Dropout(0.3),
    Linear(256, 128), ReLU(),
    Linear(128, n_classes)
])
optimizer ← AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

FOR batch IN train_loader:
    X, y ← batch
    logits ← model(X)
    loss ← cross_entropy(logits, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.2 Convolutional Neural Network (CNN)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Images, spatial data; translation-equivariant features |
| **Cons** | Fixed resolution; limited long-range context without deep stacks / attention |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $X$ | input image batch | (n, C, H, W) |
| $K$ | kernel / filter | (C_out, C_in, k, k) |
| $Y$ | conv output | (n, C_out, H', W') |
| stride, padding | spatial params | scalar |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **2D Convolution**  
   $Y_{c,i,j} = \sum_{c',u,v} K_{c,c',u,v} \cdot X_{c', i+u, j+v} + b_c$

2. **Output size**  
   $H' = \lfloor (H - k + 2p) / s \rfloor + 1$

3. **Pooling** (max)  
   $Y_{i,j} = \max_{(u,v) \in \text{window}} X_{i+u,j+v}$

4. **ResNet skip**  
   $y = \mathcal{F}(x) + x$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Forward: stack Conv → BN → ReLU → (optional Pool) blocks

2. Final: GlobalAvgPool → Flatten → Linear classifier

3. Backprop through conv (cross-correlation gradient), BN, and skip connections

4. Update all kernels and biases via SGD / AdamW

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| kernel size | 3, 5, 7 | 3 most common |
| channels | 32 → 512 | double per block |
| depth | ResNet-18 to ResNet-152 | |
| learning_rate | 1e-3 – 1e-1 | SGD + cosine for scratch |
| batch_size | 32 – 256 | GPU memory bound |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Top-1 / top-5 accuracy, train-val gap, FLOPs, inference latency. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← ResNet50(num_classes=1000, pretrained=True)
optimizer ← SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

FOR epoch IN 1..epochs:
    FOR images, labels IN train_loader:
        logits ← model(images)
        loss ← cross_entropy(logits, labels)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.3 Vision Transformer (ViT)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Large image datasets; need global context; have pretrain compute |
| **Cons** | Data-hungry; quadratic cost in number of patches; less efficient than CNN on small data |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $P$ | number of patches | $HW / p^2$ |
| $p$ | patch size | e.g. 16 |
| $d$ | embedding dim | e.g. 768 |
| $X_p$ | patch embeddings | (n, P, d) |
| CLS | class token | (n, 1, d) |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Patchify** — split image into $P$ patches, linear embed to $d$ dims

2. **Add** positional embedding + CLS token

3. **Transformer encoder** — repeat L times:  
   $\text{MHA}(\text{LN}(x)) + x$; $\text{FFN}(\text{LN}(x)) + x$

4. **Classify** from CLS token output

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Patch embed + position embed (learned)

2. Pass through $L$ Transformer encoder blocks

3. Take CLS token → linear head → logits

4. Backprop end-to-end; often pretrained then fine-tuned

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| patch size | 8, 16, 32 | smaller = more tokens |
| depth / heads | ViT-B/16, ViT-L/16 | |
| learning_rate | 1e-5 – 1e-3 | lower when fine-tuning |
| warmup steps | 5% – 10% of total | |
| weight_decay | 0.05 – 0.3 | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Top-1 accuracy, pretrain vs. finetune loss, attention map quality. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← ViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000)
optimizer ← AdamW(model.parameters(), lr=1e-3, weight_decay=0.3)

FOR images, labels IN train_loader:
    logits ← model(images)
    loss ← cross_entropy(logits, labels)
    loss.backward(); clip_grad_norm_(model, 1.0); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.4 RNN & GRU

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Short sequences; low-latency sequential models; legacy baselines |
| **Cons** | Vanishing gradients on long sequences; slow sequential inference |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $x_t$ | input at step $t$ | (n, $d_{in}$) |
| $h_t$ | hidden state | (n, $d_h$) |
| $T$ | sequence length | scalar |
| $W, U, V$ | input, recurrent, output weights | — |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Vanilla RNN**  
   $h_t = \tanh(W x_t + U h_{t-1} + b)$  
   $y_t = V h_t$

2. **GRU gates**  
   $z_t = \sigma(W_z x_t + U_z h_{t-1})$ (update)  
   $r_t = \sigma(W_r x_t + U_r h_{t-1})$ (reset)  
   $\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$  
   $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize $h_0 = 0$

2. **For** $t = 1, \ldots, T$: compute $h_t$ from $x_t, h_{t-1}$ (unroll graph)

3. Compute loss on outputs (all steps or last step)

4. **BPTT** — backprop through time; truncate for long sequences

5. Update weights; optionally clip gradients

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| hidden_size | 64 – 512 | |
| num_layers | 1 – 3 | stacked RNN |
| learning_rate | 1e-3 – 1e-2 | |
| bptt_truncation | 20 – 100 steps | |
| dropout | 0.2 – 0.5 | between layers |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Sequence loss (CE / MSE), perplexity (language), train-val gap. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← GRU(input_size=d_in, hidden_size=256, num_layers=2, batch_first=True)
classifier ← Linear(256, n_classes)

FOR sequences, labels IN train_loader:
    output, h_n ← model(sequences)          # output: (n, T, 256)
    logits ← classifier(output[:, -1, :])   # last step
    loss ← cross_entropy(logits, labels)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.5 LSTM

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Medium-length sequences; need memory cell for longer dependencies |
| **Cons** | Slower than GRU; still sequential; largely superseded by Transformers for NLP |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $c_t$ | cell state |
| $h_t$ | hidden state |
| $f_t$ | forget gate |
| $i_t$ | input gate |
| $o_t$ | output gate |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Forget gate** — $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$

2. **Input gate** — $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$

3. **Candidate cell** — $\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$

4. **Cell update** — $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

5. **Output gate** — $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$; $h_t = o_t \odot \tanh(c_t)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Initialize $h_0, c_0 = 0$

2. **For** each timestep: compute gates, update $c_t, h_t$

3. Loss on $h_t$ or sequence outputs

4. BPTT with gradient clipping

5. Update all gate weights jointly

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| hidden_size | 128 – 512 | |
| num_layers | 1 – 3 | |
| learning_rate | 1e-3 – 1e-2 | |
| dropout | 0.2 – 0.5 | |
| grad_clip | 1.0 – 5.0 | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Sequence CE loss, perplexity, BLEU (if seq2seq). See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← LSTM(input_size=128, hidden_size=256, num_layers=2, dropout=0.3, batch_first=True)
head ← Linear(256, vocab_size)

FOR x, y IN train_loader:
    out, (h, c) ← model(x)
    logits ← head(out)                      # teacher forcing
    loss ← cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward(); clip_grad_norm_(model, 1.0); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.6 Transformer

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | NLP, long sequences, multimodal; parallelizable training |
| **Cons** | $O(T^2)$ memory in sequence length; needs large data / pretraining |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $X$ | input embeddings | (n, T, d) |
| $Q, K, V$ | query, key, value | (n, T, d) |
| $h$ | number of heads | scalar |
| $d_k$ | dim per head | $d / h$ |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Scaled dot-product attention**  
   $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$

2. **Multi-head** — concat $h$ heads, project:  
   $\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$

3. **FFN** — $\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$

4. **Encoder block** — $\text{LN}(x + \text{MHA}(x))$; $\text{LN}(x + \text{FFN}(x))$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Embed tokens + positional encoding

2. Pass through $L$ encoder (and/or decoder) layers

3. Decoder uses masked self-attention + cross-attention to encoder

4. Output projection to vocab logits

5. Backprop; AdamW + warmup schedule standard

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| d_model | 512 – 4096 | |
| num_heads | 8 – 32 | $d_{model} / h = d_k$ |
| num_layers | 6 – 96 | |
| ffn_dim | $4 \times d_{model}$ | |
| dropout | 0.1 – 0.3 | |
| learning_rate | 1e-4 – 6e-4 | with warmup |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Perplexity, BLEU / ROUGE, training loss, attention entropy. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
optimizer ← AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98))

FOR src, tgt IN train_loader:
    logits ← model(src, tgt[:, :-1])        # teacher forcing
    loss ← cross_entropy(logits, tgt[:, 1:])
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.7 GPT / BERT

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | NLP pretrain + finetune; text generation (GPT) or understanding (BERT) |
| **Cons** | Massive compute for pretraining; context length limits |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| GPT | decoder-only Transformer; causal mask |
| BERT | encoder-only; bidirectional via masked LM |
| $V$ | vocabulary size |
| token_id | discrete input index |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **GPT (causal LM)** — predict $x_{t+1}$ from $x_{\le t}$; mask future tokens in attention

2. **Loss** — $\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t})$

3. **BERT (masked LM)** — randomly mask 15% tokens; predict from context

4. **Finetune** — add task head (classification, QA span) on top of pretrained encoder

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

**Pretrain:**

1. Tokenize corpus → token IDs

2. GPT: next-token prediction with causal mask

3. BERT: masked token prediction + optional NSP

4. AdamW + warmup + large batch (gradient accumulation)

**Finetune:**

1. Load pretrained weights

2. Add task-specific head; train on labeled data with lower lr

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Pretrain | Finetune |
|----------------|----------|----------|
| learning_rate | 1e-4 – 6e-4 | 1e-5 – 5e-5 |
| batch_size (tokens) | 0.5M – 4M | 16 – 128 seqs |
| warmup | 4k – 10k steps | optional |
| max_seq_len | 512 – 8192+ | task-dependent |
| weight_decay | 0.01 – 0.1 | 0.01 |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Perplexity (LM), GLUE / SuperGLUE (BERT), human eval (GPT generation). See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
# Finetune BERT for classification
tokenizer ← BertTokenizer.from_pretrained("bert-base-uncased")
model ← BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer ← AdamW(model.parameters(), lr=2e-5)

FOR batch IN train_loader:
    outputs ← model(**batch)
    loss ← outputs.loss
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# GPT generation
generated ← model.generate(input_ids, max_new_tokens=100, temperature=0.8)
```

</details>

---

### 2.8 Variational Autoencoder (VAE)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Learn smooth latent space; generate / interpolate; anomaly detection |
| **Cons** | Blurry samples vs. GAN; Gaussian latent assumption |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $z$ | latent variable | (n, $d_z$) |
| $\mu, \sigma$ | encoder outputs | (n, $d_z$) |
| $p(z)$ | prior | $\mathcal{N}(0, I)$ |
| $q(z|x)$ | encoder distribution | — |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Encoder** — $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$

2. **Decoder** — $p_\theta(x|z)$ (Bernoulli or Gaussian likelihood)

3. **ELBO**  
   $\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$

4. **Reparameterization** — $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. Encode $x$ → $\mu, \log\sigma^2$

2. Sample $z$ via reparameterization

3. Decode $\hat{x} = g_\theta(z)$

4. Loss = reconstruction + KL divergence

5. Backprop through encoder and decoder jointly

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| latent_dim | 16 – 256 | |
| kl_weight ($\beta$) | 0.1 – 10 | $\beta$-VAE |
| learning_rate | 1e-4 – 1e-3 | |
| encoder/decoder | CNN (images) / MLP | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

Reconstruction loss, KL term, FID (if generating), <a href="#latent-space">latent</a> interpolation quality. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
encoder ← Encoder(x) → mu, log_var
z ← mu + exp(0.5 * log_var) * epsilon     # reparameterize
x_hat ← Decoder(z)
recon_loss ← MSE(x, x_hat)
kl_loss ← -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
loss ← recon_loss + beta * kl_loss
loss.backward(); optimizer.step(); optimizer.zero_grad()
```

</details>

---

### 2.9 Generative Adversarial Network (GAN)

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | Sharp image synthesis; style transfer; data augmentation |
| **Cons** | Training instability; mode collapse; hard to evaluate |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $G$ | generator $z \to x$ |
| $D$ | discriminator $x \to [0,1]$ |
| $z$ | noise vector | (n, $d_z$) |
| $x_{real}$ | real samples |
| $x_{fake}$ | $G(z)$ |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Minimax objective**  
   $\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$

2. **Generator loss** (non-saturating)  
   $\mathcal{L}_G = -\mathbb{E}_z[\log D(G(z))]$

3. **Discriminator loss**  
   $\mathcal{L}_D = -\mathbb{E}_x[\log D(x)] - \mathbb{E}_z[\log(1 - D(G(z)))]$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

1. **Train D**: maximize ability to distinguish real vs. fake

2. **Train G**: maximize $D(G(z))$ (fool discriminator)

3. Alternate (or ratio) D and G updates

4. Use techniques: spectral norm, label smoothing, progressive growing (StyleGAN)

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| latent_dim | 64 – 512 | |
| lr_G, lr_D | 1e-4 – 2e-4 | often Adam |
| batch_size | 16 – 64 | |
| D/G update ratio | 1:1 or 1:5 | |
| architecture | DCGAN, StyleGAN | |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

FID, IS, visual quality, discriminator accuracy (balance ~0.5). See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
G ← Generator(latent_dim=128)
D ← Discriminator()
opt_G ← Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D ← Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

FOR real_images IN train_loader:
    z ← randn(batch_size, 128)
    fake ← G(z)
    loss_D ← -mean(log(D(real_images)) + log(1 - D(fake.detach())))
    update(D, loss_D, opt_D)
    loss_G ← -mean(log(D(fake)))
    update(G, loss_G, opt_G)
```

</details>

---

### 2.10 Diffusion Models

<details>
<summary><strong>a. Scenarios & Cons</strong></summary>

| | |
|---|---|
| **Use when** | State-of-the-art image/audio generation; controllable synthesis |
| **Cons** | Slow sampling (many steps); heavy compute for training |

</details>

<details>
<summary><strong>b. Notations & Dimensions</strong></summary>

| Symbol | Meaning |
|--------|---------|
| $x_0$ | clean data |
| $x_t$ | noisy data at step $t$ |
| $\beta_t$ | noise schedule |
| $\epsilon_\theta$ | noise prediction network (UNet) |
| $T$ | total diffusion steps |

</details>

<details>
<summary><strong>c. Math Core</strong></summary>

1. **Forward process** (fixed)  
   $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$

2. **Closed form**  
   $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$

3. **Training objective**  
   $\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\bigl[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\bigr]$

4. **Reverse sampling** — iteratively denoise $x_T \to x_0$

</details>

<details>
<summary><strong>d. Update Rules</strong></summary>

**Training:**

1. Sample $x_0$, random $t$, noise $\epsilon$

2. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

3. Predict $\hat{\epsilon} = \epsilon_\theta(x_t, t)$

4. Minimize $\|\epsilon - \hat{\epsilon}\|^2$

**Sampling:**

1. Start $x_T \sim \mathcal{N}(0,I)$

2. **For** $t = T, \ldots, 1$: denoise using learned $\epsilon_\theta$

3. Output $x_0$

</details>

<details>
<summary><strong>e. Hyperparameters & Tuning</strong></summary>

| <a href="#hyperparameter">Hyperparameter</a> | Typical range | Notes |
|----------------|---------------|-------|
| T (steps) | 100 – 1000 | DDPM; fewer with DDIM |
| noise schedule | linear, cosine | |
| learning_rate | 1e-4 – 2e-4 | |
| UNet channels | 128 – 512 | |
| guidance scale | 1 – 15 | classifier-free guidance |

</details>

<details>
<summary><strong>f. Metrics</strong></summary>

FID, CLIP score, sample quality, steps-to-quality tradeoff. See <a href="#evaluation--convergence-metrics">Evaluation & Convergence Metrics</a>.

</details>

<details>
<summary><strong>g. Example</strong></summary>

```text
model ← UNet()                            # noise predictor
FOR x_0 IN train_loader:
    t ← randint(0, T, batch_size)
    epsilon ← randn_like(x_0)
    x_t ← sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon
    epsilon_hat ← model(x_t, t)
    loss ← MSE(epsilon, epsilon_hat)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

# Sampling
x ← randn(1, C, H, W)
FOR t IN reversed(1..T):
    x ← denoise_step(x, t, model)
RETURN x
```

</details>

---

## 3. Dictionary

Click any term below, or follow links throughout the cheat sheet.

**A–C:** [Activation Function](#activation-function) · [AdamW](#adamw) · [Attention](#attention) · [Backpropagation](#backpropagation) · [Batch Normalization](#batch-normalization) · [Convolution](#convolution) · [Cross-Entropy Loss](#cross-entropy-loss)

**D–G:** [Dropout](#dropout) · [Embedding](#embedding) · [Encoder-Decoder](#encoder-decoder) · [Exploding Gradient](#exploding-gradient) · [FID](#fid) · [GAN](#gan) · [GELU](#gelu) · [Gradient Clipping](#gradient-clipping)

**L–P:** [Latent Space](#latent-space) · [Layer Normalization](#layer-normalization) · [Learning Rate Scheduler](#learning-rate-scheduler) · [Loss Function](#loss-function) · [Multi-Head Attention](#multi-head-attention) · [Perplexity](#perplexity) · [Pooling](#pooling)

**R–Z:** [ReLU](#relu) · [Residual Connection](#residual-connection) · [Self-Attention](#self-attention) · [Softmax](#softmax) · [Tokenization](#tokenization) · [Transfer Learning](#transfer-learning) · [Vanishing Gradient](#vanishing-gradient) · [Weight Decay](#weight-decay)

---

<a id="activation-function"></a>
### Activation Function

Nonlinearity $g(z)$ applied after linear layers. ReLU for CNNs/MLPs; GELU for Transformers; Sigmoid/Tanh for gates.

<a id="adamw"></a>
### AdamW

Adam optimizer with decoupled <a href="#weight-decay">weight decay</a>. Default for Transformers and many modern architectures.

<a id="attention"></a>
### Attention

Mechanism to weight inputs by relevance. <a href="#self-attention">Self-attention</a> relates positions within one sequence.

<a id="backpropagation"></a>
### Backpropagation

Chain rule applied to compute gradients through the computational graph. Core training algorithm for all neural networks.

<a id="batch-normalization"></a>
### Batch Normalization

Normalize activations across the mini-batch: $\hat{x} = (x - \mu_B)/\sigma_B$. Stabilizes CNN training.

<a id="convolution"></a>
### Convolution

Sliding filter operation extracting local spatial features. Core operator in <a href="#22-convolutional-neural-network-cnn">CNNs</a>.

<a id="cross-entropy-loss"></a>
### Cross-Entropy Loss

Classification loss: $-\sum_k y_k \log \hat{y}_k$. Standard for multi-class problems.

<a id="dropout"></a>
### Dropout

Randomly zero neurons with probability $p$ during training to prevent co-adaptation and <a href="#overfitting">overfitting</a>.

<a id="embedding"></a>
### Embedding

Learned lookup table mapping discrete tokens to dense vectors. Input to NLP and recommendation models.

<a id="encoder-decoder"></a>
### Encoder-Decoder

Architecture where encoder processes input and decoder generates output. Used in <a href="#26-transformer">Transformers</a> for seq2seq.

<a id="exploding-gradient"></a>
### Exploding Gradient

Gradients grow exponentially through layers. Fix with <a href="#gradient-clipping">gradient clipping</a>.

<a id="fid"></a>
### FID

**F**réchet **I**nception **D**istance. Measures quality of generated images vs. real distribution. Lower is better.

<a id="gan"></a>
### GAN

<a href="#29-generative-adversarial-network-gan">Generative Adversarial Network</a>. Generator vs. discriminator adversarial training.

<a id="gelu"></a>
### GELU

**G**aussian **E**rror **L**inear **U**nit. Smooth activation used in BERT, GPT, ViT.

<a id="gradient-clipping"></a>
### Gradient Clipping

Cap gradient norm: if $\|g\| > \tau$, scale $g \leftarrow g \cdot \tau / \|g\|$. Prevents <a href="#exploding-gradient">exploding gradients</a>.

<a id="hyperparameter"></a>
### Hyperparameter

Config set before training — learning rate, batch size, depth, dropout rate.

<a id="latent-space"></a>
### Latent Space

Lower-dimensional representation $z$ encoding data structure. Used in <a href="#28-variational-autoencoder-vae">VAEs</a>, GANs, diffusion models.

<a id="layer-normalization"></a>
### Layer Normalization

Normalize across features per sample (not batch). Standard in <a href="#26-transformer">Transformers</a> and RNNs.

<a id="learning-rate-scheduler"></a>
### Learning Rate Scheduler

Adjusts learning rate during training — cosine decay, step decay, warmup.

<a id="loss-function"></a>
### Loss Function

Scalar objective minimized during training — cross-entropy, MSE, adversarial loss, diffusion noise loss.

<a id="multi-head-attention"></a>
### Multi-Head Attention

Run $h$ parallel attention heads; concat and project. Captures diverse relationship types.

<a id="overfitting"></a>
### Overfitting

Model memorizes training data; low train loss but high val loss. Combat with <a href="#dropout">dropout</a>, augmentation, weight decay.

<a id="perplexity"></a>
### Perplexity

$\exp(\text{average cross-entropy})$ for language models. Lower = better next-token prediction.

<a id="pooling"></a>
### Pooling

Downsample spatial dimensions. Max pooling takes regional maximum; average pooling takes mean.

<a id="relu"></a>
### ReLU

$\max(0, z)$. Default activation for CNNs and MLPs. Avoids vanishing gradient for positive region.

<a id="residual-connection"></a>
### Residual Connection

Skip connection: $y = \mathcal{F}(x) + x$. Enables very deep networks (ResNet).

<a id="self-attention"></a>
### Self-Attention

Attention where queries, keys, values all come from the same sequence. Core of <a href="#26-transformer">Transformers</a>.

<a id="softmax"></a>
### Softmax

$\text{softmax}(z_i) = e^{z_i}/\sum_j e^{z_j}$. Converts logits to probability distribution.

<a id="tokenization"></a>
### Tokenization

Split text into subword tokens (BPE, WordPiece) mapped to integer IDs for model input.

<a id="transfer-learning"></a>
### Transfer Learning

Pretrain on large dataset; fine-tune on downstream task with smaller labeled data.

<a id="underfitting"></a>
### Underfitting

Model too simple; high loss on both train and val. Increase capacity or train longer.

<a id="vanishing-gradient"></a>
### Vanishing Gradient

Gradients shrink toward zero in early layers. Mitigate with ReLU, residuals, proper init, LayerNorm.

<a id="weight-decay"></a>
### Weight Decay

L2 regularization on weights, typically applied via AdamW. Reduces <a href="#overfitting">overfitting</a>.

---

## Quick Pseudocode Template (End-to-End)

```text
# 1. Data pipeline
train_loader ← build_dataloader(train_paths, batch_size=64, train=True)
val_loader ← build_dataloader(val_paths, batch_size=64, train=False)

# 2. Model & optimization
model ← Architecture(**config)
optimizer ← AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler ← CosineAnnealingLR(optimizer, T_max=epochs)

# 3. Train
FOR epoch IN 1..epochs:
    FOR batch IN train_loader:
        loss ← forward_backward(model, batch)
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()
    val_loss ← evaluate(model, val_loader)
    save_checkpoint IF val_loss improved

# 4. Deploy
save(model.state_dict(), "model.pt")
serve(model, inference_transform)
```
