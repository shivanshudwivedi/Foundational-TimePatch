# Project Plan: A Foundational Transformer for Asteroid Orbit Forecasting

## 1. Project Overview

This project aims to develop a **single, foundational time-series model** capable of forecasting the future trajectories of *multiple* different asteroids.

The core hypothesis is that a modern Transformer-based architecture, trained on a diverse dataset of asteroid trajectories (Ceres, Pallas, Vesta, etc.), can learn the *generalizable, underlying laws of orbital mechanics* (i.e., Kepler's laws, gravitational perturbations) purely from observation.

This model will be a sequence-to-sequence (Seq2Seq) forecaster:
* **Input:** A history of $N$ timesteps of an asteroid's 6D state vector (e.g., 24 hours of $x, y, z, v_x, v_y, v_z$).
* **Output:** A prediction of the *next* $M$ timesteps of that asteroid's 6D state vector (e.g., the next 6 hours).

This approach intentionally **excludes exogenous features** (like the positions of the Sun and Jupiter). We are testing the model's ability to build an internal *physical model* of the solar system, rather than simply correlating the asteroid's motion with its perturbers.

## 2. Core Goals

1.  **Beat the Baseline:** The primary success metric is to significantly outperform the two-stage FNN baseline model in terms of **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.
2.  **Create a Foundational Model:** We will build *one model*, not one model per asteroid. This single model must successfully forecast trajectories for all asteroids in the training set.
3.  **Test Generalizability:** The model should (in theory) be able to make reasonable predictions for an asteroid it has *not* seen, or at least be quickly fine-tuned for it (this is a stretch goal).
4.  **Implicit Physics Modeling:** Demonstrate that a Transformer can learn the complex, periodic, and non-linear patterns of orbital mechanics without being explicitly told about them.

## 3. Survey of Candidate Architectures

Standard Transformers (like in NLP) perform poorly on time-series. The field has evolved specialized architectures. Here are our top candidates:

| Architecture | Key Idea & How It Applies to Our Problem |
| :--- | :--- |
| **PatchTST (Recommended)** | **Idea:** Treats the time-series like an image. It "patches" the input (e.g., 24 hours) into smaller chunks (e.g., 4 patches of 6 hours). It runs attention over these *patches*, not individual hours. <br><br> **Why for us?** This is the current state-of-the-art on many benchmarks. It's efficient and allows the model to see both local (within a patch) and global (between patches) patterns. This is our **strongest candidate.** |
| **Autoformer** | **Idea:** Replaces standard attention with an "Auto-Correlation" mechanism. It's designed to automatically find the "seasonality" (i.e., orbital period) and decompose the series. <br><br> **Why for us?** Orbits are the definition of "seasonal" or periodic data. This is a very strong contender, but PatchTST is often simpler to implement and trains faster. |
| **DLinear / NLinear** | **Idea:** A shocking discovery in recent research. Extremely simple **Linear models** (no Transformer, no attention) often *outperform* complex Transformers on many forecasting tasks. <br><br> **Why for us?** This is our "canary in the coal mine." We should build this simple model first. If our big, complex Transformer cannot beat a simple Linear model, our Transformer design is wrong. |
| **Vanilla Transformer** | **Idea:** The original encoder-decoder model. <br><br> **Why for us?** We will *not* use this. It has quadratic complexity and no awareness of time, making it a poor choice for raw time-series. We will use the more advanced PatchTST. |

**Decision:** We will implement an architecture based on **PatchTST**. To ensure our work is robust, we will also build a simple **DLinear** model as a second, stronger baseline. Our goal is: `PatchTST > DLinear > FNN Baseline`.

## 4. Proposed Model: Detailed Implementation Plan

Here is the step-by-step plan to build our model for the best possible results.

### 4.1 The Core Idea: Patching

We will not feed the model 24 individual 1-hour timesteps. This is inefficient.
* **Input Data:** `(Batch, 24, 6)` (24 hours, 6 features)
* **1. Patching:** We will reshape this input into patches. Let's use a `patch_length` of 4.
    * Input becomes: `(Batch, 6, 4, 6)` (6 patches, 4 hours/patch, 6 features)
* **2. Flattening:** We flatten each patch to create our sequence of "tokens."
    * Input becomes: `(Batch, 6, 24)` (6 patches, $4 \times 6 = 24$ features per patch)
* **3. Embedding:** A Linear layer (a "patch embedder") projects this into the model's dimension (`D_MODEL`).
    * Input becomes: `(Batch, 6, D_MODEL)` (e.g., `(128, 6, 128)`)

This `(128, 6, 128)` tensor is what we feed to the Transformer Encoder. We are running attention over 6 "patch" tokens, not 24 "hour" tokens.

### 4.2 The "Foundational" Piece: Asteroid ID Embeddings

How does one model handle both Ceres (big orbit) and Eros (smaller orbit)? By using **embeddings**.
1.  **Modify `pre-process.py`:** When building `train.npz`, we must add a new array, `X_asteroid_ids`. This will be an integer array of shape `(N_samples,)` where `0` = Ceres, `1` = Pallas, etc.
2.  **In the Model:** We will create a PyTorch `nn.Embedding` layer: `self.asteroid_embed = nn.Embedding(num_asteroids, D_MODEL)`.
3.  **During `forward()`:**
    * We get our patch embeddings: `x = self.patch_embedder(patches)` -> `(128, 6, 128)`
    * We get our asteroid ID embeddings: `ast_id_emb = self.asteroid_embed(asteroid_ids)` -> `(128, 1, 128)` (we unsqueeze it)
    * We **add** this embedding to every patch token: `x = x + ast_id_emb`
    * We also add positional encoding: `x = x + self.pos_encoding`

This tells the model: "You are about to process a sequence of 6 patches. By the way, this *entire sequence* belongs to Asteroid #1." The model can then use its learned `Embedding[1]` vector to adjust its predictions.

### 4.3 Key Detail: SOTA Normalization (RevIN)

This is the most important trick for high performance.
* **Problem:** Our FNN baseline used `StandardScaler` on the *whole dataset*. This "leaks" information about the test set's mean/std into the training.
* **Better Way (InstanceNorm):** Normalize *each 24-hour sample* independently. `sample = (sample - sample.mean()) / sample.std()`. This is good.
* **Best Way (RevIN - Reversible Instance Normalization):**
    1.  **Forward:** Before feeding a sample into the model, calculate its `mean` and `std`. Store them. Normalize the sample `x_norm = (x - mean) / std`.
    2.  The model *only* sees `x_norm` and predicts `y_pred_norm`.
    3.  **Backward (Reversible):** De-normalize the prediction using the *input's* statistics: `y_pred = (y_pred_norm * std) + mean`.
    4.  **Loss:** Calculate MSELoss on the *de-normalized* predictions: `loss(y_pred, y_true)`.

This is a **crucial** implementation detail and a major part of why PatchTST works so well.

### 4.4 The Final Architecture (Walkthrough)

```python
class AsteroidTransformer(nn.Module):
    def __init__(self, num_asteroids, input_steps, patch_len, ...):
        super().__init__()
        # 1. Config
        self.patch_len = patch_len
        self.n_patches = input_steps // patch_len
        self.num_features = 6
        self.d_model = D_MODEL

        # 2. Reversible Normalization (RevIN)
        # This is just a simple layer to store mean/std
        self.revin = RevIN(num_features=self.num_features)

        # 3. Embeddings
        # (Patch_Len * 6_features) -> D_Model
        self.patch_embedder = nn.Linear(patch_len * self.num_features, D_MODEL)
        # Positional encoding for N_Patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, D_MODEL))
        # Asteroid ID Embedding
        self.asteroid_embed = nn.Embedding(num_asteroids, D_MODEL)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, ...)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # 5. Output Head
        # (N_Patches * D_Model) -> (Output_Steps * 6_features)
        self.output_head = nn.Linear(self.n_patches * D_MODEL, OUTPUT_STEPS * self.num_features)

    def forward(self, x_input, asteroid_ids):
        # x_input shape: (Batch, Input_Steps, 6)
        # asteroid_ids shape: (Batch,)

        # 1. Normalize
        # x shape: (Batch, Input_Steps, 6)
        x = self.revin(x_input, 'normalize') # Store mean/std

        # 2. Patching
        # x shape: (Batch, N_Patches, Patch_Len, 6)
        x = x.reshape(-1, self.n_patches, self.patch_len, self.num_features)
        # x shape: (Batch, N_Patches, Patch_Len * 6)
        x = x.reshape(-1, self.n_patches, self.patch_len * self.num_features)

        # 3. Embeddings
        # x shape: (Batch, N_Patches, D_Model)
        x = self.patch_embedder(x)
        x = x + self.pos_encoding # Add positional
        # ast_emb shape: (Batch, 1, D_Model)
        ast_emb = self.asteroid_embed(asteroid_ids).unsqueeze(1)
        x = x + ast_emb # Add asteroid ID

        # 4. Transformer
        # x shape: (Batch, N_Patches, D_Model)
        x = self.transformer(x)

        # 5. Output Head
        # x shape: (Batch, N_Patches * D_Model)
        x = x.flatten(start_dim=1)
        # pred shape: (Batch, Output_Steps * 6)
        pred = self.output_head(x)
        # pred shape: (Batch, Output_Steps, 6)
        pred = pred.reshape(-1, OUTPUT_STEPS, self.num_features)

        # 6. De-normalize (the "Reversible" part)
        # pred shape: (Batch, Output_Steps, 6)
        pred = self.revin(pred, 'denormalize')

        return pred

```