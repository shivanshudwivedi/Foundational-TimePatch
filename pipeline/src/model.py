"""
Defines the Foundational Asteroid Transformer model architecture.
"""

import torch
import torch.nn as nn

# Import all hyperparameters from our config file
import config

# -------------------------------------------------------------------
# 1. Reversible Instance Normalization (RevIN)
# -------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN)
    
    This is a crucial component for SOTA time-series models.
    It normalizes each sample (instance) independently by its mean/std
    and stores them to de-normalize the model's output.
    """
    def __init__(self, num_features: int, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # We don't use nn.Parameter, as these are not learnable
        # They are statistics of the *input sample*
        self.mean = None
        self.std = None

    def _normalize(self, x):
        """Normalize the input `x` (B, L, F)"""
        # Calculate mean/std along the sequence (L) dimension
        # Keep dim for broadcasting: (B, 1, F)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        
        # Store for de-normalization
        # We detach them so they don't become part of the compute graph
        self.mean = mean.detach()
        self.std = std.detach()
        
        # Normalize
        x_norm = (x - self.mean) / (self.std + self.eps)
        return x_norm

    def _denormalize(self, x_norm):
        """De-normalize the output `x_norm` (B, L_out, F)"""
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call 'normalize' before 'denormalize'.")
        
        # Apply the stored mean/std
        # The mean/std are (B, 1, F), which will broadcast
        # over the output sequence length (L_out)
        x_denorm = (x_norm * self.std) + self.mean
        return x_denorm

    def forward(self, x, mode: str):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F)
            mode (str): 'normalize' or 'denormalize'
        """
        if mode == 'normalize':
            return self._normalize(x)
        elif mode == 'denormalize':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be 'normalize' or 'denormalize'.")

# -------------------------------------------------------------------
# 2. Foundational Asteroid Transformer
# -------------------------------------------------------------------

class AsteroidTransformer(nn.Module):
    """
    A foundational, patch-based Transformer model for asteroid forecasting.
    
    Combines:
    1. RevIN (Reversible Instance Normalization)
    2. Patching (treats time-series like patches in an image)
    3. Asteroid ID Embeddings (to learn per-asteroid dynamics)
    4. A standard Transformer Encoder
    """
    
    def __init__(self):
        super().__init__()
        
        # --- Store config values ---
        self.num_features = config.NUM_FEATURES
        self.input_steps = config.INPUT_STEPS
        self.output_steps = config.OUTPUT_STEPS
        
        self.patch_len = config.PATCH_LEN
        self.n_patches = config.N_PATCHES
        self.d_model = config.D_MODEL
        
        # --- 1. Normalization Layer ---
        self.revin = RevIN(self.num_features)
        
        # --- 2. Embedding Layers ---
        
        # Patch Embedder: A Linear layer to project a flattened patch
        # Input dim: patch_len * num_features (e.g., 4 * 6 = 24)
        # Output dim: d_model (e.g., 128)
        self.patch_embedder = nn.Linear(
            self.patch_len * self.num_features, 
            self.d_model
        )
        
        # Asteroid ID Embedder: A learnable vector for each asteroid
        # We make it d_model so it can be added directly
        self.asteroid_embedder = nn.Embedding(
            config.N_ASTEROIDS, 
            self.d_model 
        )
        
        # Positional Encoding: A learnable parameter for each *patch*
        # Shape: (1, N_PATCHES, D_MODEL) for broadcasting
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, self.d_model)
        )
        
        # --- 3. Transformer Encoder ---
        
        # First, define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_FF,
            dropout=config.DROPOUT,
            activation=config.ACTIVATION,
            batch_first=True  # CRITICAL: Ensures (B, L, F) input
        )
        
        # Then, stack them to create the full Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.N_LAYERS
        )
        
        # --- 4. Output Head (Forecasting Head) ---
        
        # A Linear layer to map the Transformer's output
        # to the desired forecast shape.
        
        # Input dim: Flattened encoder output (N_PATCHES * D_MODEL)
        # Output dim: Flattened forecast (OUTPUT_STEPS * NUM_FEATURES)
        self.output_head = nn.Linear(
            self.n_patches * self.d_model,
            self.output_steps * self.num_features
        )

    def forward(self, x_input, asteroid_ids):
        """
        Performs the forward pass of the model.
        
        Args:
            x_input (torch.Tensor): 
                Shape: (B, INPUT_STEPS, NUM_FEATURES)
                       (e.g., 128, 24, 6)
                       
            asteroid_ids (torch.Tensor): 
                Shape: (B,) 
                (e.g., 128)
                       
        Returns:
            torch.Tensor: The forecast.
                Shape: (B, OUTPUT_STEPS, NUM_FEATURES)
                       (e.g., 128, 6, 6)
        """
        # Get Batch Size from input
        B = x_input.shape[0]
        
        # 1. Normalization
        # x shape: (B, 24, 6)
        x = self.revin(x_input, 'normalize')
        
        # 2. Patching
        # Reshape to (B, N_PATCHES, PATCH_LEN, NUM_FEATURES)
        # x shape: (B, 6, 4, 6)
        x = x.reshape(B, self.n_patches, self.patch_len, self.num_features)
        
        # Flatten patches: (B, N_PATCHES, PATCH_LEN * NUM_FEATURES)
        # x shape: (B, 6, 24)
        x = x.reshape(B, self.n_patches, self.patch_len * self.num_features)
        
        # 3. Embedding
        # Project patches to d_model
        # x shape: (B, 6, 128)
        x = self.patch_embedder(x)
        
        # Get asteroid ID embedding: (B, 128)
        # and add a dimension for broadcasting: (B, 1, 128)
        ast_emb = self.asteroid_embedder(asteroid_ids).unsqueeze(1)
        
        # Add all embeddings together
        # (B, 6, 128) + (1, 6, 128) + (B, 1, 128)
        # All broadcast correctly to (B, 6, 128)
        x = x + self.pos_encoding + ast_emb
        
        # 4. Transformer Encoder
        # x shape: (B, 6, 128) -> (B, 6, 128)
        x = self.transformer(x)
        
        # 5. Output Head
        # Flatten the encoder output: (B, 6 * 128)
        x = x.flatten(start_dim=1)
        
        # Map to forecast shape: (B, 6 * 6)
        pred = self.output_head(x)
        
        # Reshape to final forecast: (B, 6, 6)
        pred = pred.reshape(B, self.output_steps, self.num_features)
        
        # 6. De-normalize
        # This is the "Reversible" part of RevIN
        pred = self.revin(pred, 'denormalize')
        
        return pred