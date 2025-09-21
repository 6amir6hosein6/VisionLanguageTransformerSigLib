from typing import Optional, Tuple
import torch 
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_channels=3,
        image_size=224,
        patch_size=12,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self. num_image_tokens = num_image_tokens


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:

        return self.vision_model(pixel_values=pixel_values)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid", 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_postitions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_postitions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_postitions).expand((1,-1)),
            persistent=False,

        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)

        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings




class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SignlipEncoder(config)
        self.post_layernotrm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernotrm(last_hidden_state)

        return last_hidden_state

class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads =config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, 
                hidden_state: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_state.size()

        query_state = self.q_proj(hidden_state)
        key_state = self.k_proj(hidden_state)
        value_state = self.v_proj(hidden_state)

        query_state = query_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transposer(1, 2)
        key_state = key_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transposer(1, 2)
        value_state = value_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transposer(1, 2)

        atten_weights = (torch.matmul(query_state, key_state.transpose(2, 3)) * self.scale)

        if atten_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}"
                f" {atten_weights.size()}"
            )
        
        atten_weights = nn.functional.softmax(atten_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
        atten_weights = nn.functional.dropout(atten_weights, p=self.dropout, training=self.training)

        atten_out = torch.matmul(atten_weights, value_state)

        if atten_out.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention out should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}"
                f" {atten_out.size()}"
            )
        
        atten_out = atten_out.transpose(1, 2).contiguous()
        atten_out = atten_out.reshape(batch_size, seq_len, self.embed_dim)

        atten_out = self.out_proj(atten_out)

        return atten_out, atten_weights

class SignlipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norml2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states 

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states 

        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class SignlipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SignlipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        
        hidden_state = inputs_embeds

        for encoder_layer in self.layers:
            hidden_state = encoder_layer(hidden_state)

        return hidden_state