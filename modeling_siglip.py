from typing import Optional, Tuple
import torch 
import toch.nn as nn

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
        num_image_tokens = int = None,
        **kwargs
    );
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
        self.vision_model = SiglipVisionTransformer(congif)

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
        self.encoder = SignlipEncoder(congif)
        self.post_layernotrm = nn.LayerNorm(embed_dim, eps=)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernotrm(last_hidden_state)

        return last_hidden_state
