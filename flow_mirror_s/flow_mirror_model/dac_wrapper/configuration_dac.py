from transformers import PretrainedConfig
from typing import List


class DACConfig(PretrainedConfig):
    model_type = "cac"


    def __init__(
        self,
        num_codebooks: int = 9,
        encoder_rates: List[int] = [2, 4, 8, 8],
        decoder_rates: List[int] = [8, 8, 4, 2],
        model_bitrate: int = 8,  # kbps
        codebook_size: int = 1024,
        latent_dim: int = 1024,
        frame_rate: int = 86,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        self.codebook_size = codebook_size
        self.encoder_rates = encoder_rates
        self.decoder_rates = decoder_rates
        self.model_bitrate = model_bitrate
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.frame_rate = frame_rate
        self.sampling_rate = sampling_rate

        super().__init__(**kwargs)
