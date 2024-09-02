from transformers import PreTrainedModel, PretrainedConfig
from .campplus.DTDNN import CAMPPlus
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
import numpy as np

class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

class SpeakerRecCAMPPConfig(PretrainedConfig):
    model_type = "speaker_rec"
    def __init__(self, feat_dim=80, embedding_size=192, growth_rate=32, init_channels=128, config_str="batchnorm-relu", memory_efficient=True,output_level="segment", **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.growth_rate = growth_rate
        self.init_channels = init_channels
        self.config_str = config_str
        self.memory_efficient = memory_efficient
        self.output_level = output_level


class SpeakerRecCAMPP(PreTrainedModel):
    config_class = SpeakerRecCAMPPConfig

    def __init__(self, config: SpeakerRecCAMPPConfig):
        if config is None:
            raise ValueError("config cannot be None")
        super().__init__(config)

        self.model = CAMPPlus(feat_dim = config.feat_dim, 
                              embedding_size = config.embedding_size, 
                              growth_rate=config.growth_rate,
                              init_channels=config.init_channels,
                              config_str=config.config_str,
                              memory_efficient=config.memory_efficient)

        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    @torch.no_grad()
    def encode(self, input_values, **kwargs):
        # convert sample rate tp 16k
        assert input_values.dim() == 3, "input_values should be [B, C, T]"

        input_values = input_values.mean(dim=1, keepdim=True)
    
        feat = torch.stack([self.feature_extractor(input_values[i]) for i in range(input_values.shape[0])], dim=0)
        
        embedding = self.model(feat)
        return embedding
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
    
if __name__ == "__main__":

    # model = SpeakerRecERes2Net.from_pretrained("parler_tts/speaker_rec/speech_eres2net_sv_zh-cn_16k-common", cache_dir="parler_tts/speaker_rec/speech_eres2net_sv_zh-cn_16k-common", local_files_only=True)
    config = SpeakerRecCAMPPConfig()
    model = SpeakerRecCAMPP.from_config(config)

    wav = "data/X0000000000_100638174_S00012.wav"

    wav = model.process(wav).unsqueeze(0)

    print(wav.shape)

    embedding = model(wav)

    print(embedding)
        
        
        

        
        



