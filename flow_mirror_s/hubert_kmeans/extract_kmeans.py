import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F

from .feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform

import joblib
import numpy as np

import torchaudio

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")

class HubertCodeExtractor(object):
    def __init__(self, ckpt_path, km_path, layer, rank, max_chunk=1600000, dtype=torch.bfloat16):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.dtype = dtype
        self.model = model[0].eval().to(dtype).to("cuda:"+str(rank))
        self.task = task
        self.layer = layer
        self.rank = rank
        self.max_chunk = max_chunk

        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to("cuda:"+str(rank))
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to("cuda:"+str(rank))
        
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav , sr = torchaudio.load(path)
        wav = wav.numpy()
        if wav.ndim == 2:
            print("avarage")
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        if isinstance(path, str):
            x = self.read_audio(path, ref_len=ref_len)
        else:
            x = path
        with torch.no_grad():
            x = torch.from_numpy(x).float().to("cuda:"+str(self.rank))
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk.to(self.dtype),
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)
    
    def dump_label(self, x):
        x = x.to(torch.float32)
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

if __name__ == "__main__":
    wav_data = "../../assets/sts_in.wav"
    feature_extractor = HubertCodeExtractor(
        ckpt_path="ckpt_path/chinese-hubert-ckpt.pt",
        km_path="ckpt_path/hubert_kmeans/kmeans_500.pkl",
        layer=24,
        rank=0,
    )

    feat = feature_extractor.get_feats(wav_data)

    label = feature_extractor.dump_label(feat)

    print(label, len(label))

    def deduplicates(cluster_ids):
        dup_cluster_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                count = 1
        return dup_cluster_list
    
    print(deduplicates(label), len(deduplicates(label)))


    

