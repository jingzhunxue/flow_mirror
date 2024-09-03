# Flow_mirror_s

## Requirements
```
conda create -n flowmirror python=3.10
conda activate flowmirror

# downgrade pip to 23.1.1 for the requirments of fairseq
pip install pip==23.1.1

pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Model download
### modelscope
```python
from modelscope import snapshot_download
snapshot_download('jzx-ai-lab/Flow_mirror', local_dir='jzx-ai-lab/Flow_mirror')
```
## Load flow_mirror model
### Load model

```python
from flow_mirror_model import FlowmirrorForConditionalGeneration
from hubert_kmeans import HubertCodeExtractor
from transformers import AutoTokenizer

ckpt_path = "jzx-ai-lab/Flow_mirror" # download from modelscope or huggingface 
model = FlowmirrorForConditionalGeneration.from_pretrained(ckpt_path)
code_extractor = HubertCodeExtractor(
    ckpt_path=f"{ckpt_path}/chinese-hubert-ckpt-20240628.pt",
    km_path="hubert_kmeans/kmeans_500.pkl",
    layer=24,
    rank=0
)
tokenizer = AutoTokenizer.from_pretrained(f"{ckpt_path}/tokenizer")

model.eval().to(torch.float16).to("cuda")
```
### Load speaker_embedding from pt
```python
speaker_embeddings = torch.load("hubert_kmeans/speaker_embedding.pt")
```
### Extract speaker_embedding from ref-audio(make sure the sampling rate of the audio is 16k)
```python
from transformers import AutoFeatureExtractor
import soundfile as sf

speaker_encoder = model.speaker_encoder

feature_extractor = AutoFeatureExtractor.from_pretrained("hubert_kmeans")


ref_wav = f"{ckpt_path}/assets/question_example_1_MP3.mp3"
reference_audio_input = feature_extractor(sf.read(wav_example)[0],sampling_rate=16000, return_tensors="pt").to("cuda")
speaker_embedding = speaker_encoder.encode(reference_audio_input['input_values'])
```

## Inference Code
```python
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

def convert_label_to_text(label):
    text = ""
    for i in label:
        text += f"<|audio_{i}|>"
    return text

# extract code token from hubert feature
feats = code_extractor.get_feats(f"{ckpt_path}/assets/question_example_1_MP3.mp3")
codes = code_extractor.dump_label(feats)

codes = deduplicates(codes)
label_text = convert_label_to_text(codes)

# apply mode generation template
prompt = f"<|spk_embed|><|startofaudio|>{label_text}<|endofaudio|><|startofcont|>"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# define generation config
gen_kwargs = {
    "do_sample": True,
    "temperature": 0.9,
    "max_new_tokens": 512,
    "use_cache": True,
    "min_new_tokens": 9 + 1,
}

generation, text_completion = model.generate(prompt_input_ids=input_ids.to("cuda"),speaker_embedding=speaker_embedding.to(model.dtype).to(model.device), **gen_kwargs)

audio_arr = generation.float().cpu().numpy().squeeze()

# print generated text
print(tokenizer.decode(text_completion[0]))
# save generated audio
sf.write("answer.wav", audio_arr, 16000)
```

