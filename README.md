# AAD-speech-representation

Code accompanying the paper [The Effect of Speech Representations on EEG-based Auditory Attention Detection](https://www.sciencedirect.com/science/article/pii/S0167865526000929?dgcid=author).

This study conducts a comprehensive analysis of speech representations for EEG-based auditory attention detection (AAD). It compares four distinct speech representations: speech envelope, spectral features (STFT), a deep acoustic representation (Wav2Vec), and a semantic representation.

## Code

The reusable cross-modal feature fusion module is provided in [`cmff.py`](cmff.py). It implements cosine-similarity alignment, EEG-driven cognitive modulation, complementary competition between the two speech streams, and the final classifier.

```python
from cmff import CMFF

model = CMFF(feature_dim=128, hidden_dim=64, num_classes=2)
logits = model(eeg_features, speech_features_1, speech_features_2)
```

Each input tensor must have shape `[batch_size, feature_dim]`.
