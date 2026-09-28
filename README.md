# AAD-speech-representation
Code for paper "The Effect of Speech Representations on EEG-based Auditory Attention Detection"
This study conducts a comprehensive analysis of speech representations for the task of EEG-based Auditory Attention Detection (AAD). We systematically compare four distinct speech representations: speech envelope, spectral features (STFT), a deep acoustic representation (Wav2Vec), and a semantic representation.

## Code

The reusable cross-modal feature fusion module is available in [`cmff.py`](cmff.py). It implements cosine-similarity alignment, EEG-driven cognitive modulation, complementary competition between the two speech streams, and the final classifier.

```python
from cmff import CMFF

model = CMFF(feature_dim=128, hidden_dim=64, num_classes=2)
logits = model(eeg_features, speech_features_1, speech_features_2)
```

Each input tensor must have shape `[batch_size, feature_dim]`.

> ### **Note on Code Availability**
>
> The remaining training and data-processing code is currently under preparation. Please feel free to explore the paper and raise questions by opening an issue in this repository.
