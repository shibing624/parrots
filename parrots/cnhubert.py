import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)


class CNHubert(nn.Module):
    def __init__(self, cnhubert_base_path, sampling_rate=16000):
        super().__init__()
        self.model = HubertModel.from_pretrained(cnhubert_base_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cnhubert_base_path)
        self.sampling_rate = sampling_rate

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=self.sampling_rate
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats


def get_model(model_path, sampling_rate):
    model = CNHubert(model_path, sampling_rate)
    model.eval()
    return model
