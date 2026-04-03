# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom audio extractor for wav2vec2 models with vocab_size=None config issue."""

import torch
from neuralset.extractors.audio import HuggingFaceAudio


class Wav2VecEmotion(HuggingFaceAudio):
    """Wav2Vec2 extractor that patches the vocab_size=None config issue.

    The audeering emotion models have ``vocab_size: null`` in their config,
    which causes a validation error in newer huggingface_hub versions.
    This subclass patches the config before loading.
    """

    model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        import json

        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        # Download config and patch vocab_size=null before Wav2Vec2Config
        # validates it (huggingface_hub strict dataclass validation rejects None).
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        if config_dict.get("vocab_size") is None:
            config_dict["vocab_size"] = 32
        config = Wav2Vec2Config(**config_dict)

        model = Wav2Vec2Model.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)
        model.eval()
        return model
