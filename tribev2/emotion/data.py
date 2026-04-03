# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ClassificationData: adapts tribev2's Data class for speech emotion classification.

The main change is that the fMRI (``neuro``) extractor becomes optional and an
emotion label extractor is added so that each segment carries a discrete
emotion class index.
"""

import logging
import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events
from torch.utils.data import DataLoader

from ..main import Data, _free_extractor_model
from ..utils import split_segments_by_time

LOGGER = logging.getLogger(__name__)


class ClassificationData(Data):
    """Data configuration for emotion classification.

    Differences from :class:`Data`:
    * ``neuro`` is optional (defaults to ``None``).
    * A ``num_classes`` field controls the number of discrete emotion labels.
    * An ``emotion_label`` extractor is automatically created to encode the
      ``emotion`` field from ``Audio`` events.
    """

    # Make neuro optional -- not needed for emotion classification
    neuro: ns.extractors.BaseExtractor | None = None  # type: ignore[assignment]

    # Classification-specific
    num_classes: int = 4

    # Emotion label extractor -- built automatically
    emotion_label: ns.extractors.LabelEncoder = ns.extractors.LabelEncoder(
        event_field="emotion",
        event_types="Audio",
        aggregation="first",
        allow_missing=True,
    )

    # ------------------------------------------------------------------
    # TR property -- fall back to ``self.frequency`` or the first available
    # feature extractor when ``neuro`` is not set.
    # ------------------------------------------------------------------

    @property
    def TR(self) -> float:  # noqa: N802
        if self.neuro is not None:
            return 1 / self.neuro.frequency
        if self.frequency is not None:
            return 1 / self.frequency
        # Fall back: use the frequency of the first available feature extractor
        for modality in self.features_to_use:
            extractor = getattr(self, f"{modality}_feature", None)
            if extractor is not None and hasattr(extractor, "frequency") and extractor.frequency:
                return 1 / extractor.frequency
        raise ValueError(
            "Cannot determine TR: set `neuro`, `frequency`, or provide a "
            "feature extractor with a `frequency` attribute."
        )

    # ------------------------------------------------------------------
    # get_loaders -- add the emotion_label extractor to the extractor dict
    # ------------------------------------------------------------------

    def get_loaders(
        self,
        events: pd.DataFrame | None = None,
        split_to_build: tp.Literal["train", "val", "all"] | None = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events()
        else:
            events = standardize_events(events)

        # Build extractor dict for stimulus features
        extractors: dict[str, ns.extractors.BaseExtractor] = {}
        for modality in self.features_to_use:
            extractors[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique() and self.neuro is not None:
            extractors["fmri"] = self.neuro

        # Dummy categorical events (same logic as parent)
        dummy_events = []
        for timeline_name, timeline in events.groupby("timeline"):
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                assert (
                    len(splits) == 1
                ), f"Timeline {timeline_name} has multiple splits: {splits}"
                split = splits[0]
            else:
                split = "all"
            dummy_event = {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
            dummy_events.append(dummy_event)
        events = pd.concat([events, pd.DataFrame(dummy_events)])
        events = standardize_events(events)

        # Classification-specific: include emotion label extractor
        extractors["emotion_label"] = self.emotion_label
        extractors["subject_id"] = self.subject_id

        # Remove extractors whose event types are not present in events
        features_to_remove = set()
        for extractor_name, extractor in extractors.items():
            event_types = EventTypesHelper(extractor.event_types).names
            if not any(
                event_type in events.type.unique() for event_type in event_types
            ):
                features_to_remove.add(extractor_name)
        for extractor_name in features_to_remove:
            del extractors[extractor_name]
            LOGGER.warning(
                "Removing extractor %s as there are no corresponding events",
                extractor_name,
            )

        for name, extractor in extractors.items():
            LOGGER.info("Preparing extractor: %s", name)
            extractor.prepare(events)
            _free_extractor_model(extractor)

        # Build dataloaders
        loaders: dict[str, DataLoader] = {}
        if split_to_build is None:
            splits = ["train", "val"]
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all" or self.split_segments_by_time:
                split_sel = [True] * len(events)
                shuffle = False
                overlap_trs = self.overlap_trs_train
            else:
                split_sel = events.split == split
                if split not in events.split.unique():
                    shuffle = False
                else:
                    shuffle = (
                        self.shuffle_train if split == "train" else self.shuffle_val
                    )
                if split == "val":
                    overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                else:
                    overlap_trs = self.overlap_trs_train

            sel = np.array(split_sel)
            segments = ns.segments.list_segments(
                events[sel],
                triggers=events[sel].type == "CategoricalEvent",
                stride=(self.duration_trs - overlap_trs) * self.TR,
                duration=self.duration_trs * self.TR,
                stride_drop_incomplete=self.stride_drop_incomplete,
            )
            if self.split_segments_by_time:
                LOGGER.info(f"Total number of segments: {len(segments)}")
                segments = split_segments_by_time(
                    segments,
                    val_ratio=self.study.transforms["split"].val_ratio,
                    split=split,
                )
                LOGGER.info(f"# {split} segments: {len(segments)}")
            if len(segments) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            dataset = ns.dataloader.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
            loaders[split] = dataloader

        return loaders
