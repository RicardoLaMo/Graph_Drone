from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans


EPS = 1e-8


@dataclass(frozen=True)
class SegmentationResult:
    name: str
    segment_ids: np.ndarray
    feature_names: list[str]
    features: np.ndarray
    summary_rows: list[dict[str, float | int | str]]


class GeoSegmenter:
    def fit(self, geo_train: np.ndarray) -> "GeoSegmenter":
        raise NotImplementedError

    def predict(self, geo: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GridSegmenter(GeoSegmenter):
    def __init__(self, lat_step: float, lon_step: float):
        self.lat_step = float(lat_step)
        self.lon_step = float(lon_step)
        self.lat0: float | None = None
        self.lon0: float | None = None

    def fit(self, geo_train: np.ndarray) -> "GridSegmenter":
        self.lat0 = float(geo_train[:, 0].min())
        self.lon0 = float(geo_train[:, 1].min())
        return self

    def predict(self, geo: np.ndarray) -> np.ndarray:
        assert self.lat0 is not None and self.lon0 is not None
        lat_bin = np.floor((geo[:, 0] - self.lat0) / self.lat_step).astype(np.int64)
        lon_bin = np.floor((geo[:, 1] - self.lon0) / self.lon_step).astype(np.int64)
        return (lat_bin * 10000 + lon_bin).astype(np.int64)


class KMeansSegmenter(GeoSegmenter):
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.model: KMeans | None = None

    def fit(self, geo_train: np.ndarray) -> "KMeansSegmenter":
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state,
        ).fit(geo_train)
        return self

    def predict(self, geo: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(geo).astype(np.int64)

    def centroid_distance(self, geo: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        assert self.model is not None
        centroids = self.model.cluster_centers_[segment_ids]
        return np.linalg.norm(geo - centroids, axis=1).astype(np.float32)


def _segment_stats_from_train(
    y: np.ndarray,
    seg_ids: np.ndarray,
    train_idx: np.ndarray,
) -> dict[int, tuple[float, float, float]]:
    global_mean = float(y[train_idx].mean())
    global_std = float(y[train_idx].std() + EPS)
    global_count = float(len(train_idx))
    stats: dict[int, tuple[float, float, float]] = {}
    train_seg = seg_ids[train_idx]
    for seg in np.unique(train_seg):
        mask = train_seg == seg
        y_seg = y[train_idx][mask]
        stats[int(seg)] = (
            float(y_seg.mean()),
            float(y_seg.std() + EPS),
            float(len(y_seg)),
        )
    stats[-1] = (global_mean, global_std, global_count)
    return stats


def _features_from_stats(
    seg_ids: np.ndarray,
    stats: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    global_mean, global_std, global_count = stats[-1]
    feats = np.zeros((len(seg_ids), 4), dtype=np.float32)
    for i, seg in enumerate(seg_ids):
        mean, std, count = stats.get(int(seg), (global_mean, global_std, global_count))
        feats[i, 0] = mean
        feats[i, 1] = std
        feats[i, 2] = np.log1p(count)
        feats[i, 3] = count / global_count
    return feats


def _summary_rows(
    name: str,
    seg_ids: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    train_seg = seg_ids[train_idx]
    for seg in np.unique(train_seg):
        mask = train_seg == seg
        y_seg = y[train_idx][mask]
        rows.append(
            {
                "scheme": name,
                "segment_id": int(seg),
                "count": int(mask.sum()),
                "target_mean": float(y_seg.mean()),
                "target_std": float(y_seg.std() + EPS),
            }
        )
    rows.sort(key=lambda row: int(row["count"]), reverse=True)
    return rows


def build_segmentation_result(
    name: str,
    segmenter: GeoSegmenter,
    raw_geo: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> SegmentationResult:
    segmenter.fit(raw_geo[train_idx])
    seg_ids = segmenter.predict(raw_geo)
    stats = _segment_stats_from_train(y, seg_ids, train_idx)
    features = _features_from_stats(seg_ids, stats)
    feature_names = [
        f"{name}_target_mean",
        f"{name}_target_std",
        f"{name}_log_count",
        f"{name}_count_frac",
    ]
    if isinstance(segmenter, KMeansSegmenter):
        centroid_dist = segmenter.centroid_distance(raw_geo, seg_ids)[:, None]
        features = np.concatenate([features, centroid_dist], axis=1)
        feature_names.append(f"{name}_centroid_dist")
    return SegmentationResult(
        name=name,
        segment_ids=seg_ids,
        feature_names=feature_names,
        features=features.astype(np.float32),
        summary_rows=_summary_rows(name, seg_ids, y, train_idx),
    )


def build_default_segmentations(
    raw_geo: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> Dict[str, SegmentationResult]:
    schemes: dict[str, GeoSegmenter] = {
        "grid_coarse": GridSegmenter(lat_step=0.50, lon_step=0.50),
        "grid_fine": GridSegmenter(lat_step=0.25, lon_step=0.25),
        "kmeans32": KMeansSegmenter(n_clusters=32),
        "kmeans96": KMeansSegmenter(n_clusters=96),
    }
    return {
        name: build_segmentation_result(name, segmenter, raw_geo, y, train_idx)
        for name, segmenter in schemes.items()
    }
