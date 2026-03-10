from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoReference:
    model: str
    test_rmse: float
    source: str
    notes: str


REPO_REFERENCES = [
    RepoReference(
        model="B1_HGBR",
        test_rmse=0.4430,
        source="california_v35_routed_regression full run",
        notes="current-run baseline on the repo California split",
    ),
    RepoReference(
        model="G2_GoRA_v1_ref",
        test_rmse=0.4546,
        source="saved v3 reference",
        notes="saved reference, not rerun on the aligned foundation branch",
    ),
    RepoReference(
        model="CA_v35b",
        test_rmse=0.4762,
        source="california_v35_routed_regression full run",
        notes="regression-safe routed baseline without regression label-context",
    ),
    RepoReference(
        model="HR_v4_headgated_diverse",
        test_rmse=0.4722,
        source="head-gated decoder branch",
        notes="shared head-routing backbone with diverse decoder smoke/full branch result",
    ),
]
