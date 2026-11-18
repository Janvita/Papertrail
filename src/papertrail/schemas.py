"""The PaperCard schema: the single source of truth for what the Reader agent extracts.

Design notes (good interview talking points):
- Every important claim carries an `evidence` quote from the paper. The Critic agent
  (week 7) checks these quotes against the parsed text to catch hallucinations.
- Fields are Optional / default-empty on purpose: "not reported in paper" is a valid,
  honest answer. Forcing a value invites the model to invent one.
- The same schema is used for the hand-labeled golden cards, so evals compare like with like.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidenced(BaseModel):
    """A short claim plus the sentence from the paper that supports it."""

    claim: str
    evidence: Optional[str] = Field(
        default=None, description="Verbatim sentence from the paper supporting the claim"
    )


class Dataset(BaseModel):
    name: str = Field(description="Dataset name as written in the paper")
    size: Optional[str] = Field(default=None, description="e.g. '12k sequences'")
    public: Optional[bool] = Field(default=None, description="True if publicly available")
    split: Optional[str] = Field(
        default=None, description="How train/test were split, e.g. 'random 80/20', 'cluster-based'"
    )
    role: Literal["train", "test", "train+test", "external_validation", "unknown"] = "unknown"


class MetricResult(BaseModel):
    metric: str = Field(description="e.g. 'Spearman', 'AUROC', 'RMSE'")
    value: float
    dataset: Optional[str] = None
    is_main_result: bool = Field(default=False, description="The headline number of the paper")


class CodeInfo(BaseModel):
    url_in_paper: Optional[str] = Field(default=None, description="Repo link written in the paper")
    repo_url: Optional[str] = Field(default=None, description="Repo found/verified by Code Hunter")
    verified: bool = False
    stars: Optional[int] = None
    last_push: Optional[str] = None
    has_readme: Optional[bool] = None
    notes: Optional[str] = None


class PaperCard(BaseModel):
    # identity
    paper_id: str = Field(description="arXiv id, DOI, or file stem")
    title: str
    year: Optional[int] = None

    # what & how
    problem: Evidenced = Field(description="The problem the paper solves, one sentence")
    task_type: Optional[str] = Field(
        default=None, description="e.g. 'regression', 'classification', 'generation'"
    )
    method_summary: Evidenced = Field(description="Core method in 1-2 sentences")
    model_family: Optional[str] = Field(
        default=None, description="e.g. 'protein language model', 'GNN', 'gradient boosting'"
    )

    # data & results
    datasets: list[Dataset] = Field(default_factory=list)
    results: list[MetricResult] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list, description="Methods compared against")

    # judgement
    strengths: list[Evidenced] = Field(default_factory=list)
    weaknesses: list[Evidenced] = Field(default_factory=list)

    # lineage & code
    builds_on: list[str] = Field(
        default_factory=list, description="Titles of 1-3 prior papers this work most directly extends"
    )
    code: CodeInfo = Field(default_factory=CodeInfo)
