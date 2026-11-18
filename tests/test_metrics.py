import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from metrics import code_match, evidence_grounding, results_recall, set_f1  # noqa: E402

from papertrail.schemas import CodeInfo, Dataset, Evidenced, MetricResult, PaperCard  # noqa: E402


def make_card(**kw) -> PaperCard:
    base = dict(
        paper_id="x",
        title="T",
        problem=Evidenced(claim="p", evidence="We predict antibody aggregation from sequence."),
        method_summary=Evidenced(claim="m"),
    )
    base.update(kw)
    return PaperCard(**base)


def test_set_f1():
    assert set_f1(["SAbDab", "OAS"], ["sabdab"]) == 2 / 3
    assert set_f1([], []) is None
    assert set_f1(["a"], []) == 0.0


def test_results_recall_uses_main_result_and_tolerance():
    gold = make_card(results=[MetricResult(metric="Spearman", value=0.71, is_main_result=True)])
    pred = make_card(results=[MetricResult(metric="spearman", value=0.7105)])
    assert results_recall(pred, gold) == 1.0


def test_code_match_rewards_honest_no_code():
    gold, pred = make_card(), make_card()
    assert code_match(pred, gold) == 1.0
    pred_bad = make_card(code=CodeInfo(repo_url="https://github.com/fake/repo"))
    assert code_match(pred_bad, gold) == 0.0


def test_evidence_grounding():
    card = make_card()
    assert evidence_grounding(card, "Intro. We predict antibody aggregation from sequence. More.") == 1.0
    assert evidence_grounding(card, "Totally different text.") == 0.0


def test_template_is_valid_schema_shape():
    card = make_card(datasets=[Dataset(name="SAbDab", role="train")])
    assert PaperCard.model_validate_json(card.model_dump_json()) == card
