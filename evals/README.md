# Evals

Numbers for judging PaperTrail. Settings are in `configs/evals.yaml`.

## Commands (run from the repo root)

```bash
# 1. Retrieval recall vs the reference review (no LLM, uses cached OpenAlex calls)
python -m evals.recall survey_<slug>

# 2. Blind labeling sheet (no model scores in it)
python -m evals.make_label_sheet survey_<slug>
#    fill the `label` column (0-3) in evals/labels/candidates.csv

# 3. Filter comparison (calls Groq, needs --confirm-quota)
python -m evals.filter_eval evals/labels/candidates.csv                  # cost estimate only
python -m evals.filter_eval evals/labels/candidates.csv --confirm-quota

# 4. Record a row set in evals/results.csv
python -m evals.run_eval --tag baseline survey_<slug> [--labels evals/labels/candidates.csv --confirm-quota]
```

## Metrics

- **Pool recall:** share of the review's references (matched by normalized title) found among all candidates.
- **Final recall:** the same, but only among papers that survive the relevance filter.
- **On-topic recall:** the same, restricted to references whose titles contain a topic keyword. This is fairer, because the review also cites general ML and biology papers this search never targets.
- **Spearman:** rank correlation between model scores (0-3) and hand labels. 1 is perfect agreement, 0 is none.
- **Agreement@2:** share of papers where "score >= 2" matches "label >= 2" (keep-or-drop agreement).
- **Saturation:** share of papers given the most common score. 1.0 means the model gave every paper the same score.

## Reading the numbers

- Recall is a lower bound. A review is one expert's list, from 2022, and it cites subtopics we did not search.
- Matching is by title, so a paper with a different title in another version will be missed.
- Labels come from one person. With about 60 papers, Spearman has wide error bars. Always report n.

## Results file

`evals/results.csv` is long format: `date, tag, metric, model, value, n`. Compare runs by tag, for example `baseline` against a later `snowball`.

Labels and results are local and gitignored.
