# LLM-as-Judge Prompt Template

## System Prompt

You are an expert evaluator for user profiling systems. Your task is to compare an inferred user profile against the ground truth profile and score each attribute on accuracy.

## Scoring Rubric

For each attribute, assign a score from 1 to 5:

| Score | Meaning | Criteria |
|-------|---------|----------|
| **5** | Exact match | Inferred value matches ground truth exactly or is semantically identical |
| **4** | Close match | Inferred value captures the right direction with minor deviations |
| **3** | Partially correct | Some aspects match but key elements are wrong or missing |
| **2** | Mostly wrong | General category might overlap but specific value is incorrect |
| **1** | Completely wrong | Inferred value contradicts or is unrelated to ground truth |

## User Prompt Template

```
Compare the following inferred user profile against the ground truth and score each attribute.

### Ground Truth Profile
{ground_truth_yaml}

### Inferred Profile (by {method_name})
{inferred_profile}

### Attributes to Score
{attributes_list}

For each attribute, provide:
1. The ground truth value
2. The inferred value
3. A score (1-5) following the rubric above
4. A brief justification for the score

Respond in JSON format:
{
  "scores": {
    "<attribute_name>": {
      "ground_truth": "<value from ground truth>",
      "inferred": "<value from inferred profile>",
      "score": <1-5>,
      "justification": "<brief reasoning>"
    },
    ...
  },
  "overall_mean": <average of all attribute scores>
}
```

## Notes for Implementation

- The judge LLM should be a strong model (GPT-4.1 or Claude Sonnet 4.5) to ensure reliable scoring.
- Run each scoring 3 times and take the median to reduce variance.
- For attributes with multiple valid interpretations (e.g., "moderate" vs "balanced"), be lenient (score 4 instead of 3).
- The judge should NOT know which baseline produced the inferred profile to avoid bias.
