# Pairwise LLM Judge Prompt — Designer Role-Play Win Rate
# Usage: GPT-4o as judge, position-debiased (swap A/B and run twice, majority vote)
# Version: v1 (2026-05-25)

## System Prompt

```
You are an expert evaluator for designer cognitive role-play. You will compare two responses generated for the same designer under the same scenario, and judge which response better captures that designer's authentic cognitive style.

You must evaluate along FOUR dimensions, then give a final verdict.

### Dimensions

1. **Persona Fidelity** — Does the response reflect this specific designer's trait profile (belief tendencies, reasoning habits)? Penalize generic designer-speak that could belong to anyone.

2. **Cognitive State Authenticity** — Does the response exhibit plausible cognitive states (perception, knowledge activation, value weighing, choice commitment) rather than surface-level opinion statements? Look for evidence of internal reasoning process, not just conclusions.

3. **Scenario Grounding** — Is the response grounded in the specific design scenario presented? Penalize responses that ignore scenario constraints or give advice applicable to any scenario.

4. **Linguistic Naturalness** — Does the response read like a real designer thinking aloud, with appropriate hedging, domain jargon, and personal style? Penalize robotic, overly structured, or excessively verbose outputs.

### Scoring Rules

- Judge ONLY based on the four dimensions above.
- Do NOT favor longer responses. Concise and precise beats verbose and vague.
- Do NOT let position bias affect your judgment (Response A appearing first does not make it better).
- If both responses are equally good or equally bad, you MUST say Tie.
- Base your judgment on the designer profile provided. A response that is eloquent but inconsistent with the profile is WORSE than a rough but authentic one.

### Output Format (strict JSON)

{
  "persona_fidelity":        {"winner": "A" | "B" | "Tie", "reason": "<1 sentence>"},
  "cognitive_authenticity":   {"winner": "A" | "B" | "Tie", "reason": "<1 sentence>"},
  "scenario_grounding":      {"winner": "A" | "B" | "Tie", "reason": "<1 sentence>"},
  "linguistic_naturalness":  {"winner": "A" | "B" | "Tie", "reason": "<1 sentence>"},
  "final_verdict":           {"winner": "A" | "B" | "Tie", "reason": "<1 sentence>"}
}

The final_verdict should reflect the overall winner considering all four dimensions. It is NOT a simple majority vote — persona_fidelity and cognitive_authenticity should weigh more heavily than the other two.
```

## User Prompt Template

```
### Designer Profile
- **Designer ID**: {designer_id}
- **Cluster**: {cluster_id}
- **Trait Summary**: {trait_summary}
{%- if factor_scores %}
- **Factor Scores**: {factor_scores}
{%- endif %}

### Scenario
- **Question**: {question_id} — {question_text}
{%- if scenario_context %}
- **Context**: {scenario_context}
{%- endif %}

### Response A
{response_a}

### Response B
{response_b}

Compare Response A and Response B for this designer under this scenario. Output your evaluation as JSON.
```

## Position Debiasing Protocol

For each (designer, question) pair, run the judge TWICE:

1. **Round 1**: model output = Response A, baseline = Response B
2. **Round 2**: model output = Response B, baseline = Response A

Aggregate rule:
- If model wins both rounds → **Win**
- If model loses both rounds → **Lose**
- Otherwise (split or any tie) → **Tie**

This eliminates GPT-4o's known position bias toward Response A.

## Caller Code Sketch (Python)

```python
import json, random
from openai import OpenAI

SYSTEM_PROMPT = """..."""  # paste system prompt above
USER_TEMPLATE = """..."""  # paste user template above

client = OpenAI(base_url="https://yunwu.ai/v1", api_key=API_KEY)

def judge_pair(profile: dict, question: dict,
               model_response: str, baseline_response: str,
               model_name: str = "gpt-4o") -> dict:
    results = []
    for swap in [False, True]:
        if swap:
            resp_a, resp_b = baseline_response, model_response
            label_a, label_b = "baseline", "model"
        else:
            resp_a, resp_b = model_response, baseline_response
            label_a, label_b = "model", "baseline"

        user_msg = USER_TEMPLATE.format(
            designer_id=profile["designer_id"],
            cluster_id=profile["cluster_id"],
            trait_summary=profile["trait_summary"],
            factor_scores=profile.get("factor_scores", ""),
            question_id=question["question_id"],
            question_text=question["question_text"],
            scenario_context=question.get("scenario_context", ""),
            response_a=resp_a,
            response_b=resp_b,
        )

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        verdict = json.loads(resp.choices[0].message.content)
        raw_winner = verdict["final_verdict"]["winner"]

        # map back to model/baseline
        if raw_winner == "A":
            actual = label_a
        elif raw_winner == "B":
            actual = label_b
        else:
            actual = "tie"

        results.append({
            "swap": swap,
            "raw_verdict": verdict,
            "actual_winner": actual,
        })

    # aggregate
    wins = [r["actual_winner"] for r in results]
    if wins[0] == "model" and wins[1] == "model":
        final = "win"
    elif wins[0] == "baseline" and wins[1] == "baseline":
        final = "lose"
    else:
        final = "tie"

    return {"final": final, "rounds": results}
```

## Win Rate Computation

```python
def compute_win_rate(judgments: list[dict]) -> dict:
    w = sum(1 for j in judgments if j["final"] == "win")
    l = sum(1 for j in judgments if j["final"] == "lose")
    t = sum(1 for j in judgments if j["final"] == "tie")
    n = len(judgments)
    # standard: win_rate = (wins + 0.5 * ties) / total
    win_rate = (w + 0.5 * t) / n if n > 0 else 0.0
    return {
        "win": w, "lose": l, "tie": t, "total": n,
        "win_rate": round(win_rate, 4),
        "win_rate_strict": round(w / n, 4) if n > 0 else 0.0,
    }
```
