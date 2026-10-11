"""Offline verification of the saved final-paper results; no inference or API calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['seg2_macro_TED','seg2_non_V_recall','seg2_chain_diversity',
           'seg1_trait_pred_rho','seg3_criterion_uniqueness']
EXPECTED = {
    'b0': [2.489,.111,.660,-.018,.459],
    'b1': [2.434,.198,.660,.029,.445],
    'b2': [2.458,.235,.778,-.018,.502],
    'b3': [2.740,.284,.796,.010,.496],
    'v2': [2.121,.383,.821,.037,.545],
    'v2_no_caps': [2.804,.358,.765,-.029,.513],
    'v2_no_seg3_rule': [2.798,.284,.846,.048,.508],
    'v2_no_beta_summary': [2.462,.296,.877,-.043,.517],
    'v2_plus_peer': [2.991,.333,.901,.077,.573],
    'v2_hybrid': [2.121,.383,.821,.037,.503],
}

def main():
    data = json.loads((ROOT/'reports/datav3/cross_model_full_metrics_v2.json').read_text())
    checked = 0
    discrepancies = []
    for condition, expected in EXPECTED.items():
        row = data['gpt4o']['metrics'][condition]
        assert row['n'] == 162
        for key, value in zip(METRICS, expected):
            assert abs(row[key]-value) <= .0005, (condition, key, row[key], value)
            checked += 1
    deltas = {
        'gpt4o': [-.37,.27,.16,.06,.09],
        'gemini31flashlite': [-.15,-.21,.28,.09,.01],
        'claudehaiku45': [-.13,.47,.65,.03,.46],
        'deepseekv4flash': [-.18,.15,.16,.17,.18],
        'qwen35b': [-.29,.23,.26,.08,.09],
    }
    for model, expected in deltas.items():
        rows = data[model]['metrics']
        for key, value in zip(METRICS, expected):
            actual = rows['v2'][key]-rows['b0'][key]
            if abs(actual-value) > .005:
                discrepancies.append({'table': 4, 'model': model, 'metric': key,
                                      'saved_delta': actual, 'paper_delta': value})
            checked += 1
    # Figure 4's prose numbers are traceable to the epoch-5 report, not v3c.
    report = (ROOT/'reports/datav3/sft_base_experiments_summary.md').read_text()
    for token in ['1.821','0.592','0.185','2.516','0.455']:
        assert token in report
    results = {'checked_table_cells': checked, 'exact_table3_table6_cells': 50,
               'table4_discrepancies': discrepancies,
               'sft_figure_prose_source': 'sft_base_experiments_summary.md',
               'training_rows': {f: sum(bool(l.strip()) for l in (ROOT/'data_v3/processed'/f).read_text().splitlines())
                                 for f in ['sft_train.jsonl','sft_val.jsonl']}}
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
