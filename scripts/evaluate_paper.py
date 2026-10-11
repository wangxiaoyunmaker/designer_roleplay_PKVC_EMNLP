"""Run the original bootstrap evaluator on final-paper conditions only.

This calls the embedding API if cached embeddings are missing. Supply API credentials
in environment variables; the excluded embedding cache can be regenerated.
"""
import argparse
from pathlib import Path
import scripts.final_eval_with_bootstrap as evaluation

GPT_FILES = ['ladder_b0_h','ladder_b1_h','ladder_b2_h','ladder_b3_h','b6_v2_full',
             'ablation_v2_no_caps','ablation_v2_no_seg3_rule','ablation_v2_no_beta_summary',
             'ablation_v2_plus_peer','b6_v2hybrid_full']
LABELS = ['B0','B1','B2','B3','B4','B4-NoCaps','B4-NoRule','B4-NoSum','B4-PlusPeer','B4-Hybrid']
CONDITIONS = ['b0','v2','v2_no_caps','v2_no_seg3_rule','v2_no_beta_summary','v2_hybrid','v2_plus_peer']
KEEP_METRICS = {'seg2_macro_TED','seg2_non_V_recall','seg2_chain_diversity',
                'seg1_trait_pred_rho','seg3_criterion_uniqueness'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['gpt4o','sft'], default='gpt4o')
    args = p.parse_args()
    if args.group == 'gpt4o':
        pairs = list(zip(LABELS, GPT_FILES))
    else:
        pairs = [(f'{model}/{cond}', f'{model}_{cond}')
                 for model in ['base','sft'] for cond in CONDITIONS]
    evaluation.BASELINES = [(label, f'reports/datav3/{stem}_inference.jsonl',
                            f'reports/datav3/{stem}_pkvc_eval.jsonl', None)
                           for label, stem in pairs]
    evaluation.METRICS = [m for m in evaluation.METRICS if m[0] in KEEP_METRICS]
    for _, inf, ev, _ in evaluation.BASELINES:
        if not Path(inf).is_file() or not Path(ev).is_file():
            raise FileNotFoundError((inf, ev))
    evaluation.main()

if __name__ == '__main__':
    main()
