"""Q1-Q6 explanation NL expansion pipeline.

Designer 在 Q1-Q6 选完之后写的 text_supplement 经常只有 1-10 字 (e.g. "直觉"),
直接做 SFT target 信号太弱. 本 subpackage 实现 model-assisted grounded rewrite:
对每条短解释, 用 GPT-4o 在 *该 designer 自己的全部材料* (Q7 长 NL + 量表 polarity
+ cardId 排序 + Q8 self-report + 选项原文 + 短 NL 本身) 约束下扩到 50-100 字.

Pipeline 5 步:
  1. screen.py  - 按字符阈值筛选需扩条目 + 装配 ground 材料
  2. prompt.py  - 渲染 system + user prompt
  3. batch.py   - 异步 LLM 批跑 + cache
  4. parser.py  - 出参 schema 校验 + 反 hallucination 检测
  5. merge.py   - 回填 task_responses_expanded.csv (prep.py 可直接消费)

CLI 入口: ``python -m sft_pipeline.expand.cli <subcommand>``
"""
from sft_pipeline.expand.screen import (
    ExpansionCandidate, screen_candidates, write_candidates, load_candidates,
)

__all__ = [
    "ExpansionCandidate",
    "screen_candidates",
    "write_candidates",
    "load_candidates",
]

# parser / batch / merge 在各自模块写完后追加到 __all__
