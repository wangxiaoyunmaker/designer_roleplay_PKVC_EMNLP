# Extractor v6 — Holdout F/G 测试 (cleaned 输入 + 3 few-shot)

- prompt: sft_pipeline/config/prompts/extractor_v6.md
- input: docs/notes/被试回答专家标注_cleaned.md (cleaned 版)
- model: gpt-4o, temperature=0.0, no cache
- 评估: dominant_factor 一致性 + criterion grounded

---
## 附录 F
- **专家结论**: C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)

### seg1 (第一反应)
- INPUT (223字): 关于这个项目, 零售巨头希望在老城区做一个全国首个全数字化、无人化的高效零售标杆, 我觉得有趣且兴奋, 它带有历史性的意义. 关于三个矛盾, 我想先说最后一个——第三点: 面对抗议和质疑, 上层认为 ...
- LLM dominant_factor: **P** vs 专家 ['C', 'C→P', 'P'] → HIT (P)
- LLM criterion: NOT_IN_INPUT: 'placeholder'
- LLM o_primary: P
- LLM p_surface: forces_transition_to
- LLM confidence: None
- LLM notes: 

### seg2 (决策链路)
- INPUT (348字): 这个问题对我来讲有一些陌生. 数据、商业要求与用户伦理三者中, 首先数据是一个客观的材料, 我在做这个项目的时候一定会去研究; 商业要求是我在这个项目里的本职工作, 也是我会第一时间假设成立的内容; ...
- LLM dominant_factor: **K→V** vs 专家 ['C', 'K', 'C→K', 'K→V'] → HIT (K→V)
- LLM criterion: NOT_IN_INPUT: '需要调研老城区居民的购物习惯'
- LLM o_primary: V
- LLM p_surface: forces_transition_to
- LLM confidence: None
- LLM notes: 

### seg3 (核心 tradeoff)
- INPUT (222字): 最核心的设计底线或妥协是关注人类福祉, 为人们提供幸福、自由和公平的环境而努力. 不能欺骗消费者, 不能损害大家的身体健康. 我想说的一个是老年人——针对老年人的数字化方案, 应该是能够有效解决老年人...
- LLM dominant_factor: **K→V** vs 专家 ['V', 'V→C', 'C', 'V→P'] → PARTIAL (K→V 路径终点 = V)
- LLM criterion: NOT_IN_INPUT: '需要调研老城区居民的购物习惯'
- LLM o_primary: V
- LLM p_surface: negotiates_around
- LLM confidence: None
- LLM notes: 

### F 评估
- HIT: 2/3, PARTIAL: 1/3, MISS: 0/3

---
## 附录 G
- **专家结论**: P (情境矛盾) → K → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)

### seg1 (第一反应)
- INPUT (335字): 首先我们是一家激进的零售巨头, 希望打造"全国首个全数字化、无人化的高效零售标杆". 那么出现这个困境的最大问题是, 区域选址在了老城区核心地段, 这个场景就决定了必然对于一部分人来说传统的零售方式是...
- LLM dominant_factor: **K** vs 专家 ['P', 'P→K', 'P→V', 'P→C'] → PARTIAL (K = P→K 路径终点)
- LLM criterion: NOT_IN_INPUT: '需要调研老城区居民的购物习惯'
- LLM o_primary: K
- LLM p_surface: forces_transition_to
- LLM confidence: None
- LLM notes: 

### seg2 (决策链路)
- INPUT (348字): 数据再怎么说都是客观冷静的反馈, 不过设计存在的本质是为人们提供更好的体验, 而非榨干所有利益价值. 在我看来, 完全听信数据推导出来的最有利于商业价值转化的方案是有待商榷的——它根本上不是在为用户解...
- LLM dominant_factor: **P** vs 专家 ['V', 'V→K', 'V→C', 'K→V'] → MISS (P; expected ['V', 'V→K', 'V→C', 'K→V'])
- LLM criterion: NOT_IN_INPUT: 'placeholder'
- LLM o_primary: P
- LLM p_surface: forces_transition_to
- LLM confidence: None
- LLM notes: 

### seg3 (核心 tradeoff)
- INPUT (419字): 核心设计底线总的来说是: 我们应该结果导向, 将思维回到原本的设计理念——"全国首个全数字化、无人化的高效零售标杆". 第一要做到高效, 那么在用户采购的过程中应该为他们提供最清晰的选择, 而不是将商...
- LLM dominant_factor: **P** vs 专家 ['C', 'C→V', 'V→C', 'V'] → MISS (P; expected ['C', 'C→V', 'V→C', 'V'])
- LLM criterion: NOT_IN_INPUT: 'placeholder'
- LLM o_primary: P
- LLM p_surface: forces_transition_to
- LLM confidence: None
- LLM notes: 

### G 评估
- HIT: 0/3, PARTIAL: 1/3, MISS: 2/3

---
## 总评

| letter | HIT | PARTIAL | MISS |
|---|---|---|---|
| F | 2/3 | 1/3 | 0/3 |
| G | 0/3 | 1/3 | 2/3 |

**通过门槛**: HIT+PARTIAL ≥ 4/6 = 可用; ≥ 2/6 = 用但 sample 人工校;
≤ 1/6 = prompt 还要 iterate.