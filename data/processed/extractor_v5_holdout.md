# Extractor v5 — Holdout F/G 测试 (只标 PKVC, 不抽 SADR)

- prompt: sft_pipeline/config/prompts/extractor_v5.md
- model: gpt-4o, temperature=0.2
- 评估: dominant_factor 一致性 + criterion grounded 检查

---
## 附录 F
- **专家结论**: C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)

### seg1 (第一反应)
- 原文 (220字): 关于这个。零售巨头希望在。老城区做一个全国性的首个全数字化无人化高效零售标杆，我觉得有趣且兴奋。它带有历史性的意义。关于三个矛盾啊，我想先说最后一个。第三点，呃，面对抗议和质疑啊，上层认为3个月内必须按照数字化方案铺开，这是公司战略战啊，不...
- LLM dominant_factor: **None** vs 专家 ['C', 'C→P'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### seg2 (决策链路)
- 原文 (356字): 这个问题对我来讲有一些陌生啊，数据商业要求与用户伦理，首先数据是一个客观的材料，那我在做这个项目的时候一定会去研究和然后商业要求是我可能在这个事件里面的一个本职工作，也是我会第一时间假设成立的一个内容，用户伦理可能是需要对社会文化国民的最终...
- LLM dominant_factor: **None** vs 专家 ['C', 'K', 'C→K'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### seg3 (核心 tradeoff)
- 原文 (232字): 最核心的设计底线或妥协 关注人类福祉。嗯。为人们提供幸福、自由和公平的。环境而努力。不能欺骗消费者。不能损害大家的身体健康。我想说的一个是老年人。嗯，有老年人的数字化方案，嗯，他应该也是可以呃有效的解决老年的这个。购物收银的最后这个环节的问...
- LLM dominant_factor: **None** vs 专家 ['V', 'V→C', 'C'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### F 评估

- HIT: 0/3, PARTIAL: 0/3, MISS: 3/3

---
## 附录 G
- **专家结论**: P (情境矛盾) → K (双重) → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)

### seg1 (第一反应)
- 原文 (314字): 首先我们是一家激进的零售巨头，希望打造"全国首个全数字化、无人化的高效零售标杆"。那么出现这个困境的最大问题是，区域选址在了老城区核心地段，这个场景就决定了必然对于一部分人来说传统的零售方式是他们更易于接受的。那么这样的情况上层如何说服我，...
- LLM dominant_factor: **None** vs 专家 ['P', 'P→K', 'P→V'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### seg2 (决策链路)
- 原文 (330字): 数据再怎么说都是客观冷静的反馈，不过设计存在的本质是为人们提供更好的体验，而非榨干所有利益价值。在我看来，完全听信数据推导出来的最有利于商业价值转化的方案，是有待商榷的。他根本上不是在为用户解决问题，那么长此以往用户在我们的设计中得不到关怀...
- LLM dominant_factor: **None** vs 专家 ['V', 'V→K', 'V→C'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### seg3 (核心 tradeoff)
- 原文 (386字): 核心设计底线总的来说是：我们应该结果导向，将思维回到原本的设计理念——"全国首个全数字化、无人化的高效零售标杆"。第一要做到高效，那么在用户采购的过程中应该为他们提供最清晰的选择，而不是将商品隐藏式排列；第二要做到零售标杆，那么一定是需要得...
- LLM dominant_factor: **None** vs 专家 ['C', 'C→V', 'V→C'] → MISS (null)
- LLM criterion: null
- LLM o_primary: None
- LLM p_surface: None
- LLM confidence: None
- LLM notes: 

### G 评估

- HIT: 0/3, PARTIAL: 0/3, MISS: 3/3

---
## 总评

| letter | HIT | PARTIAL | MISS |
|---|---|---|---|
| F | 0/3 | 0/3 | 3/3 |
| G | 0/3 | 0/3 | 3/3 |

**通过门槛**: 实际 234 段 Q7 大部分是单一 K/V/P 主导, 比 F/G 简单. 
若 holdout HIT+PARTIAL ≥ 4/6 → 可用于 Q7 全量; ≥ 2/6 → 用于 Q7 但需人工 sample 校.
≤ 1/6 → 不可用, prompt 继续 iterate.