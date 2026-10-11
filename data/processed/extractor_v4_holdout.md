# Extractor v4 — Holdout F/G 测试结果

- prompt: sft_pipeline/config/prompts/extractor_v4.md
- model: gpt-4o, temperature=0.2
- holdout: 附录 F (C 先启 复杂模式), 附录 G (V 双形态)
- 评估角度: dominant_factor / SADR 命中率 / criterion 一致性

---
## 附录 F
- **专家结论**: C 先启 → P (重定义) → K (补充知识) → V (价值锚定) → P/C (再次收束)
- **专家 strong SADR**: #30+, #15+, #26+, #24-, #10+, #12+
- **专家 criterion**: 关注人类福祉, 为人们提供幸福、自由和公平

### seg1 (第一反应)
原文: 关于这个。零售巨头希望在。老城区做一个全国性的首个全数字化无人化高效零售标杆，我觉得有趣且兴奋。它带有历史性的意义。关于三个矛盾啊，我想先说最后一个。第三点，呃，面对抗议和质疑啊，上层认为3个月内必须按照数字化方案铺开，这是公司战略战啊，不能为了少部分人牺牲新贵的极致体验。我觉得不，我觉得这两个不矛盾，就是首先我会非常...

- LLM dominant_factor: **confidence=low**
- LLM p_surface: None
- LLM o_primary: None
- LLM confidence: None
- LLM criterion: None
- LLM SADR (0 条):(空)

### seg2 (决策链路)
原文: 这个问题对我来讲有一些陌生啊，数据商业要求与用户伦理，首先数据是一个客观的材料，那我在做这个项目的时候一定会去研究和然后商业要求是我可能在这个事件里面的一个本职工作，也是我会第一时间假设成立的一个内容，用户伦理可能是需要对社会文化国民的最终共性的价值观等都有一些了解后内化到自己的设计系统里的构建方案的时候，通常还是会先...

- LLM dominant_factor: **K**
- LLM p_surface: negotiates_around
- LLM o_primary: neutral
- LLM confidence: None
- LLM criterion: None
- LLM SADR (3 条):(空)

### seg3 (核心 tradeoff)
原文: 最核心的设计底线或妥协 关注人类福祉。嗯。为人们提供幸福、自由和公平的。环境而努力。不能欺骗消费者。不能损害大家的身体健康。我想说的一个是老年人。嗯，有老年人的数字化方案，嗯，他应该也是可以呃有效的解决老年的这个。购物收银的最后这个环节的问题的，包括整个体验应该是有专门针对老年人的这个。设计方案出现，嗯，然后 所以我诱...

- LLM dominant_factor: **K**
- LLM p_surface: negotiates_around
- LLM o_primary: neutral
- LLM confidence: None
- LLM criterion: None
- LLM SADR (3 条):(空)

### F 评估

- LLM dominant 序列: confidence=low → K → K
- LLM SADR 并集 (跨 3 段): []
- 命中专家 strong SADR: **0/6** = 0%
  - hit: []
  - miss: ['#30+', '#15+', '#26+', '#24-', '#10+', '#12+']
  - extra (LLM 多抽的, 可能 noise 也可能漏掉的真信号): []

---
## 附录 G
- **专家结论**: P (情境矛盾) → K (双重) → V₁ (上下文锚定) → C₁ (缓冲) → V₂ (宣言) → C₂ (综合)
- **专家 strong SADR**: #15+, #24-, #26+, #10-, #12+, #30+, #27-, #29+
- **专家 criterion**: 设计存在本质是为人们提供更好的体验, 给用户选择权

### seg1 (第一反应)
原文: 首先我们是一家激进的零售巨头，希望打造

- LLM dominant_factor: **confidence=low**
- LLM p_surface: None
- LLM o_primary: None
- LLM confidence: None
- LLM criterion: None
- LLM SADR (0 条):(空)

### seg2 (决策链路)
原文: 数据再怎么说都是客观冷静的反馈，不过设计存在的本质是为人们提供更好的体验，而非榨干所有利益价值。在我看来，完全听信数据推导出来的最有利于商业价值转化的方案，是有待商榷的。他根本上不是在为用户解决问题，那么长此以往用户在我们的设计中得不到关怀，是很有可能出现大量收益的流失的，那时候反而适得其反，我们不能尽着眼于当下的利益...

- LLM dominant_factor: **K**
- LLM p_surface: negotiates_around
- LLM o_primary: neutral
- LLM confidence: None
- LLM criterion: None
- LLM SADR (3 条):(空)

### seg3 (核心 tradeoff)
原文: 核心设计底线总的来说是：我们应该结果导向，将思维回到原本的设计理念——

- LLM dominant_factor: **K**
- LLM p_surface: negotiates_around
- LLM o_primary: neutral
- LLM confidence: None
- LLM criterion: None
- LLM SADR (3 条):(空)

### G 评估

- LLM dominant 序列: confidence=low → K → K
- LLM SADR 并集 (跨 3 段): []
- 命中专家 strong SADR: **0/8** = 0%
  - hit: []
  - miss: ['#15+', '#24-', '#26+', '#10-', '#12+', '#30+', '#27-', '#29+']
  - extra (LLM 多抽的, 可能 noise 也可能漏掉的真信号): []

---
## 总评

| letter | strong SADR hit | rate | dominant 序列 |
|---|---|---|---|
| F | 0/6 | 0% | confidence=low → K → K |
| G | 0/8 | 0% | confidence=low → K → K |
