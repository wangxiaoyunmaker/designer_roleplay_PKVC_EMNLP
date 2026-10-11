# Extractor token budget (model: `gpt-4o`)

## Q7 segments

- Records: **234**
- Char length: min=93, median=267, max=1175, mean=295
- Estimated input tokens (incl. 1500 prompt overhead/call): **454,892**
- Estimated output tokens (300/record): **70,200**
- **Cost: $1.84** (in $1.14 + out $0.70)

## Q1-Q6 explanations

- Records: **390**
- Char length: min=2, median=42, max=366, mean=61
- Estimated input tokens (incl. 1500 prompt overhead/call): **621,178**
- Estimated output tokens (300/record): **117,000**
- **Cost: $2.72** (in $1.55 + out $1.17)

## κ pilot (30 Q7 segments × 2 models)

- Per model: ~$0.23; both: **~$0.47**

> 假设: prompt overhead per call = 1500 tokens (system + few-shot); output per call ≈ 300 tokens (JSON). 中文 token 估算用 1.5 chars/token 上界, 实际可能略低.