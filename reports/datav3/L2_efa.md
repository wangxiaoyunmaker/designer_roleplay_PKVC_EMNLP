# L2 EFA Pilot (data_v3, 2026-05-24)

- N = 163, items = 30, N/item = 5.43  (临界可做)
- **KMO**: 0.603  (PASS≥0.6)
- **Bartlett χ²**: 962.9, p=5.89e-42  (PASS)
- **Parallel analysis 建议 K**: 4  (real eig > 95th pct random eig)

## 特征值 (前 10) vs random 95-pct

|   idx |   real_eig |   rand_95 |
|------:|-----------:|----------:|
|     1 |      3.917 |     2.054 |
|     2 |      2.144 |     1.866 |
|     3 |      2.072 |     1.756 |
|     4 |      1.686 |     1.648 |
|     5 |      1.51  |     1.555 |
|     6 |      1.431 |     1.493 |
|     7 |      1.363 |     1.437 |
|     8 |      1.273 |     1.358 |
|     9 |      1.187 |     1.31  |
|    10 |      1.114 |     1.248 |

## K = 3 factor solution

- cumulative variance: 19.5%
- Cronbach α per factor (|loading|>0.4): {'F1': 0.652, 'F2': 0.535, 'F3': 0.223}
- top 5 items per factor (by |loading|):
  - **F1**: ['复杂工具信仰', '资产开源', '风险规避', '封闭生态竞争', '时代 vs 永恒']
  - **F2**: ['完美主义/MVP', '创新代价', '混沌与灵感', '仿生与有机', '本土文化妥协']
  - **F3**: ['商业操纵伦理', '数据隐私交换', '系统微模块', 'AI 替代论', '数据 vs 直觉']

## K = 4 factor solution

- cumulative variance: 23.1%
- Cronbach α per factor (|loading|>0.4): {'F1': 0.652, 'F2': 0.509, 'F3': 0.495, 'F4': 0.454}
- top 5 items per factor (by |loading|):
  - **F1**: ['复杂工具信仰', '资产开源', '封闭生态竞争', '风险规避', '仿生与有机']
  - **F2**: ['完美主义/MVP', '创新代价', '本土文化妥协', '混沌与灵感', '仿生与有机']
  - **F3**: ['时代 vs 永恒', '商业操纵伦理', '技术支持', 'AI 替代论', '黑盒透明度']
  - **F4**: ['独裁 vs 委员会', '数据 vs 直觉', '团队依恋', '系统微模块', '装饰合法性']

## K = 5 factor solution

- cumulative variance: 27.2%
- Cronbach α per factor (|loading|>0.4): {'F1': 0.652, 'F2': 0.509, 'F3': 0.495, 'F4': None, 'F5': 0.454}
- top 5 items per factor (by |loading|):
  - **F1**: ['复杂工具信仰', '资产开源', '封闭生态竞争', '风险规避', '仿生与有机']
  - **F2**: ['完美主义/MVP', '创新代价', '本土文化妥协', '混沌与灵感', '仿生与有机']
  - **F3**: ['商业操纵伦理', '时代 vs 永恒', '技术支持', 'AI 替代论', '黑盒透明度']
  - **F4**: ['装饰合法性', '天赋信仰', '系统责任归属', '数据隐私交换', '情感附加值']
  - **F5**: ['数据 vs 直觉', '独裁 vs 委员会', '团队依恋', '股东服务优先', '系统微模块']

## K = 6 factor solution

- cumulative variance: 31.1%
- Cronbach α per factor (|loading|>0.4): {'F1': 0.652, 'F2': 0.509, 'F3': 0.495, 'F4': 0.454, 'F5': None, 'F6': None}
- top 5 items per factor (by |loading|):
  - **F1**: ['复杂工具信仰', '资产开源', '风险规避', '封闭生态竞争', '仿生与有机']
  - **F2**: ['完美主义/MVP', '创新代价', '本土文化妥协', '混沌与灵感', '仿生与有机']
  - **F3**: ['商业操纵伦理', '技术支持', '时代 vs 永恒', 'AI 替代论', '黑盒透明度']
  - **F4**: ['数据 vs 直觉', '独裁 vs 委员会', '团队依恋', '股东服务优先', '系统微模块']
  - **F5**: ['装饰合法性', '天赋信仰', '系统责任归属', '数据隐私交换', '情感附加值']
  - **F6**: ['创作者署名', '时代 vs 永恒', '系统责任归属', '秩序与防错', '弱势群体优先']

## Verdict

- **EFA-K 推荐**: K = 4 (parallel analysis)
- factor scores 已存 reports/datav3/per_designer/factor_scores_k{3,4,5,6}.csv (gitignored)
- 这只决定 **EFA factor count (L_cls head 维度)**, K-means 的 K 由 L3 独立挑