# L4c — 30 raw items → Q-option LASSO (data_v3, 2026-05-24)

回应 quad pipeline Step 2a 的精确需求: 30 维原始量表 (S1..S30) → 每个 (Q, option) top-2-3 predictor item.

- predictors: 30 raw 量表题 (S{1..30}), source = scale_responses (per-designer wide)
- model: multinomial L1 logistic (C=0.5), per option top-3 |coef|≥0.05
- per-item p-value: marginal permutation (500×), 非 conditional
- LASSO 系数已 standardize 过 (StandardScaler on items), 系数大小可跨 item 比
- 元数据: item 极性 (from docs/notes/PKVC_30节点本体.md, 30 items 已 load)
- 元数据: Q-option PKVC 因子 (from docs/notes/Q1-Q6每个选项的元数据.md, 20 options 已 load)


## Q1 (n=163, options=['A', 'B', 'C', 'D'])


### Q1.A (n_chose=66/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 商业操纵伦理 | +0.263 | higher | 0.228 |
| 资产开源 | +0.261 | higher | 0.287 |
| 数据隐私交换 | +0.235 | higher | 0.096 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.263, marginal p=0.228) and higher scores on "资产开源" (= 开源道德派 stance; β=+0.261, marginal p=0.287) and higher scores on "数据隐私交换" (= 体验换隐私 stance; β=+0.235, marginal p=0.096) have a higher probability of selecting option A (activates PKVC factor: C) (n_chose=66/163).

### Q1.B (n_chose=60/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 系统责任归属 | -0.380 | lower | 0.012 |
| 时代 vs 永恒 | +0.325 | higher | 0.060 |
| 商业操纵伦理 | +0.296 | higher | 0.102 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统责任归属" (= 用户归责派 stance; β=-0.380, marginal p=0.012) and higher scores on "时代 vs 永恒" (= 时代潮流派 stance; β=+0.325, marginal p=0.060) and higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.296, marginal p=0.102) have a higher probability of selecting option B (activates PKVC factor: P) (n_chose=60/163).

### Q1.C (n_chose=23/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 仿生与有机 | -0.229 | lower | 0.259 |
| 系统微模块 | +0.220 | higher | 0.387 |
| 混沌与灵感 | +0.185 | higher | 0.377 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "仿生与有机" (= 几何工业派 stance; β=-0.229, marginal p=0.259) and higher scores on "系统微模块" (= 解耦优先 stance; β=+0.220, marginal p=0.387) and higher scores on "混沌与灵感" (= 包容失控派 stance; β=+0.185, marginal p=0.377) have a higher probability of selecting option C (activates PKVC factor: K) (n_chose=23/163).

### Q1.D (n_chose=14/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 数据 vs 直觉 | +0.537 | higher | 0.040 |
| 团队依恋 | -0.465 | lower | 0.108 |
| 封闭生态竞争 | -0.445 | lower | 0.072 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.537, marginal p=0.040) and lower scores on "团队依恋" (= 团队共创派 stance; β=-0.465, marginal p=0.108) and lower scores on "封闭生态竞争" (= 开放兼容派 stance; β=-0.445, marginal p=0.072) have a higher probability of selecting option D (activates PKVC factor: V) (n_chose=14/163).

## Q3 (n=163, options=['A', 'B', 'C', 'D'])


### Q3.A (n_chose=11/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 天赋信仰 | -0.349 | lower | 0.054 |
| 混沌与灵感 | +0.275 | higher | 0.273 |
| 团队依恋 | +0.247 | higher | 0.230 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "天赋信仰" (= 后天努力论 stance; β=-0.349, marginal p=0.054) and higher scores on "混沌与灵感" (= 包容失控派 stance; β=+0.275, marginal p=0.273) and higher scores on "团队依恋" (= 独立打磨派 stance; β=+0.247, marginal p=0.230) have a higher probability of selecting option A (activates PKVC factor: K, chain=k2) (n_chose=11/163).

### Q3.B (n_chose=60/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 完美主义/MVP | +0.164 | higher | 0.519 |
| 弱势群体优先 | +0.147 | higher | 0.345 |
| 创新代价 | -0.139 | lower | 0.375 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "完美主义/MVP" (= 完美主义 stance; β=+0.164, marginal p=0.519) and higher scores on "弱势群体优先" (= 边缘群体优先 stance; β=+0.147, marginal p=0.345) and lower scores on "创新代价" (= 惯例守护派 stance; β=-0.139, marginal p=0.375) have a higher probability of selecting option B (activates PKVC factor: P, chain=p3➡️k3) (n_chose=60/163).

### Q3.C (n_chose=8/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 仿生与有机 | -0.861 | lower | 0.028 |
| 数据 vs 直觉 | +0.692 | higher | 0.054 |
| AI 替代论 | -0.435 | lower | 0.080 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "仿生与有机" (= 几何工业派 stance; β=-0.861, marginal p=0.028) and higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.692, marginal p=0.054) and lower scores on "AI 替代论" (= 岗位保护主义 stance; β=-0.435, marginal p=0.080) have a higher probability of selecting option C (activates PKVC factor: V, chain=v1➡️c1) (n_chose=8/163).

### Q3.D (n_chose=84/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 救世主情结 | +0.362 | higher | 0.016 |
| 审美反民粹 | -0.288 | lower | 0.020 |
| 仿生与有机 | +0.179 | higher | 0.319 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "救世主情结" (= 救世主情结 stance; β=+0.362, marginal p=0.016) and lower scores on "审美反民粹" (= 数据民粹妥协 stance; β=-0.288, marginal p=0.020) and higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.179, marginal p=0.319) have a higher probability of selecting option D (activates PKVC factor: K, chain=k2) (n_chose=84/163).

## Q4 (n=163, options=['A', 'B', 'C', 'D'])


### Q4.A (n_chose=78/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 本土文化妥协 | -0.341 | lower | 0.046 |
| 股东服务优先 | +0.209 | higher | 0.078 |
| 天赋信仰 | -0.161 | lower | 0.335 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.341, marginal p=0.046) and higher scores on "股东服务优先" (= 股东忠诚派 stance; β=+0.209, marginal p=0.078) and lower scores on "天赋信仰" (= 后天努力论 stance; β=-0.161, marginal p=0.335) have a higher probability of selecting option A (activates PKVC factor: K, chain=k1➡️p1) (n_chose=78/163).

### Q4.B (n_chose=8/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 股东服务优先 | -0.337 | lower | 0.036 |
| 仿生与有机 | +0.330 | higher | 0.168 |
| 秩序与防错 | -0.290 | lower | 0.088 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "股东服务优先" (= 用户忠诚派 stance; β=-0.337, marginal p=0.036) and higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.330, marginal p=0.168) and lower scores on "秩序与防错" (= 探索开放派 stance; β=-0.290, marginal p=0.088) have a higher probability of selecting option B (activates PKVC factor: V, chain=v3) (n_chose=8/163).

### Q4.C (n_chose=29/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 资产开源 | +0.759 | higher | 0.002 |
| 黑盒透明度 | -0.364 | lower | 0.130 |
| 风险规避 | -0.345 | lower | 0.467 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "资产开源" (= 开源道德派 stance; β=+0.759, marginal p=0.002) and lower scores on "黑盒透明度" (= 过程透明派 stance; β=-0.364, marginal p=0.130) and lower scores on "风险规避" (= 风险偏好 stance; β=-0.345, marginal p=0.467) have a higher probability of selecting option C (activates PKVC factor: C, chain=c2) (n_chose=29/163).

### Q4.D (n_chose=48/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 审美反民粹 | -0.360 | lower | 0.054 |
| 数据 vs 直觉 | +0.300 | higher | 0.539 |
| 创作者署名 | -0.282 | lower | 0.036 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "审美反民粹" (= 数据民粹妥协 stance; β=-0.360, marginal p=0.054) and higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.300, marginal p=0.539) and lower scores on "创作者署名" (= 商业去人格化 stance; β=-0.282, marginal p=0.036) have a higher probability of selecting option D (activates PKVC factor: K, chain=k2➡️p2) (n_chose=48/163).

## Q5 (n=163, options=['A', 'B', 'C', 'D'])


### Q5.A (n_chose=28/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 数据 vs 直觉 | -0.516 | lower | 0.006 |
| 技术支持 | +0.417 | higher | 0.004 |
| 封闭生态竞争 | +0.412 | higher | 0.022 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "数据 vs 直觉" (= 重数据立场 stance; β=-0.516, marginal p=0.006) and higher scores on "技术支持" (= 技术激进主义 stance; β=+0.417, marginal p=0.004) and higher scores on "封闭生态竞争" (= 封闭生态派 stance; β=+0.412, marginal p=0.022) have a higher probability of selecting option A (activates PKVC factor: K, chain=k2) (n_chose=28/163).

### Q5.B (n_chose=36/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 系统微模块 | -0.418 | lower | 0.002 |
| 技术支持 | -0.405 | lower | 0.002 |
| 完美主义/MVP | -0.355 | lower | 0.106 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统微模块" (= 整体感优先 stance; β=-0.418, marginal p=0.002) and lower scores on "技术支持" (= 技术审慎主义 stance; β=-0.405, marginal p=0.002) and lower scores on "完美主义/MVP" (= 速度优先 stance; β=-0.355, marginal p=0.106) have a higher probability of selecting option B (activates PKVC factor: C, chain=c2) (n_chose=36/163).

### Q5.C (n_chose=20/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 商业操纵伦理 | -0.461 | lower | 0.004 |
| 技术支持 | -0.280 | lower | 0.062 |
| 黑盒透明度 | -0.279 | lower | 0.040 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "商业操纵伦理" (= 用户伦理底线 stance; β=-0.461, marginal p=0.004) and lower scores on "技术支持" (= 技术审慎主义 stance; β=-0.280, marginal p=0.062) and lower scores on "黑盒透明度" (= 过程透明派 stance; β=-0.279, marginal p=0.040) have a higher probability of selecting option C (activates PKVC factor: V, chain=v3) (n_chose=20/163).

### Q5.D (n_chose=79/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 本土文化妥协 | -0.441 | lower | 0.018 |
| 混沌与灵感 | -0.291 | lower | 0.018 |
| 独裁 vs 委员会 | +0.288 | higher | 0.098 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.441, marginal p=0.018) and lower scores on "混沌与灵感" (= 验证至上派 stance; β=-0.291, marginal p=0.018) and higher scores on "独裁 vs 委员会" (= 独裁愿景派 stance; β=+0.288, marginal p=0.098) have a higher probability of selecting option D (activates PKVC factor: P, chain=p1) (n_chose=79/163).

## Q6 (n=159, options=['A', 'B', 'C'])


### Q6.A (n_chose=21/159)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 仿生与有机 | +0.437 | higher | 0.014 |
| 本土文化妥协 | -0.293 | lower | 0.044 |
| 系统责任归属 | +0.261 | higher | 0.006 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.437, marginal p=0.014) and lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.293, marginal p=0.044) and higher scores on "系统责任归属" (= 系统归责派 stance; β=+0.261, marginal p=0.006) have a higher probability of selecting option A (activates PKVC factor: V) (n_chose=21/159).

### Q6.B (n_chose=61/159)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 本土文化妥协 | +0.539 | higher | 0.004 |
| AI 替代论 | -0.510 | lower | 0.126 |
| 商业操纵伦理 | +0.427 | higher | 0.048 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "本土文化妥协" (= 本土深耕派 stance; β=+0.539, marginal p=0.004) and lower scores on "AI 替代论" (= 岗位保护主义 stance; β=-0.510, marginal p=0.126) and higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.427, marginal p=0.048) have a higher probability of selecting option B (activates PKVC factor: V, chain=v1➡️c2) (n_chose=61/159).

### Q6.C (n_chose=77/159)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 封闭生态竞争 | +0.471 | higher | 0.387 |
| 系统责任归属 | -0.416 | lower | 0.030 |
| 救世主情结 | -0.280 | lower | 0.042 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "封闭生态竞争" (= 封闭生态派 stance; β=+0.471, marginal p=0.387) and lower scores on "系统责任归属" (= 用户归责派 stance; β=-0.416, marginal p=0.030) and lower scores on "救世主情结" (= 工具中立派 stance; β=-0.280, marginal p=0.042) have a higher probability of selecting option C (activates PKVC factor: V, chain=v2➡️p1) (n_chose=77/159).

## Q8 (n=163, options=['A', 'B', 'C'])


### Q8.A (n_chose=47/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 创新代价 | -0.447 | lower | 0.004 |
| 弱势群体优先 | -0.139 | lower | 0.285 |
| 资产开源 | -0.111 | lower | 0.291 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "创新代价" (= 惯例守护派 stance; β=-0.447, marginal p=0.004) and lower scores on "弱势群体优先" (= 多数效率优先 stance; β=-0.139, marginal p=0.285) and lower scores on "资产开源" (= 竞争壁垒派 stance; β=-0.111, marginal p=0.291) have a higher probability of selecting option A (n_chose=47/163).

### Q8.B (n_chose=61/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 系统微模块 | -0.253 | lower | 0.016 |
| 复杂工具信仰 | -0.210 | lower | 0.086 |
| 资产开源 | +0.133 | higher | 0.852 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统微模块" (= 整体感优先 stance; β=-0.253, marginal p=0.016) and lower scores on "复杂工具信仰" (= 工具无关论 stance; β=-0.210, marginal p=0.086) and higher scores on "资产开源" (= 开源道德派 stance; β=+0.133, marginal p=0.852) have a higher probability of selecting option B (n_chose=61/163).

### Q8.C (n_chose=55/163)

**Top items:**

| item | β (standardized LASSO coef) | direction | marginal perm-p |
|---|---|---|---|
| 股东服务优先 | +0.332 | higher | 0.054 |
| 系统微模块 | +0.261 | higher | 0.002 |
| 装饰合法性 | +0.256 | higher | 0.082 |

**Statistical Observation (English, for paper narrative / CPO Compiler input):**

> In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "股东服务优先" (= 股东忠诚派 stance; β=+0.332, marginal p=0.054) and higher scores on "系统微模块" (= 解耦优先 stance; β=+0.261, marginal p=0.002) and higher scores on "装饰合法性" (= 装饰合法派 stance; β=+0.256, marginal p=0.082) have a higher probability of selecting option C (n_chose=55/163).

---

## 全部 Statistical Observation 段落 (for direct CPO Compiler input)

1. In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.263, marginal p=0.228) and higher scores on "资产开源" (= 开源道德派 stance; β=+0.261, marginal p=0.287) and higher scores on "数据隐私交换" (= 体验换隐私 stance; β=+0.235, marginal p=0.096) have a higher probability of selecting option A (activates PKVC factor: C) (n_chose=66/163).

2. In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统责任归属" (= 用户归责派 stance; β=-0.380, marginal p=0.012) and higher scores on "时代 vs 永恒" (= 时代潮流派 stance; β=+0.325, marginal p=0.060) and higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.296, marginal p=0.102) have a higher probability of selecting option B (activates PKVC factor: P) (n_chose=60/163).

3. In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "仿生与有机" (= 几何工业派 stance; β=-0.229, marginal p=0.259) and higher scores on "系统微模块" (= 解耦优先 stance; β=+0.220, marginal p=0.387) and higher scores on "混沌与灵感" (= 包容失控派 stance; β=+0.185, marginal p=0.377) have a higher probability of selecting option C (activates PKVC factor: K) (n_chose=23/163).

4. In question Q1, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.537, marginal p=0.040) and lower scores on "团队依恋" (= 团队共创派 stance; β=-0.465, marginal p=0.108) and lower scores on "封闭生态竞争" (= 开放兼容派 stance; β=-0.445, marginal p=0.072) have a higher probability of selecting option D (activates PKVC factor: V) (n_chose=14/163).

5. In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "天赋信仰" (= 后天努力论 stance; β=-0.349, marginal p=0.054) and higher scores on "混沌与灵感" (= 包容失控派 stance; β=+0.275, marginal p=0.273) and higher scores on "团队依恋" (= 独立打磨派 stance; β=+0.247, marginal p=0.230) have a higher probability of selecting option A (activates PKVC factor: K, chain=k2) (n_chose=11/163).

6. In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "完美主义/MVP" (= 完美主义 stance; β=+0.164, marginal p=0.519) and higher scores on "弱势群体优先" (= 边缘群体优先 stance; β=+0.147, marginal p=0.345) and lower scores on "创新代价" (= 惯例守护派 stance; β=-0.139, marginal p=0.375) have a higher probability of selecting option B (activates PKVC factor: P, chain=p3➡️k3) (n_chose=60/163).

7. In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "仿生与有机" (= 几何工业派 stance; β=-0.861, marginal p=0.028) and higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.692, marginal p=0.054) and lower scores on "AI 替代论" (= 岗位保护主义 stance; β=-0.435, marginal p=0.080) have a higher probability of selecting option C (activates PKVC factor: V, chain=v1➡️c1) (n_chose=8/163).

8. In question Q3, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "救世主情结" (= 救世主情结 stance; β=+0.362, marginal p=0.016) and lower scores on "审美反民粹" (= 数据民粹妥协 stance; β=-0.288, marginal p=0.020) and higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.179, marginal p=0.319) have a higher probability of selecting option D (activates PKVC factor: K, chain=k2) (n_chose=84/163).

9. In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.341, marginal p=0.046) and higher scores on "股东服务优先" (= 股东忠诚派 stance; β=+0.209, marginal p=0.078) and lower scores on "天赋信仰" (= 后天努力论 stance; β=-0.161, marginal p=0.335) have a higher probability of selecting option A (activates PKVC factor: K, chain=k1➡️p1) (n_chose=78/163).

10. In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "股东服务优先" (= 用户忠诚派 stance; β=-0.337, marginal p=0.036) and higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.330, marginal p=0.168) and lower scores on "秩序与防错" (= 探索开放派 stance; β=-0.290, marginal p=0.088) have a higher probability of selecting option B (activates PKVC factor: V, chain=v3) (n_chose=8/163).

11. In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "资产开源" (= 开源道德派 stance; β=+0.759, marginal p=0.002) and lower scores on "黑盒透明度" (= 过程透明派 stance; β=-0.364, marginal p=0.130) and lower scores on "风险规避" (= 风险偏好 stance; β=-0.345, marginal p=0.467) have a higher probability of selecting option C (activates PKVC factor: C, chain=c2) (n_chose=29/163).

12. In question Q4, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "审美反民粹" (= 数据民粹妥协 stance; β=-0.360, marginal p=0.054) and higher scores on "数据 vs 直觉" (= 重直觉立场 stance; β=+0.300, marginal p=0.539) and lower scores on "创作者署名" (= 商业去人格化 stance; β=-0.282, marginal p=0.036) have a higher probability of selecting option D (activates PKVC factor: K, chain=k2➡️p2) (n_chose=48/163).

13. In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "数据 vs 直觉" (= 重数据立场 stance; β=-0.516, marginal p=0.006) and higher scores on "技术支持" (= 技术激进主义 stance; β=+0.417, marginal p=0.004) and higher scores on "封闭生态竞争" (= 封闭生态派 stance; β=+0.412, marginal p=0.022) have a higher probability of selecting option A (activates PKVC factor: K, chain=k2) (n_chose=28/163).

14. In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统微模块" (= 整体感优先 stance; β=-0.418, marginal p=0.002) and lower scores on "技术支持" (= 技术审慎主义 stance; β=-0.405, marginal p=0.002) and lower scores on "完美主义/MVP" (= 速度优先 stance; β=-0.355, marginal p=0.106) have a higher probability of selecting option B (activates PKVC factor: C, chain=c2) (n_chose=36/163).

15. In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "商业操纵伦理" (= 用户伦理底线 stance; β=-0.461, marginal p=0.004) and lower scores on "技术支持" (= 技术审慎主义 stance; β=-0.280, marginal p=0.062) and lower scores on "黑盒透明度" (= 过程透明派 stance; β=-0.279, marginal p=0.040) have a higher probability of selecting option C (activates PKVC factor: V, chain=v3) (n_chose=20/163).

16. In question Q5, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.441, marginal p=0.018) and lower scores on "混沌与灵感" (= 验证至上派 stance; β=-0.291, marginal p=0.018) and higher scores on "独裁 vs 委员会" (= 独裁愿景派 stance; β=+0.288, marginal p=0.098) have a higher probability of selecting option D (activates PKVC factor: P, chain=p1) (n_chose=79/163).

17. In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "仿生与有机" (= 有机自然派 stance; β=+0.437, marginal p=0.014) and lower scores on "本土文化妥协" (= 全球通用派 stance; β=-0.293, marginal p=0.044) and higher scores on "系统责任归属" (= 系统归责派 stance; β=+0.261, marginal p=0.006) have a higher probability of selecting option A (activates PKVC factor: V) (n_chose=21/159).

18. In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "本土文化妥协" (= 本土深耕派 stance; β=+0.539, marginal p=0.004) and lower scores on "AI 替代论" (= 岗位保护主义 stance; β=-0.510, marginal p=0.126) and higher scores on "商业操纵伦理" (= 商业操纵容忍 stance; β=+0.427, marginal p=0.048) have a higher probability of selecting option B (activates PKVC factor: V, chain=v1➡️c2) (n_chose=61/159).

19. In question Q6, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "封闭生态竞争" (= 封闭生态派 stance; β=+0.471, marginal p=0.387) and lower scores on "系统责任归属" (= 用户归责派 stance; β=-0.416, marginal p=0.030) and lower scores on "救世主情结" (= 工具中立派 stance; β=-0.280, marginal p=0.042) have a higher probability of selecting option C (activates PKVC factor: V, chain=v2➡️p1) (n_chose=77/159).

20. In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "创新代价" (= 惯例守护派 stance; β=-0.447, marginal p=0.004) and lower scores on "弱势群体优先" (= 多数效率优先 stance; β=-0.139, marginal p=0.285) and lower scores on "资产开源" (= 竞争壁垒派 stance; β=-0.111, marginal p=0.291) have a higher probability of selecting option A (n_chose=47/163).

21. In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with lower scores on "系统微模块" (= 整体感优先 stance; β=-0.253, marginal p=0.016) and lower scores on "复杂工具信仰" (= 工具无关论 stance; β=-0.210, marginal p=0.086) and higher scores on "资产开源" (= 开源道德派 stance; β=+0.133, marginal p=0.852) have a higher probability of selecting option B (n_chose=61/163).

22. In question Q8, LASSO multinomial logistic regression on 30 raw trait items shows that designers with higher scores on "股东服务优先" (= 股东忠诚派 stance; β=+0.332, marginal p=0.054) and higher scores on "系统微模块" (= 解耦优先 stance; β=+0.261, marginal p=0.002) and higher scores on "装饰合法性" (= 装饰合法派 stance; β=+0.256, marginal p=0.082) have a higher probability of selecting option C (n_chose=55/163).
