"""Baseline 实现 (B0 / B0' / B1 / B2 / B3 / B4).

B3 K近邻 + B4 GBM 在本地直接跑 (无大模型, 无 SFT).
B0 / B0' / B1 / B2 需要调 LLM API 或加载 SFT checkpoint.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class BaselinePrediction:
    user_id: str
    task_id: str                        # Q1..Q6 / B1..B3 / Q7
    answer: str | None
    method: str                         # baseline 标识


# ====================================
# B3 K近邻同簇投票 (trivial baseline, 最可能反咬)
# ====================================

def baseline_b3_knn(
    train_profile: pd.DataFrame,        # rows: user_id, cols: 30 量表 numerical
    train_answers: pd.DataFrame,        # cols: user_id, task_id, answer
    train_cluster: dict[str, int],
    test_users: list[str],
    test_profile: pd.DataFrame,
    test_cluster: dict[str, int],
    k: int = 3,
) -> list[BaselinePrediction]:
    """同簇 top-k 邻居 majority vote, 跨题独立预测.

    P2-7 fix: 用 intersection 防御 - 如果 train_answers 里有 user_id 不在
    train_profile 索引中, 旧版会抛 KeyError. 这里取 set intersection 然后
    再筛 train_profile.
    """
    answer_uids = set(train_answers["user_id"].astype(str).unique())
    avail_uids = answer_uids & set(train_profile.index.astype(str))
    if not avail_uids:
        return []
    train_profile_aligned = train_profile.loc[sorted(avail_uids)]
    train_users_list = train_profile_aligned.index.tolist()
    scaler = StandardScaler().fit(train_profile_aligned.values)
    train_X = scaler.transform(train_profile_aligned.values)

    answers_by_user = {(r.user_id, r.task_id): r.answer for r in train_answers.itertuples()}
    task_ids = train_answers["task_id"].unique().tolist()

    preds = []
    for uid in test_users:
        if uid not in test_profile.index:
            continue
        target_cluster = test_cluster.get(uid)
        if target_cluster is None:
            continue
        # 同簇 train user
        same_cluster_users = [u for u in train_users_list if train_cluster.get(u) == target_cluster and u != uid]
        if not same_cluster_users:
            continue
        same_X = train_X[[train_users_list.index(u) for u in same_cluster_users]]
        target_X = scaler.transform(test_profile.loc[[uid]].values)
        nn = NearestNeighbors(n_neighbors=min(k, len(same_cluster_users))).fit(same_X)
        _, idx = nn.kneighbors(target_X)
        neighbor_users = [same_cluster_users[i] for i in idx[0]]

        for task_id in task_ids:
            neighbor_ans = [answers_by_user.get((u, task_id)) for u in neighbor_users]
            neighbor_ans = [a for a in neighbor_ans if a is not None]
            if not neighbor_ans:
                continue
            majority = Counter(neighbor_ans).most_common(1)[0][0]
            preds.append(BaselinePrediction(uid, task_id, majority, "B3_knn_majority"))
    return preds


# ====================================
# B4 GBM on profile_30d
# ====================================

def baseline_b4_gbm(
    train_profile: pd.DataFrame,
    train_answers: pd.DataFrame,
    test_users: list[str],
    test_profile: pd.DataFrame,
) -> list[BaselinePrediction]:
    """每个 task 单独训一个 GBM classifier, 用 profile_30d 预测 answer."""
    task_ids = train_answers["task_id"].unique().tolist()
    preds = []

    for task_id in task_ids:
        task_data = train_answers[train_answers["task_id"] == task_id]
        if len(task_data) < 5:
            continue
        X = train_profile.loc[task_data["user_id"]].values
        y = task_data["answer"].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        if len(set(y_enc)) < 2:
            continue

        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X, y_enc)

        for uid in test_users:
            if uid not in test_profile.index:
                continue
            pred_enc = clf.predict(test_profile.loc[[uid]].values)[0]
            pred = le.inverse_transform([pred_enc])[0]
            preds.append(BaselinePrediction(uid, task_id, str(pred), "B4_gbm"))
    return preds


# ====================================
# B0 / B0' / B1 / B2 (prompt or SFT 版)
# ====================================

def baseline_b0_random_uniform(
    test_users: list[str],
    train_answers: pd.DataFrame,
    seed: int = 42,
) -> list[BaselinePrediction]:
    """B0 离线 baseline: per-task uniform random over observed option pool.

    用途: chance-level 锚点. paper 用真 B0 (LLM prompt-only) 替换; 这里保
    offline-runnable 让 smoke test / Tier 1 pipeline 不挂.

    Note: 不需要 profile / model, 只靠每题的 option 分布 (从 train_answers 抽).
    """
    rng = np.random.default_rng(seed)
    task_options: dict[str, list[str]] = {}
    for tid, grp in train_answers.groupby("task_id"):
        opts = sorted(set(grp["answer"].astype(str).dropna()))
        if opts:
            task_options[str(tid)] = opts
    preds: list[BaselinePrediction] = []
    for uid in test_users:
        for tid, opts in task_options.items():
            preds.append(BaselinePrediction(
                user_id=str(uid), task_id=str(tid),
                answer=str(rng.choice(opts)), method="B0_random_uniform",
            ))
    return preds


def baseline_b0_prompt_only(
    test_users: list[str],
    train_answers: pd.DataFrame,
    test_profile: pd.DataFrame | None = None,
    model: str = "Qwen2.5-7B-Instruct",
    seed: int = 42,
    api_key: str | None = None,
    api_base_url: str | None = None,
    fallback_random: bool = True,
) -> list[BaselinePrediction]:
    """B0 基础大模型 prompt-only (无 RAG / 无 PKVC schema / 无 SFT).

    实际跑法: 调 vllm 本地或 OpenAI-兼容 API (yunwu.ai 中转).
    没 API key 且 fallback_random=True → 退化为 ``baseline_b0_random_uniform``,
    给个 chance-level 锚点保 Tier 1 不缺指标.

    Note: 真正 paper 用要把题面 + 5 选项 + designer profile 拼 prompt 喂模型,
    那块是 prep.py 的工作; 这里只负责 baseline 调用接口.
    """
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        if fallback_random:
            return baseline_b0_random_uniform(test_users, train_answers, seed=seed)
        raise NotImplementedError("B0 needs API key; set OPENAI_API_KEY or pass api_key=...")
    # 实跑路径预留: 注入 OpenAI-兼容 client; 推理后 parse single letter.
    # 投稿前接 sft_pipeline.infer 或 vllm/openai client; 这里 raise 避免
    # 跑出来一坨假数据进 paper.
    raise NotImplementedError(
        "B0 真 LLM 调用接入点 (yunwu.ai / vllm) 待实现; "
        "smoke / dry-run 请 fallback_random=True."
    )


def baseline_b0_strong_prompt(
    test_users: list[str],
    train_answers: pd.DataFrame,
    user_cluster: dict[str, int] | None = None,
    model: str = "gpt-4o",
    seed: int = 42,
    api_key: str | None = None,
    fallback_mode: bool = True,
) -> list[BaselinePrediction]:
    """B0' 强模型 + PKVC schema + peer RAG (Self-Discover 反证).

    实际跑法: 调 GPT-4o / Claude, prompt 含 PKVC schema + 3 peer NL.
    没 API key 且 fallback_mode=True → 退化为 per-cluster majority
    (比 random 强, 验证 "强 LLM 至少应该过 cluster majority floor").
    """
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        if fallback_mode and user_cluster is not None:
            return _baseline_cluster_majority(
                test_users, train_answers, user_cluster, method="B0_prime_cluster_majority_fallback",
            )
        raise NotImplementedError(
            "B0' needs API key; set OPENAI_API_KEY or pass api_key=... "
            "(or set fallback_mode=True with user_cluster)."
        )
    raise NotImplementedError(
        "B0' 真 LLM 调用接入点 (GPT-4o + PKVC schema + peer RAG) 待实现; "
        "smoke / dry-run 请 fallback_mode=True + 传 user_cluster."
    )


def _baseline_cluster_majority(
    test_users: list[str],
    train_answers: pd.DataFrame,
    user_cluster: dict[str, int],
    method: str = "cluster_majority",
) -> list[BaselinePrediction]:
    """Per-(cluster, task) majority answer; 离线 baseline / B0' fallback."""
    from collections import Counter, defaultdict
    ct_dist: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for r in train_answers.itertuples():
        c = user_cluster.get(str(r.user_id))
        if c is None or not r.answer:
            continue
        ct_dist[(int(c), str(r.task_id))][str(r.answer)] += 1
    task_ids = sorted(set(str(t) for t in train_answers["task_id"].unique()))
    preds: list[BaselinePrediction] = []
    for uid in test_users:
        c = user_cluster.get(str(uid))
        if c is None:
            continue
        for tid in task_ids:
            dist = ct_dist.get((int(c), tid))
            if not dist:
                continue
            ans = dist.most_common(1)[0][0]
            preds.append(BaselinePrediction(str(uid), tid, ans, method))
    return preds


def baseline_b1_sft_no_pkvc(checkpoint_dir: str, test_data: list[dict]) -> list[BaselinePrediction]:
    """SFT 但 output 不含 PKVC 段 (验证 PKVC 显化贡献).

    TODO: 加载 checkpoint, 推理时只生成 [答案] 不生成 [PKVC 分析].
    """
    raise NotImplementedError("B1 待 SFT 训练完成 + 推理脚本接入")


def baseline_b2_sft_random_traj(checkpoint_dir: str, test_data: list[dict]) -> list[BaselinePrediction]:
    """SFT 但 PKVC 标签随机洗牌 (验证标注真实性).

    TODO: 训练时 PKVC factor/sadr 字段 shuffle 后训, 推理走 normal pipeline.
    """
    raise NotImplementedError("B2 待 SFT 训练完成")


def baseline_b1_sft_no_pkvc(checkpoint_dir: str, test_data: list[dict]) -> list[BaselinePrediction]:
    """SFT 但 output 不含 PKVC 段 (验证 PKVC 显化贡献).

    TODO: 加载 checkpoint, 推理时只生成 [答案] 不生成 [PKVC 分析].
    """
    raise NotImplementedError("B1 待 SFT 训练完成 + 推理脚本接入")


def baseline_b2_sft_random_traj(checkpoint_dir: str, test_data: list[dict]) -> list[BaselinePrediction]:
    """SFT 但 PKVC 标签随机洗牌 (验证标注真实性).

    TODO: 训练时 PKVC factor/sadr 字段 shuffle 后训, 推理走 normal pipeline.
    """
    raise NotImplementedError("B2 待 SFT 训练完成")


# ====================================
# Baseline 统一 runner
# ====================================

def run_baseline(name: str, **kwargs) -> list[BaselinePrediction]:
    """统一 baseline 调用入口.

    Names: B0 (LLM prompt-only) | B0_prime (strong+RAG) | B0_random (offline) |
           B1 (SFT-no-PKVC) | B2 (SFT-random-traj) | B3 (KNN) | B4 (GBM)
    """
    if name == "B3":
        return baseline_b3_knn(**kwargs)
    if name == "B4":
        return baseline_b4_gbm(**kwargs)
    if name == "B0":
        return baseline_b0_prompt_only(**kwargs)
    if name == "B0_random":
        return baseline_b0_random_uniform(**kwargs)
    if name == "B0_prime":
        return baseline_b0_strong_prompt(**kwargs)
    if name == "B1":
        return baseline_b1_sft_no_pkvc(**kwargs)
    if name == "B2":
        return baseline_b2_sft_random_traj(**kwargs)
    raise ValueError(f"Unknown baseline: {name}")
