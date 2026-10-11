"""模型加载 + LoRA + 辅助头 (v3.7: L_cc 保留, L_cls 砍, trait_reg 新加).

v3.7 设计:
  - QwenSFTModel 包装 PEFT LoRA 模型
  - 头部:
      trait_reg_head:  prompt 末端 hidden → 4-d trait factor 回归 (替代 cls_head, v3.7)
      cls_head (deprecated): K-way cluster_id 分类 — v3.7 cluster 弱区分实证后砍掉,
                              仅 enable_legacy_cls=True 时保留 (兼容 v3.5.1 实验复现)
      (L_cc 不需要 head, 直接对 response 末端 hidden 做 contrastive — 保留)
  - forward 返回 (sft_loss, trait_reg_pred, cc_anchor_hidden, [cls_logits 可选])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenSFTModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        lora_config: dict,
        n_clusters: int = 0,
        trait_dim: int = 4,
        enable_legacy_cls: bool = False,
        enable_trait_reg: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """v3.7:
        - enable_trait_reg (default True): 加 trait_reg_head [hidden → trait_dim=4]
        - enable_legacy_cls (default False): 加旧 cls_head [hidden → n_clusters]
          仅 v3.5.1 实验复现时设 True; 默认 v3.7 砍掉 cluster
        - n_clusters: 仅 enable_legacy_cls=True 时用, 否则 ignored
        - trait_dim: factor score 维度 (跟 EFA K_EFA 对齐, 默认 4)
        """
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            device_map="auto",
        )
        lc = LoraConfig(
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("alpha", 64),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        self.lm = get_peft_model(base, lc)
        self.lm.gradient_checkpointing_enable()
        self.lm.enable_input_require_grads()
        self.lm.print_trainable_parameters()

        hidden_size = base.config.hidden_size
        # heads live on same device as base model (device_map="auto" → GPU 0 single-GPU).
        device = next(base.parameters()).device

        # v3.7 default head: trait factor regression (continuous, 替代 cluster cls)
        self.enable_trait_reg = enable_trait_reg
        self.trait_dim = trait_dim
        if enable_trait_reg:
            self.trait_reg_head = nn.Linear(hidden_size, trait_dim, dtype=torch_dtype, device=device)
        else:
            self.trait_reg_head = None

        # [DEPRECATED v3.7] cls_head: cluster K-way classification
        # 仅 enable_legacy_cls=True 时存在, 用于 v3.5.1 实验复现
        self.enable_legacy_cls = enable_legacy_cls
        self.n_clusters = n_clusters
        if enable_legacy_cls:
            assert n_clusters > 0, "enable_legacy_cls=True 需要 n_clusters>0"
            self.cls_head = nn.Linear(hidden_size, n_clusters, dtype=torch_dtype, device=device)
        else:
            self.cls_head = None

        # Capture last hidden state via forward hook on final RMSNorm
        # — avoids output_hidden_states=True which materializes all 28 layers
        # and breaks grad checkpoint memory savings (brief v3.5.1 §4C).
        self._captured_last_hidden = None
        final_norm = base.model.norm   # Qwen2Model.norm (final RMSNorm)
        final_norm.register_forward_hook(self._capture_last_hidden)

    def _capture_last_hidden(self, module, inputs, output):
        self._captured_last_hidden = output

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None,
                prompt_end_pos: torch.Tensor | None = None,
                response_end_pos: torch.Tensor | None = None,
                output_hidden_states: bool = True) -> dict:
        """
        Args:
          prompt_end_pos: [B] 每个样本 prompt 末端 token 位置 (用于 L_cls anchor)
          response_end_pos: [B] response 末端 token 位置 (用于 L_cc anchor)
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        sft_loss = out.loss
        last_hidden = self._captured_last_hidden  # [B, T, H] from hook

        # v3.7 trait_reg anchor: prompt 末端 (替代 L_cls anchor 同位置)
        trait_reg_pred = None
        cls_logits = None
        if prompt_end_pos is not None:
            B = input_ids.size(0)
            prompt_hidden = last_hidden[torch.arange(B), prompt_end_pos]  # [B, H]
            if self.trait_reg_head is not None:
                trait_reg_pred = self.trait_reg_head(prompt_hidden)       # [B, trait_dim]
            if self.cls_head is not None:  # legacy v3.5.1
                cls_logits = self.cls_head(prompt_hidden)

        # L_cc anchor: response 末端 (v3.7 保留)
        cc_anchor = None
        if response_end_pos is not None:
            B = input_ids.size(0)
            cc_anchor = last_hidden[torch.arange(B), response_end_pos]  # [B, H]

        return {
            "loss": sft_loss,
            "trait_reg_pred": trait_reg_pred,
            "cls_logits": cls_logits,
            "cc_anchor": cc_anchor,
            "lm_logits": out.logits,
        }


def compute_cluster_contrastive_loss(cc_anchors: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
    """L_cc = Σ_(c ≠ c') |⟨h_c, h_c'⟩| / N_pairs

    Args:
      cc_anchors: [B, H] response 末端 hidden
      cluster_ids: [B] each sample 的 cluster id
    """
    # 每 cluster 的 mean hidden
    unique_clusters = torch.unique(cluster_ids)
    cluster_means = []
    for c in unique_clusters:
        mask = (cluster_ids == c)
        cluster_means.append(cc_anchors[mask].mean(dim=0))
    cluster_means = torch.stack(cluster_means)  # [K, H]

    # 归一化
    cluster_means = F.normalize(cluster_means, dim=-1)

    # 余弦相似度矩阵 (off-diagonal 取绝对值求和)
    sim = cluster_means @ cluster_means.T  # [K, K]
    K = cluster_means.size(0)
    if K < 2:
        return torch.tensor(0.0, device=cc_anchors.device, dtype=cc_anchors.dtype)
    off_diag = sim - torch.diag(torch.diag(sim))
    n_pairs = K * (K - 1)
    return off_diag.abs().sum() / n_pairs


def compute_cluster_classification_loss(cls_logits: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
    """[DEPRECATED v3.7] L_cls = CE(cls_logits, cluster_ids).

    v3.7 cluster 弱区分实证后砍掉. 仅 enable_legacy_cls=True 时调用.
    """
    return F.cross_entropy(cls_logits, cluster_ids)


def compute_trait_regression_loss(
    trait_reg_pred: torch.Tensor,
    trait_target: torch.Tensor,
) -> torch.Tensor:
    """v3.7: L_trait = MSE(trait_reg_pred, trait_target) on 4-d EFA factor scores.

    替代 L_cls (cluster K-way) — 用 trait factor 连续回归作 auxiliary supervision.

    Args:
      trait_reg_pred: [B, trait_dim=4]
      trait_target: [B, trait_dim=4] (factor_scores from collate_fn)
    """
    return F.mse_loss(trait_reg_pred, trait_target)


def load_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
