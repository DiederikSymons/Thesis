# Thesis
# GNN Anomaly Detection — v11 Experimental Blueprint
### A Mathematically Fair, Bug-Free Showdown for EU Bilateral Trade Flow Anomaly Detection

> **Status:** Pre-run design document. Results unknown. All reporting structures are dynamic.
> **Reference Architecture:** GUIDE (Yuan et al., 2024) adapted for temporal edge-level detection.

---

## Table of Contents

1. [The Dynamic Storyboard](#1-the-dynamic-storyboard)
2. [The Standardized Pipeline Architecture](#2-the-standardized-pipeline-architecture)
3. [Preemptive Bug-Squashing](#3-preemptive-bug-squashing)
4. [The Experimental Matrix](#4-the-experimental-matrix)
5. [Refactored Code Blueprint](#5-refactored-code-blueprint)

---

## 1. The Dynamic Storyboard

### 1.1 The Overarching Narrative

We are modeling the **EU-27 bilateral trade network** as a directed, attributed graph where:
- **Nodes** are 27 EU member states, each carrying 37 macroeconomic time-varying attributes
- **Edges** are directed trade flows (702 = 27×26), each enriched with 7 static CEPII geo-economic features
- **Edge targets** are 7-dimensional monthly trade volume vectors (TOTAL + 6 commodity categories)
- **Time** is a 143-step monthly panel from 2014-01 to 2025-11

The story we are testing is precisely this:

> *"Does incorporating graph topology — the structural relationships between EU economies — provide statistically meaningful lift for detecting anomalous trade flow events, compared to methods that treat each bilateral flow as an independent time series?"*

This is the central thesis of the GNN anomaly detection literature. The EU trade network is a natural laboratory for it: **geography, economic interdependence, shared currency blocs, and supply chain integration all encode real structural priors**. A flow between Germany and Poland is not independent of the Germany–Czech Republic flow. If GNNs are useful anywhere, it should be here.

The experiment will tell one of four stories — and our reporting structure must accommodate all of them:

| Outcome | Story |
|---|---|
| GNN >> Classical | Structure is signal. Graph topology captures cross-border shock propagation that univariate time-series methods miss. |
| GNN ≈ Classical | The trade network's structure is well-approximated by temporal autocorrelation alone. GNNs add complexity without detection gain. |
| GNN < Classical | The GNN's attribution branch introduces noise that degrades performance. Pure flow-residual baselines are the right tool. |
| Mixed (GNN wins some anomaly types) | Graph structure helps for specific anomaly morphologies (e.g., propagating shocks) but not for isolated spikes that are trivially detectable by Z-score. |

### 1.2 The Core Hypotheses

These are falsifiable, pre-registered before running the pipeline.

**H1 — The GNN Hypothesis (primary):**
> The best GNN-based method (GUIDE or DOMINANT) achieves a statistically higher AUROC on the held-out test set than the best classical baseline across the full set of 6 injected anomaly types.

**H2 — The Topology Hypothesis:**
> A semantically informed sparse adjacency (CEPII-KNN, K=6) produces equal or better AUROC than a complete graph adjacency, demonstrating that structural sparsity is a feature, not a limitation.

**H3 — The Dual-Branch Hypothesis (GUIDE-specific):**
> The combined GUIDE score (α·attr + (1-α)·flow) outperforms the flow-only branch (DOMINANT), providing evidence that node attribute reconstruction adds a complementary anomaly signal.

**H4 — The Alpha Hypothesis:**
> There exists a non-trivial optimal α ∈ (0, 1) that maximizes GUIDE's combined AUROC. Specifically, α = 0 (flow only) and α = 1 (attr only) are both suboptimal.

**H5 — The Anomaly-Type Hierarchy:**
> Detection difficulty is not uniform across the 6 injected anomaly types. Specifically: T1 (Spike) and T2 (Crash) will be easiest for classical methods; T5 (Node Shock) and T6 (Direction Swap) will be easiest for GNNs because they require relational reasoning.

### 1.3 The Dynamic Findings Template

Fill this in after running the pipeline. Every section has a slot for both a positive and negative result.

```
══════════════════════════════════════════════════════════════
  v11 EXPERIMENT RESULTS — EU TRADE FLOW ANOMALY DETECTION
  Labels: {n_anomalous_cells:,} anomalous / {n_total_cells:,} total ({pct_anom:.1f}%)
  Test window: 2024-01 → 2025-11  |  {n_test} steps × {E} edges × {N_CATS} cats
══════════════════════════════════════════════════════════════

── H1 VERDICT (GNN vs. Classical) ─────────────────────────
  Best GNN   : {best_gnn_name}   AUROC={best_gnn_auroc:.4f}
  Best Class : {best_cls_name}   AUROC={best_cls_auroc:.4f}
  Δ AUROC    : {delta:.4f}
  
  [IF Δ > 0.02]:  H1 SUPPORTED — GNNs provide meaningful lift
  [IF |Δ| < 0.02]: H1 INCONCLUSIVE — methods are statistically equivalent
  [IF Δ < -0.02]: H1 REFUTED — classical methods outperform GNNs

── H2 VERDICT (Topology) ───────────────────────────────────
  Complete  AUROC: {adj_complete_auroc:.4f}
  CEPII-KNN AUROC: {adj_cepii_auroc:.4f}
  Random    AUROC: {adj_random_auroc:.4f}
  
  [INTERPRET: if Random ≈ CEPII-KNN, structure doesn't matter]
  [INTERPRET: if CEPII-KNN > Complete, sparsity improves aggregation]

── H3 VERDICT (Dual-Branch) ───────────────────────────────
  GUIDE (dual, α={alpha:.2f}): AUROC={guide_auroc:.4f}
  DOMINANT (flow-only, α=0):  AUROC={dominant_auroc:.4f}
  GUIDE-flow (isolated):      AUROC={flow_only_auroc:.4f}
  GUIDE-attr (isolated):      AUROC={attr_only_auroc:.4f}
  
  [INTERPRET: if GUIDE > DOMINANT, dual branch adds value]
  [INTERPRET: if GUIDE < DOMINANT, attr branch is diluting a strong flow signal]
  [INTERPRET: if attr-only AUROC < 0.5, attribution is anti-correlated — inversion bug?]

── H4 VERDICT (Alpha Search) ──────────────────────────────
  α sweep results (best 3):
    α={a1:.2f}: AUROC={v1:.4f}
    α={a2:.2f}: AUROC={v2:.4f}
    α={a3:.2f}: AUROC={v3:.4f}
  Optimal α: {opt_alpha:.2f}
  
  [IF opt_alpha == 0.0]: H4 REFUTED — flow branch dominates entirely
  [IF opt_alpha in (0.05, 0.5)]: H4 SUPPORTED — dual branch has a true optimum

── H5 VERDICT (Per-Anomaly-Type) ──────────────────────────
  Best detector per type (AUROC):
    T1-Spike    : {t1_best_method} ({t1_auroc:.4f})
    T2-Crash    : {t2_best_method} ({t2_auroc:.4f})
    T3-Cessation: {t3_best_method} ({t3_auroc:.4f})
    T4-CatSwap  : {t4_best_method} ({t4_auroc:.4f})
    T5-NodeShock: {t5_best_method} ({t5_auroc:.4f})
    T6-DirSwap  : {t6_best_method} ({t6_auroc:.4f})

── FULL LEADERBOARD (test set) ─────────────────────────────
  {leaderboard_table}

── KEY FINDING ─────────────────────────────────────────────
  [One sentence written after reviewing results]
══════════════════════════════════════════════════════════════
```

---

## 2. The Standardized Pipeline Architecture

### 2.1 The Universal Core Flow

Every single model — classical or GNN — must traverse this exact pipeline. No exceptions.

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIVERSAL CORE FLOW                      │
│                                                             │
│  RAW DATA                                                   │
│    │                                                        │
│    ▼                                                        │
│  DataRegistry.load()  ← single call, shared by all         │
│    │ → node_feat_tensor  (T, N, F_total)                    │
│    │ → edge_feat_tensor  (E, F_edge)   [static]             │
│    │ → target_tensor     (T, E, C)                          │
│    │                                                        │
│    ▼                                                        │
│  Splitter.split()  ← defines train_idx, val_idx, test_idx  │
│    │  ─── WALL: train | val | test ───                      │
│    │                                                        │
│    ▼                                                        │
│  Normalizer.fit(train_only).transform(all)                  │
│    │ → node_feat_norm   (T, N, F_total)                     │
│    │ → edge_feat_norm   (E, F_edge)                         │
│    │ → target_scaled    (T, E, C)    ← CLEAN copy           │
│    │                                                        │
│    ▼                                                        │
│  AnomalyInjector.inject(target_scaled, test_idx only)       │
│    │ → target_scaled    (T, E, C)    ← PERTURBED test only  │
│    │ → label_tensor     (n_test, E, C)  [bool]              │
│    │                                                        │
│    ▼                                                        │
│  GraphBuilder.build(ADJ_CONFIG)                             │
│    │ → adj_configs: {name: (edge_index, edge_attr)}         │
│    │                                                        │
│    ▼                                                        │
│  for model in ExperimentMatrix:                             │
│      model.fit(train_data)          ← train only            │
│      scores = model.score(test_data)  ← (n_test, E, C)     │
│      results[model.name] = Evaluator.evaluate(scores,       │
│                                  label_tensor)              │
│    ▼                                                        │
│  Reporter.render(results)                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Standardized Data-Serving Contract

All models receive data in precisely this format. This is the contract every model must accept.

```python
@dataclass
class TrainBundle:
    """Everything a model needs during fit(). All statistics from train only."""
    node_feat_norm:  torch.Tensor   # (T, N, F_total)  — train timesteps
    edge_feat_norm:  torch.Tensor   # (E, F_edge)       — static, full E
    target_scaled:   torch.Tensor   # (T, E, C)         — CLEAN, train timesteps
    train_idx:       List[int]      # indices into common_time (for t → t+1 target)
    adj_configs:     Dict[str, Tuple[Tensor, Tensor]]  # name → (ei, ea)

@dataclass
class ScoreBundle:
    """Everything a model needs during score(). No labels visible here."""
    node_feat_norm:  torch.Tensor   # (T, N, F_total)  — ALL timesteps
    edge_feat_norm:  torch.Tensor   # (E, F_edge)
    target_scaled:   torch.Tensor   # (T, E, C)         — PERTURBED test
    test_idx:        List[int]
    adj_configs:     Dict[str, Tuple[Tensor, Tensor]]

# Contract: model.score() must return shape (n_test, E, C)
# NaN is allowed for missing flow cells; the Evaluator masks them.
```

### 2.3 The Canonical Normalization Protocol

This section is the single source of truth for all preprocessing. Every deviation is a bug.

**Node features (T, N, F_total):**
1. Log-transform: for each feature `f`, if `min(train) > 0`, apply `log1p`. Mask computed from train only.
2. Z-score: `(x - μ_train) / σ_train`. μ and σ computed per `(node, feature)` across train timesteps only.
3. Clamp σ < 1e-5 → 1.0 (avoids division by near-zero for constant features).

**Edge features (E, F_edge) [static CEPII]:**
1. Distance columns (`dist`, `distcap`, `distw`, `distwces`): apply `log1p`.
2. Min-max normalize each column using `min` and `max` over all E edges (static — no temporal leakage).
3. Binary columns (`comlang_ethno`, `contig`, `smctry`): leave as-is (already in {0,1}).

**Trade targets (T, E, C):**
1. Apply `log1p(clamp(x, min=0))` to handle zeros and large values.
2. Compute `TARGET_SCALE = quantile(0.99)` of non-NaN train values only.
3. Scale: `target_scaled = target_log / TARGET_SCALE`.
4. NaN propagation: cells where original was NaN remain NaN throughout.

**Critical invariant:** `target_scaled` is a mutable tensor. `AnomalyInjector` modifies it in-place on the test portion only. The Normalizer produces a clean copy first. Any model that calls `.fit()` only ever sees the clean pre-injection copy.

### 2.4 The Canonical Train/Val/Test Split

```python
TRAIN_END = pd.Timestamp("2019-12-01")   # inclusive
VAL_END   = pd.Timestamp("2023-12-01")   # inclusive

# One-step-ahead: t is the "context" step, t+1 is the "target" step
train_idx = [t for t in range(T-1) if common_time[t+1] <= TRAIN_END]
val_idx   = [t for t in range(T-1) if TRAIN_END < common_time[t+1] <= VAL_END]
test_idx  = [t for t in range(T-1) if common_time[t+1] > VAL_END]

# Expected: n_train=71, n_val=48, n_test=23
```

**Val role:** Monitoring only. Val MSE is logged for early-stopping reference and overfitting diagnosis. Val labels are never created. **Evaluation is exclusively on test.**

### 2.5 Evaluation Protocol (Universal)

Applied identically to every model's output array `scores: (n_test, E, C)`.

```python
class Evaluator:
    @staticmethod
    def evaluate(
        scores:  np.ndarray,   # (n_test, E, C)
        labels:  np.ndarray,   # (n_test, E, C)  bool
    ) -> EvalResult:
        flat_s = scores.reshape(-1)
        flat_l = labels.reshape(-1).astype(int)
        valid  = ~np.isnan(flat_s)
        s, l   = flat_s[valid], flat_l[valid]

        # Guard: no anomalies or all anomalies → skip
        if l.sum() == 0 or l.sum() == len(l):
            return EvalResult.null()

        # 99.9th percentile clip BEFORE AUROC (suppresses extreme reconstruction errors
        # caused by injected anomalies back-propagating into score arrays)
        s_clip = np.clip(s, 0, np.percentile(s, 99.9))

        # Threshold at true prevalence for F1
        prevalence = l.sum() / len(l)
        thresh     = np.percentile(s, 100.0 * (1.0 - prevalence))
        pred_b     = (s >= thresh).astype(int)

        # Recall@K: fraction of true positives in top-K predictions
        n_pos = int(l.sum())
        recalls = {}
        for k_mult in [0.5, 1.0, 2.0]:
            k = max(1, int(n_pos * k_mult))
            top_k = np.argsort(s)[::-1][:k]
            recalls[f"R@{k_mult}x"] = int(l[top_k].sum()) / n_pos

        return EvalResult(
            auroc  = roc_auc_score(l, s_clip),
            auprc  = average_precision_score(l, s_clip),
            f1     = f1_score(l, pred_b, zero_division=0),
            recalls= recalls,
        )

    @staticmethod
    def evaluate_per_type(
        scores:       np.ndarray,           # (n_test, E, C)
        labels_typed: Dict[str, np.ndarray], # type_name → (n_test, E, C) bool
    ) -> Dict[str, EvalResult]:
        return {name: Evaluator.evaluate(scores, lab)
                for name, lab in labels_typed.items()}

    @staticmethod
    def evaluate_per_category(
        scores:  np.ndarray,   # (n_test, E, C)
        labels:  np.ndarray,   # (n_test, E, C) bool
        cat_names: List[str],
    ) -> Dict[str, EvalResult]:
        return {cat: Evaluator.evaluate(scores[:,:,ci], labels[:,:,ci])
                for ci, cat in enumerate(cat_names)}
```

**AUROC Sanity Check (mandatory, not optional):**
```python
def auroc_sanity_check(name: str, auroc: float) -> None:
    if auroc < 0.45:
        warnings.warn(
            f"[SANITY] {name}: AUROC={auroc:.4f} < 0.45. "
            f"This is below random chance. Check: (1) score sign is correct "
            f"(high score should mean MORE anomalous); "
            f"(2) label alignment with score array — test_idx offset; "
            f"(3) calibration statistics computed on TRAIN not TEST."
        )
```

---

## 3. Preemptive Bug-Squashing

The following bugs were found across v2–v10. Each is catalogued with its root cause, the version it was introduced, the version it was fixed, and how the v11 architecture permanently prevents it.

---

### BUG-01: Anomaly Injection into Training / Val Data

| Field | Detail |
|---|---|
| **Versions affected** | v2–v7 |
| **Fixed in** | v8 (partial), v10 (complete) |
| **Severity** | CRITICAL — invalidates all results |
| **Root cause** | `AnomalyInjector` drew injection timesteps from `val_idx` (v2–v7), meaning: (a) the model was potentially trained on contaminated data, (b) val MSE was penalized by the injected anomalies it was supposed to detect, creating circular evaluation. In v8–v9, COVID/Ukraine "soft labels" were also applied to val steps. |

**v11 Fix:**
```python
class AnomalyInjector:
    def inject(self, target_scaled: Tensor, test_idx: List[int], ...) -> Tuple[Tensor, ndarray]:
        # HARD CONSTRAINT: Only timesteps in test_idx may be perturbed.
        # This is asserted, not just assumed.
        perturbed = target_scaled.clone()  # work on a copy
        label = np.zeros((len(test_idx), E, N_CATS), dtype=bool)
        
        for vi, t in enumerate(test_idx):
            t_tgt = t + 1  # the target timestep we can perturb
            # inject here ...
        
        # Assertion: training data is byte-for-byte identical to pre-injection
        assert torch.equal(perturbed[:test_idx[0]+1], target_scaled[:test_idx[0]+1]), \
            "FATAL: AnomalyInjector modified data before test window!"
        
        return perturbed, label
```

---

### BUG-02: COVID/Ukraine Soft Labels Contaminating Evaluation

| Field | Detail |
|---|---|
| **Versions affected** | v8, v9 |
| **Fixed in** | v10 |
| **Severity** | HIGH — evaluation conflates real events with injected synthetic anomalies |
| **Root cause** | `label_event` was derived from real macroeconomic windows (2020-03→2020-08, 2022-02→2022-12). But these windows were defined from domain knowledge AFTER observing the data. A model flagging genuine COVID disruptions is not "detecting anomalies" — it is fitting to historically real structural breaks. The combined `label_combined = label_synth | label_event` made AUROC meaningless: a model could score high purely by learning to flag COVID months. |

**v11 Fix:**
> **Principle: Ground truth must be generated by the experiment, not derived from external observation of the data.**
> v11 uses ONLY synthetic injection labels. Zero soft labels. Zero event windows.
> If you want to study COVID detection, that is a separate study on a separate notebook with a clearly stated null hypothesis.

---

### BUG-03: Normalization Statistics Computed Across Full Time Horizon

| Field | Detail |
|---|---|
| **Versions affected** | v2–v5 |
| **Fixed in** | v6 (node features), v8 (targets) |
| **Severity** | HIGH — introduces temporal leakage |
| **Root cause** | In v5, `TARGET_SCALE = target_log.quantile(0.99)` was computed over ALL T timesteps, including val and test. Same for node feature mean/std in earlier versions. This leaks test-period distribution into the scaling constant, giving the model unfair "foreknowledge" of the test data's scale. |

**v11 Fix:**
```python
class Normalizer:
    def fit(self, train_idx: List[int]):
        # Node features: μ, σ from train only
        train_nodes = self.node_feat_raw[train_idx]   # (n_train, N, F)
        self.node_mean = train_nodes.mean(dim=0)
        self.node_std  = train_nodes.std(dim=0).clamp(min=1e-5)
        
        # Target scale: 99th pct of non-NaN train values only
        train_tgt_log  = torch.log1p(self.target_raw[train_idx].clamp(min=0))
        valid_train    = train_tgt_log[~torch.isnan(train_tgt_log)]
        self.target_scale = float(valid_train.quantile(0.99))
        
        # Edge features: min-max (static — no time dimension, no leakage)
        # ... min/max computed over all E, not time-sliced
        return self
    
    def transform(self, ...):
        # Apply statistics from fit() to ALL data
        # Never re-compute statistics here
        ...
```

---

### BUG-04: Baselines Seeing Test-Period Data During Fit

| Field | Detail |
|---|---|
| **Versions affected** | v2–v7 |
| **Fixed in** | v9 (partial), v10 (verified) |
| **Severity** | HIGH — classical methods had an unfair advantage over GNNs |
| **Root cause** | In several versions, Z-score and EWMA computed their mean/variance over ALL timesteps including test. LOF in some versions was `fit()` called on all data before splitting. ARIMA in v5 used rolling-window forecasting that extended into the test period. |

**v11 Fix — universal rule:**
```python
class BaseDetector(ABC):
    @abstractmethod
    def fit(self, bundle: TrainBundle) -> "BaseDetector":
        """ALL statistics, model parameters, and calibration computed here.
        TrainBundle contains ONLY train_idx data. Implementors must not access
        any data outside bundle.train_idx."""
        ...

    @abstractmethod  
    def score(self, bundle: ScoreBundle) -> np.ndarray:
        """Returns (n_test, E, C) anomaly scores. No model re-fitting here.
        Uses frozen parameters from fit()."""
        ...
    
    # Enforced contract: score() is read-only with respect to model parameters.
    # Any call to self.fit() inside score() raises NotImplementedError.
```

---

### BUG-05: Self-Loop Handling Inconsistency

| Field | Detail |
|---|---|
| **Versions affected** | v6, v7 |
| **Fixed in** | v8 (CONV_NEEDS_SL dict) |
| **Severity** | MEDIUM — caused NaN embeddings for GCNConv; silent failures |
| **Root cause** | `GCNConv` requires self-loops for numerical stability of its symmetric normalization (D^{-1/2} A D^{-1/2}). `GATv2Conv` and `TransformerConv` have their own internal self-loop logic and should NOT have them added externally (double-adding caused incorrect attention distributions). `SAGEConv` uses a separate `root` weight matrix and does not use A at all — adding self-loops to it is harmless but wasteful. Early versions either added self-loops to everyone or no one. |

**v11 Fix:**
```python
# Canonical self-loop policy — baked into GraphBuilder, not into each model
CONV_NEEDS_SELFLOOP:  Set[Type] = {GCNConv}
CONV_HAS_INTERNAL_SL: Set[Type] = {GATv2Conv, TransformerConv}  # do NOT add externally
CONV_NO_SL:           Set[Type] = {SAGEConv}

def build_mp_graph(
    pairs:    List[Tuple[int,int]],
    conv_cls: Type,
    N:        int,
    edge_feat_norm: Tensor,
    edge_pair_to_idx: Dict,
    F_edge: int,
) -> Tuple[Tensor, Tensor]:
    ei = torch.tensor([[p[0] for p in pairs], [p[1] for p in pairs]], dtype=torch.long)
    ea = torch.stack([edge_feat_norm[edge_pair_to_idx[p]] for p in pairs])
    
    if conv_cls in CONV_NEEDS_SELFLOOP:
        sl = torch.arange(N, dtype=torch.long)
        ei = torch.cat([ei, torch.stack([sl, sl])], dim=1)
        ea = torch.cat([ea, torch.zeros(N, F_edge)], dim=0)
    # GATv2Conv and TransformerConv: add_self_loops=False in constructor
    # SAGEConv: no action needed
    return ei, ea
```

---

### BUG-06: t vs. t+1 Index Offset Misalignment (The "Off-By-One Temporal Leak")

| Field | Detail |
|---|---|
| **Versions affected** | v2–v5, intermittently in v6 |
| **Severity** | CRITICAL — uses future data as input |
| **Root cause** | The indexing convention is: `train_idx = [t]` where `t` is the context step and `t+1` is the target step. In several places, code used `node_feat_norm[t]` when it should have used `node_feat_norm[t+1]`, or used `target_scaled[t]` as both input and target. Because node features and targets are independent tensors, this is not always caught by shape errors. |

**v11 Fix — documented canonical convention:**
```python
# CANONICAL INDEXING CONVENTION — never deviate from this
# ─────────────────────────────────────────────────────────
# t       = context step index into common_time
# t+1     = target step index into common_time
# common_time[t]   = the month we OBSERVE (input features)
# common_time[t+1] = the month we PREDICT (target flows)
#
# For autoencoder training (reconstruction, not prediction):
#   x = node_feat_norm[t+1]    ← reconstruct THIS step's attributes
#   y = target_scaled[t+1]     ← reconstruct THIS step's flows
#   (t is used only to index into train_idx; t+1 is always the data step)
#
# LOF / classical baselines:
#   lag1 = target_np[max(0, t_tgt-1)]  where t_tgt = t+1
#   This is fine: lag1 is a PAST value relative to t_tgt
```

---

### BUG-07: GUIDE Score Combination Sign / Direction

| Field | Detail |
|---|---|
| **Versions affected** | v7, v8 (α assignment was internally inconsistent) |
| **Fixed in** | v9 |
| **Severity** | HIGH — AUROC < 0.5 means score is inverted |
| **Root cause** | In some versions, the GUIDE combined score was `α·flow + (1-α)·attr` while the calibration residuals were computed as `y_hat - y` (instead of `y - y_hat`). This inverted the sign in one branch, making the combined score anti-correlated with anomalies. The symptom: GUIDE-attr AUROC of 0.36 in v10 strongly suggests the attr branch is inverted. |

**v11 Fix — explicit sign convention + assertion:**
```python
# In compute_guide_scores():
# SIGN RULE: higher score = more anomalous. Always.
# Residual = |actual - predicted|  (absolute value, always positive)
# Z-score  = (|residual| - mean_train_residual) / std_train_residual

flow_resid = np.abs(target_scaled[t_tgt].numpy() - y_hat.cpu().numpy())  # ABSOLUTE
attr_resid = np.abs(x.cpu().numpy() - x_hat.cpu().numpy())               # ABSOLUTE

z_flow = (flow_resid - flow_mu_abs) / flow_sig
z_attr_node = np.mean(z_attr, axis=-1)  # (N,) → per-node attr anomaly score

# Assertion: flow score should be positively correlated with spike magnitude
# (Test with a single known-injected timestep before full evaluation)
```

---

### BUG-08: Recall@K with Hardcoded K = {50, 100, 150}

| Field | Detail |
|---|---|
| **Versions affected** | v9, v10 |
| **Severity** | LOW — not wrong, but uninterpretable across anomaly rates |
| **Root cause** | With 9,818 anomalous cells in v10, Recall@50 = 0.005 means only 0.5% of anomalies are in the top-50 predictions. The K values are not calibrated to the anomaly prevalence, making comparisons meaningless and making the numbers look trivially small (0.005 / 0.010 / 0.015 for all methods suggests all methods retrieve at most 3 anomalies in their top-150). |

**v11 Fix:**
```python
# Recall@K relative to anomaly count (much more interpretable)
# R@0.5x = recall when alerting on 50% of the true anomaly count
# R@1.0x = recall when alerting on exactly n_anomalies predictions (oracle budget)
# R@2.0x = recall when alerting on twice the anomaly count
for k_mult in [0.5, 1.0, 2.0]:
    k = max(1, int(n_anomalies * k_mult))
    top_k = np.argsort(s)[::-1][:k]
    recall = int(l[top_k].sum()) / n_anomalies
```

---

### BUG-09: Anomaly Injection Contaminates Calibration Statistics

| Field | Detail |
|---|---|
| **Versions affected** | v8–v9 (latent, partially fixed in v10) |
| **Severity** | MEDIUM — inflates anomaly scores at test time |
| **Root cause** | The calibration step (computing `flow_mu`, `flow_sig` from train residuals) is correctly done on train. However, if `target_scaled` is a shared mutable tensor and injection mutates it before calibration in an out-of-order cell execution, calibration statistics could see injected values. v10 executes injection (Cell E) before training (Cell I), but `target_scaled` is the same object throughout. |

**v11 Fix:**
```python
class DataRegistry:
    def __init__(self):
        self._target_clean: Tensor  # immutable reference copy, never passed to models
        self.target_for_training: Tensor  # clean copy, given to TrainBundle
        self.target_for_scoring: Tensor   # perturbed copy (test only), given to ScoreBundle
    
    def inject_and_split(self, injector: AnomalyInjector, test_idx):
        # training copy: never perturbed
        self.target_for_training = self._target_clean.clone()
        # scoring copy: perturbed at test positions
        self.target_for_scoring, self.labels = injector.inject(
            self._target_clean.clone(), test_idx
        )
        # models receive TrainBundle with target_for_training
        # score() receives ScoreBundle with target_for_scoring
```

---

### BUG-10: TOTAL Category Collinearity in Multi-Category Evaluation

| Field | Detail |
|---|---|
| **Versions affected** | v7–v10 |
| **Severity** | MEDIUM — TOTAL is a linear sum of the other 6 categories |
| **Root cause** | Injecting anomalies in ALL_CATS including `TOTAL` means that an injection into `Chemicals` also shows up in `TOTAL`. AUROC computed over the flattened (n_test × E × C) array double-counts anomalies (once in `TOTAL`, once in the specific category). |

**v11 Fix — two evaluation modes:**
```python
# Mode 1: Full (all 7 categories including TOTAL)
#   Used for comparison with v9/v10 results; may be slightly inflated
eval_full = Evaluator.evaluate(scores[:,:,:], labels[:,:,:])

# Mode 2: Commodity-only (exclude TOTAL category, idx=0)
#   More conservative and meaningful; no double-counting
eval_commodity = Evaluator.evaluate(scores[:,:,1:], labels[:,:,1:])

# REPORT BOTH. Primary metric for H1-H5 is eval_commodity.
```

---

### BUG-11: The α Sweep Was Never Systematically Tested

| Field | Detail |
|---|---|
| **Versions affected** | All versions (never addressed) |
| **Severity** | MEDIUM — ALPHA was set by intuition (0.2→0.3 across versions) |
| **Root cause** | α=0.3 in v10 was chosen based on the paper's recommendation, not empirical validation. Given that GUIDE-attr alone scores 0.36 AUROC (below random), it is plausible that any α > 0 degrades performance. This has never been tested with a grid search. |

**v11 Fix:**
```python
ALPHA_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
# Train GUIDE once with each α; use the SAME model weights, only vary the
# combination formula at score-time (no re-training needed):
# score(α) = (1-α)*z_flow + α*z_attr_bc
# This is cheap: one training run, nine score computations
for alpha in ALPHA_SWEEP:
    score_arr = (1 - alpha) * z_flow + alpha * z_attr_bc
    auroc = Evaluator.evaluate(score_arr, labels).auroc
    alpha_results[alpha] = auroc
```

---

## 4. The Experimental Matrix

The following models are entered into the fair showdown. Every model uses the same data, same split, same normalizer, same anomaly labels, same Evaluator.

### 4.1 Classical Baseline Tier

| ID | Method | Type | Key Params | Notes |
|---|---|---|---|---|
| `ZSCORE` | Per-edge Z-score | Tabular | train μ/σ per (edge, category) | Fastest; best in v10 (0.654 AUROC) |
| `EWMA` | Exponentially Weighted MA | Time-series | span=12 | Fit on train residuals |
| `STL` | Seasonal-Trend Decomposition | Time-series | period=12 | Residual = actual − (trend + seasonal) |
| `ARIMA` | ARIMA(1,1,1) | Time-series | fit per edge×category | Slow (~500s); parallelized |
| `LOF` | Local Outlier Factor | Tabular+lag | n_neighbors=20, lag features | Fit on train featurized rows |
| `AE_MLP` | MLP Autoencoder | Deep tabular | H=64, latent=32 | No graph structure |

**Shared classical requirement:** All fit on `train_idx`. All score `test_idx`. Use the same `target_scaled` after injection (i.e., they see the same perturbed test targets as the GNNs — this is correct: anomaly scores are residuals, and the injected anomalies are the large residuals we want to find).

### 4.2 GNN Tier — DOMINANT (Flow-Only AE)

| ID | Conv Layer | Adjacency | Notes |
|---|---|---|---|
| `DOM_GCN_COMPLETE` | GCNConv | Complete | Full graph |
| `DOM_GCN_CEPII` | GCNConv | CEPII-KNN | Sparse semantic |
| `DOM_GCN_RANDOM` | GCNConv | Random | Sparse control |
| `DOM_GAT_CEPII` | GATv2Conv | CEPII-KNN | Attention-weighted |
| `DOM_SAGE_CEPII` | SAGEConv | CEPII-KNN | Inductive aggregation |
| `DOM_TFM_CEPII` | TransformerConv | CEPII-KNN | Best in v10 (DOMINANT) |

**DOMINANT = EdgeFlowAE only. No node attr reconstruction. Anomaly score = z_flow only.**

### 4.3 GNN Tier — GUIDE (Dual AE)

| ID | Conv Layer | Adjacency | α (combined score) | Notes |
|---|---|---|---|---|
| `GUIDE_TFM_CEPII_a0` | TransformerConv | CEPII-KNN | 0.0 (flow only) | Equivalent to DOMINANT |
| `GUIDE_TFM_CEPII_a05` | TransformerConv | CEPII-KNN | 0.05 | α sweep point |
| `GUIDE_TFM_CEPII_a10` | TransformerConv | CEPII-KNN | 0.10 | α sweep point |
| `GUIDE_TFM_CEPII_a20` | TransformerConv | CEPII-KNN | 0.20 | v9 default |
| `GUIDE_TFM_CEPII_a30` | TransformerConv | CEPII-KNN | 0.30 | v10 default |
| `GUIDE_TFM_CEPII_a50` | TransformerConv | CEPII-KNN | 0.50 | Balanced |
| `GUIDE_TFM_CEPII_BEST` | TransformerConv | CEPII-KNN | `opt_α` | Best from sweep |
| `GUIDE_GCN_CEPII_BEST` | GCNConv | CEPII-KNN | `opt_α` | Architecture ablation |
| `GUIDE_GAT_CEPII_BEST` | GATv2Conv | CEPII-KNN | `opt_α` | Architecture ablation |

**Note:** The α sweep models share the same training run. Train TransformerConv GUIDE once; compute 9 score arrays at different α values. This is cheap.

### 4.4 Model Hyperparameter Invariants (All GNN Models)

These are **frozen** across all GNN experiments. No per-model tuning that could advantage one variant.

```python
GNN_HPARAMS = dict(
    hidden_dim    = 128,
    n_encoder_layers = 2,
    n_epochs      = 100,
    lr            = 3e-4,        # overridden per conv class (ARCH_LR)
    grad_clip     = 1.0,
    dropout       = 0.1,
    heads         = 4,           # for attention-based convs
    weight_decay  = 1e-5,
    scheduler     = "cosine",    # CosineAnnealingLR, eta_min=lr/50
    seed          = 42,
)

# Per-architecture learning rate (well-established in literature)
ARCH_LR = {
    GCNConv:         3e-4,
    GATv2Conv:       1e-4,
    SAGEConv:        3e-4,
    TransformerConv: 1e-4,
}
```

---

## 5. Refactored Code Blueprint

### 5.1 Module Structure

```
gnn_anomaly_v11/
├── config.py            # All constants, paths, hyperparameters — single source of truth
├── data/
│   ├── registry.py      # DataRegistry — loads CSVs, assembles tensors
│   ├── normalizer.py    # Normalizer — fit/transform, train-only statistics
│   ├── splitter.py      # Splitter — canonical index sets
│   ├── injector.py      # AnomalyInjector — 6 types, test-only, assertion-guarded
│   └── graph_builder.py # GraphBuilder — adjacency configs, self-loop policy
├── models/
│   ├── base.py          # BaseDetector ABC (fit/score interface)
│   ├── classical/
│   │   ├── zscore.py
│   │   ├── ewma.py
│   │   ├── stl_detector.py
│   │   ├── arima_detector.py
│   │   ├── lof_detector.py
│   │   └── mlp_ae.py
│   └── gnn/
│       ├── conv_utils.py     # _build_conv, build_mp_graph, ARCH_LR
│       ├── node_attr_ae.py   # GUIDENodeAttrAE
│       ├── edge_flow_ae.py   # GUIDEEdgeFlowAE
│       ├── dominant.py       # DOMINANTDetector (flow AE only)
│       └── guide.py          # GUIDEDetector (dual AE + α sweep)
├── evaluation/
│   ├── evaluator.py     # Evaluator (universal evaluate/per_type/per_category)
│   └── reporter.py      # Reporter (leaderboard, hypothesis verdicts)
├── viz/
│   └── plots.py         # All visualization functions
└── experiment.py        # Main entry: orchestrates the full pipeline
```

### 5.2 Core Class Interfaces

```python
# ══════════════════════════════════════════════════════════════════════════
# config.py
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    start_month:  str   = "2014-01"
    end_month:    str   = "2025-11"
    countries:    tuple = ("AT","BE","BG","CY","CZ","DE","DK","EE","EL",
                           "ES","FI","FR","HR","HU","IE","IT","LT","LU",
                           "LV","MT","NL","PL","PT","RO","SE","SI","SK")
    all_cats:     tuple = ("TOTAL","Food_drinks_tobacco","Raw_materials",
                           "Energy","Chemicals","Machinery_vehicles",
                           "Other_manufactured")
    cepii_cols:   tuple = ("comlang_ethno","contig","smctry",
                           "dist","distcap","distw","distwces")
    distance_cols:tuple = ("dist","distcap","distw","distwces")
    
    # Split
    train_end:    str   = "2019-12-01"
    val_end:      str   = "2023-12-01"
    
    # Anomaly injection
    spike_mult:   float = 30.0
    crash_mult:   float = 0.02
    n_per_type:   int   = 150
    
    # GNN
    hidden_dim:   int   = 128
    n_epochs:     int   = 100
    lr:           float = 3e-4
    grad_clip:    float = 1.0
    heads:        int   = 4
    k_neighbors:  int   = 6
    dropout:      float = 0.1
    weight_decay: float = 1e-5
    seed:         int   = 42
    
    # Alpha sweep
    alpha_sweep:  tuple = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    
    # Logging
    log_every:    int   = 10


# ══════════════════════════════════════════════════════════════════════════
# data/registry.py
# ══════════════════════════════════════════════════════════════════════════
class DataRegistry:
    """Single point of data loading and tensor assembly.
    All downstream components receive data from this registry — never load CSVs twice.
    """
    def __init__(self, cfg: ExperimentConfig, data_dir: Path):
        self.cfg = cfg
        self.data_dir = data_dir
        
        # Public tensors (set after load())
        self.node_feat_tensor: Tensor  # (T, N, F_total)
        self.edge_feat_tensor: Tensor  # (E, F_edge)
        self.target_tensor:    Tensor  # (T, E, C)  — raw, unscaled
        self.common_time: pd.DatetimeIndex
        self.edge_pairs:  List[Tuple[int,int]]
        self.T: int; self.N: int; self.E: int
        self.F_total: int; self.F_edge: int; self.N_cats: int
    
    def load(self) -> "DataRegistry":
        """Load all CSVs, validate, assemble tensors."""
        ...
        return self
    
    def validate(self) -> None:
        """Assert: no NaNs in node features; correct shapes; no duplicate edges."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# data/normalizer.py
# ══════════════════════════════════════════════════════════════════════════
class Normalizer:
    """Fits all scaling statistics exclusively on training data."""
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        # Fitted parameters (set after fit())
        self.node_mean:    Tensor  # (N, F_total)
        self.node_std:     Tensor  # (N, F_total)
        self.log_mask:     Tensor  # (F_total,) bool
        self.target_scale: float
        self.edge_min:     Tensor  # (F_edge,)
        self.edge_max:     Tensor  # (F_edge,)
        self._is_fitted:   bool = False
    
    def fit(self, registry: DataRegistry, train_idx: List[int]) -> "Normalizer":
        ...
        self._is_fitted = True
        return self
    
    def transform(self, registry: DataRegistry) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (node_feat_norm, edge_feat_norm, target_scaled_clean).
        target_scaled_clean is NEVER modified after this point."""
        assert self._is_fitted, "Call fit() before transform()"
        ...
    
    def detransform_target(self, x: Tensor) -> Tensor:
        """Inverse of target scaling, for interpretability."""
        return torch.expm1(x.clamp(min=-30) * self.target_scale)


# ══════════════════════════════════════════════════════════════════════════
# data/injector.py
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class InjectionResult:
    target_perturbed: Tensor                  # (T, E, C) — test portion modified
    labels:           np.ndarray              # (n_test, E, C) bool
    labels_per_type:  Dict[str, np.ndarray]   # type_name → (n_test, E, C) bool
    synth_events:     List[Tuple]             # (vi, e_idx, c_idx, type_name) for viz

class AnomalyInjector:
    """
    Injects 6 types of anomalies into the test portion of target_scaled only.
    
    Types:
      T1 - Flow Spike:      multiply by SPIKE_MULT (30x)
      T2 - Flow Crash:      multiply by CRASH_MULT (0.02x)
      T3 - Cessation:       set to 0.0 (trade stops)
      T4 - Category Swap:   swap two commodity categories for one edge
      T5 - Node Shock:      inject spike across ALL outgoing edges from one node
      T6 - Direction Swap:  swap A→B and B→A flows for 3 consecutive months
    
    All types: N_PER_TYPE injection attempts each.
    All types: only timesteps in test_idx are eligible.
    All types: NaN cells are skipped (cannot inject into missing flows).
    """
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
    
    def inject(
        self,
        target_clean:   Tensor,
        test_idx:       List[int],
        edge_pairs:     List[Tuple[int,int]],
    ) -> InjectionResult:
        np.random.seed(self.cfg.seed)
        target = target_clean.clone()
        
        labels          = np.zeros((len(test_idx), *target.shape[1:]), dtype=bool)
        labels_per_type = {}
        synth_events    = []
        
        for type_fn, type_name in [
            (self._inject_spike,     "T1-Spike"),
            (self._inject_crash,     "T2-Crash"),
            (self._inject_cessation, "T3-Cessation"),
            (self._inject_cat_swap,  "T4-CatSwap"),
            (self._inject_node_shock,"T5-NodeShock"),
            (self._inject_dir_swap,  "T6-DirSwap"),
        ]:
            type_labels = np.zeros_like(labels)
            cnt = self._attempt(
                self.cfg.n_per_type, type_fn,
                target, test_idx, edge_pairs, type_labels, synth_events
            )
            labels_per_type[type_name] = type_labels.copy()
            labels |= type_labels
            print(f"  {type_name:<16}: {cnt} injection events")
        
        # ASSERTION: no training data was modified
        first_test_t = test_idx[0] + 1
        assert torch.equal(target[:first_test_t], target_clean[:first_test_t]), \
            "FATAL: AnomalyInjector modified pre-test data!"
        
        return InjectionResult(target, labels, labels_per_type, synth_events)
    
    def _attempt(self, n_target, fn, *args) -> int:
        cnt, attempts = 0, 0
        while cnt < n_target and attempts < n_target * 20:
            if fn(*args):
                cnt += 1
            attempts += 1
        return cnt
    
    def _inject_spike(self, target, test_idx, edge_pairs, labels, events) -> bool: ...
    def _inject_crash(self, target, test_idx, edge_pairs, labels, events) -> bool: ...
    def _inject_cessation(self, target, test_idx, edge_pairs, labels, events) -> bool: ...
    def _inject_cat_swap(self, target, test_idx, edge_pairs, labels, events) -> bool: ...
    def _inject_node_shock(self, target, test_idx, edge_pairs, labels, events) -> bool: ...
    def _inject_dir_swap(self, target, test_idx, edge_pairs, labels, events) -> bool: ...


# ══════════════════════════════════════════════════════════════════════════
# models/base.py
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class TrainBundle:
    node_feat_norm: Tensor          # (T, N, F_total)
    edge_feat_norm: Tensor          # (E, F_edge)
    target_scaled:  Tensor          # (T, E, C)  — CLEAN (pre-injection)
    train_idx:      List[int]
    val_idx:        List[int]       # for monitoring MSE only
    adj_configs:    Dict[str, Tuple[Tensor, Tensor]]
    cfg:            ExperimentConfig

@dataclass
class ScoreBundle:
    node_feat_norm: Tensor          # (T, N, F_total)
    edge_feat_norm: Tensor          # (E, F_edge)
    target_scaled:  Tensor          # (T, E, C)  — PERTURBED (post-injection)
    test_idx:       List[int]
    adj_configs:    Dict[str, Tuple[Tensor, Tensor]]
    cfg:            ExperimentConfig

class BaseDetector(ABC):
    """
    Universal interface for all anomaly detectors (classical + GNN).
    
    Contract:
      1. fit()   → reads only train_idx data from bundle; sets all model parameters
      2. score() → reads only test_idx data; returns (n_test, E, C) scores
      3. score() must NOT re-fit or modify self parameters
      4. Higher score = more anomalous (universally enforced)
    """
    def __init__(self, name: str, cfg: ExperimentConfig):
        self.name = name
        self.cfg  = cfg
        self._fitted = False
    
    @abstractmethod
    def fit(self, bundle: TrainBundle) -> "BaseDetector":
        ...
    
    @abstractmethod
    def score(self, bundle: ScoreBundle) -> np.ndarray:
        """Returns shape (n_test, E, C). NaN allowed for missing cells."""
        ...
    
    def fit_score(
        self, train_bundle: TrainBundle, score_bundle: ScoreBundle
    ) -> np.ndarray:
        self.fit(train_bundle)
        scores = self.score(score_bundle)
        # Sanity check: shape
        expected = (len(score_bundle.test_idx), 
                    self.cfg.n * (self.cfg.n - 1), 
                    len(self.cfg.all_cats))
        assert scores.shape == expected, \
            f"{self.name}: score shape {scores.shape} != expected {expected}"
        return scores


# ══════════════════════════════════════════════════════════════════════════
# models/gnn/guide.py
# ══════════════════════════════════════════════════════════════════════════
class GUIDEDetector(BaseDetector):
    """
    GUIDE-inspired dual GNN autoencoder.
    
    Mathematical formulation:
      Encoder (shared for both branches):
        h^(0) = x_t  ∈ R^{N × F_total}
        h^(l) = σ( Conv_l(h^{l-1}, A, E_feat) )
        h^(l) = LayerNorm(h^(l)) + residual
        z_t = h^(L)  ∈ R^{N × H}
      
      Branch 1 — Node Attribute Reconstruction:
        x̂_t = Dec_node(z_t, A, E_feat)  ∈ R^{N × F_total}
        L_attr = MSE(x̂_t, x_t)
      
      Branch 2 — Edge Flow Reconstruction:
        ŷ_{ij,t} = MLP([z_i, z_j, e_{ij}])  ∈ R^{C}
        L_flow = MSE(ŷ_t[valid], y_t[valid])  # masked for NaN flows
      
      Combined loss:
        L = (1 - α) · L_flow  +  α · L_attr
      
      Anomaly score for edge (i→j) at test step v:
        z_flow_{ij,v,c} = |y_{ij,v,c} - ŷ_{ij,v,c} - μ_flow| / σ_flow
        z_attr_{v,ij}   = 0.5 · (z_node_v[i] + z_node_v[j])
                where z_node_v[i] = mean_f(|x_{v,i,f} - x̂_{v,i,f} - μ_attr| / σ_attr)
        
        score(α)_{ij,v,c} = (1-α) · z_flow_{ij,v,c}  +  α · z_attr_{v,ij}
    
    Note: μ_flow, σ_flow, μ_attr, σ_attr are calibrated on TRAIN residuals.
          Both are absolute-value residuals (BUG-07 fix).
    """
    def __init__(
        self,
        name:     str,
        cfg:      ExperimentConfig,
        conv_cls: Type,
        adj_name: str,
        alpha:    float = 0.0,      # default flow-only; sweep outside
    ):
        super().__init__(name, cfg)
        self.conv_cls = conv_cls
        self.adj_name = adj_name
        self.alpha    = alpha
        
        # Set after fit()
        self.node_ae:    nn.Module
        self.flow_ae:    nn.Module
        self.calib:      Dict     # {flow_mu_abs, flow_sig, attr_mu_abs, attr_sig}
        self.val_mse:    float    # monitoring only
    
    def fit(self, bundle: TrainBundle) -> "GUIDEDetector":
        ei_d, ea_d = bundle.adj_configs[self.adj_name]
        ...
        self._fitted = True
        return self
    
    def score(self, bundle: ScoreBundle) -> np.ndarray:
        assert self._fitted
        ...
        return score_arr  # (n_test, E, C)
    
    def score_with_alpha(
        self, bundle: ScoreBundle, alpha: float
    ) -> np.ndarray:
        """Score with a different α without re-training. Used for α sweep."""
        assert self._fitted
        # Returns (n_test, E, C) for the given α value
        ...
    
    def score_alpha_sweep(
        self, bundle: ScoreBundle, alphas: List[float]
    ) -> Dict[float, np.ndarray]:
        """One training run → N score arrays for N α values. O(N) not O(N×epochs)."""
        return {α: self.score_with_alpha(bundle, α) for α in alphas}


# ══════════════════════════════════════════════════════════════════════════
# evaluation/evaluator.py
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class EvalResult:
    auroc:   float
    auprc:   float
    f1:      float
    recalls: Dict[str, float]   # {"R@0.5x": ..., "R@1.0x": ..., "R@2.0x": ...}
    n_valid: int
    n_pos:   int
    
    @classmethod
    def null(cls):
        return cls(auroc=np.nan, auprc=np.nan, f1=np.nan,
                   recalls={}, n_valid=0, n_pos=0)
    
    def is_valid(self) -> bool:
        return not np.isnan(self.auroc)

class Evaluator:
    @staticmethod
    def evaluate(scores: np.ndarray, labels: np.ndarray) -> EvalResult:
        ...  # (see Section 2.5 above for full implementation)
    
    @staticmethod
    def evaluate_per_type(
        scores: np.ndarray,
        labels_per_type: Dict[str, np.ndarray],
    ) -> Dict[str, EvalResult]:
        ...
    
    @staticmethod
    def evaluate_per_category(
        scores: np.ndarray,
        labels: np.ndarray,
        cat_names: List[str],
    ) -> Dict[str, EvalResult]:
        ...
    
    @staticmethod
    def evaluate_commodity_only(
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> EvalResult:
        """Excludes TOTAL category (idx=0) to avoid double-counting. Primary metric."""
        return Evaluator.evaluate(scores[:,:,1:], labels[:,:,1:])


# ══════════════════════════════════════════════════════════════════════════
# evaluation/reporter.py
# ══════════════════════════════════════════════════════════════════════════
class Reporter:
    def __init__(self, cfg: ExperimentConfig, all_cats: List[str]):
        self.cfg = cfg
        self.all_cats = all_cats
        self.results: Dict[str, EvalResult] = {}
    
    def register(self, model_name: str, result: EvalResult) -> None:
        self.results[model_name] = result
        # Sanity check on registration
        if result.is_valid():
            auroc_sanity_check(model_name, result.auroc)
    
    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for name, r in sorted(self.results.items(), key=lambda x: -(x[1].auroc or 0)):
            rows.append({
                "Method":   name,
                "ROC-AUC":  f"{r.auroc:.4f}" if r.is_valid() else "N/A",
                "PR-AUC":   f"{r.auprc:.4f}" if r.is_valid() else "N/A",
                "F1":       f"{r.f1:.4f}"    if r.is_valid() else "N/A",
                "R@0.5x":   f"{r.recalls.get('R@0.5x', np.nan):.3f}",
                "R@1.0x":   f"{r.recalls.get('R@1.0x', np.nan):.3f}",
                "R@2.0x":   f"{r.recalls.get('R@2.0x', np.nan):.3f}",
            })
        return pd.DataFrame(rows)
    
    def hypothesis_verdicts(self, ...) -> Dict[str, str]:
        """Evaluates H1–H5 and returns a verdict string for each."""
        ...
    
    def print_full_report(self) -> None:
        """Prints the full dynamic findings template (Section 1.3) filled in."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# experiment.py  — The Orchestrator
# ══════════════════════════════════════════════════════════════════════════
def run_experiment(cfg: ExperimentConfig, data_dir: Path) -> Reporter:
    """
    The single entry point. Runs the entire pipeline in order.
    Returns a Reporter with all results filled in.
    """
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    
    # 1. Load
    print("── [1/7] Loading data ──")
    registry = DataRegistry(cfg, data_dir).load()
    registry.validate()
    
    # 2. Split
    print("── [2/7] Splitting ──")
    splitter = Splitter(cfg)
    train_idx, val_idx, test_idx = splitter.split(registry.common_time)
    
    # 3. Normalize (train statistics only)
    print("── [3/7] Normalizing ──")
    normalizer = Normalizer(cfg).fit(registry, train_idx)
    node_norm, edge_norm, target_clean = normalizer.transform(registry)
    
    # 4. Inject anomalies (test only)
    print("── [4/7] Injecting anomalies ──")
    injector = AnomalyInjector(cfg)
    inj_result = injector.inject(target_clean, test_idx, registry.edge_pairs)
    
    # 5. Build graph topologies
    print("── [5/7] Building adjacency configs ──")
    graph_builder = GraphBuilder(cfg)
    adj_configs = graph_builder.build_all(registry, edge_norm)
    
    # 6. Assemble bundles
    train_bundle = TrainBundle(
        node_feat_norm=node_norm,
        edge_feat_norm=edge_norm,
        target_scaled=target_clean,     # CLEAN for training
        train_idx=train_idx,
        val_idx=val_idx,
        adj_configs=adj_configs,
        cfg=cfg,
    )
    score_bundle = ScoreBundle(
        node_feat_norm=node_norm,
        edge_feat_norm=edge_norm,
        target_scaled=inj_result.target_perturbed,  # PERTURBED for scoring
        test_idx=test_idx,
        adj_configs=adj_configs,
        cfg=cfg,
    )
    
    # 7. Run all models
    print("── [6/7] Running experiment matrix ──")
    reporter = Reporter(cfg, list(cfg.all_cats))
    
    for detector in build_experiment_matrix(cfg):
        print(f"  Training: {detector.name}")
        scores = detector.fit_score(train_bundle, score_bundle)
        
        # Primary metric: commodity-only AUROC
        result = Evaluator.evaluate_commodity_only(scores, inj_result.labels)
        reporter.register(detector.name, result)
        
        # Secondary: per-type, per-category
        per_type = Evaluator.evaluate_per_type(scores, inj_result.labels_per_type)
        per_cat  = Evaluator.evaluate_per_category(scores, inj_result.labels,
                                                   list(cfg.all_cats))
        reporter.register_detailed(detector.name, per_type, per_cat)
    
    # 8. Alpha sweep for best GUIDE model
    print("  Alpha sweep for GUIDE-TFM-CEPII...")
    guide_model = GUIDEDetector("GUIDE_TFM_CEPII", cfg, TransformerConv, "CEPII-KNN")
    guide_model.fit(train_bundle)
    alpha_results = {}
    for alpha in cfg.alpha_sweep:
        sc = guide_model.score_with_alpha(score_bundle, alpha)
        r  = Evaluator.evaluate_commodity_only(sc, inj_result.labels)
        alpha_results[alpha] = r
        reporter.register(f"GUIDE_TFM_CEPII_a{int(alpha*100):02d}", r)
    
    # 9. Print findings
    print("── [7/7] Results ──")
    reporter.print_full_report()
    reporter.print_hypothesis_verdicts(alpha_results)
    
    return reporter


def build_experiment_matrix(cfg: ExperimentConfig) -> List[BaseDetector]:
    """Returns the full list of detectors in the order they should run
    (classical first — cheap; GNN last — expensive)."""
    detectors = []
    
    # Classical tier
    detectors += [
        ZScoreDetector("ZSCORE", cfg),
        EWMADetector("EWMA", cfg),
        STLDetector("STL-resid", cfg),
        ARIMADetector("ARIMA-resid", cfg),
        LOFDetector("LOF", cfg),
        MLPAEDetector("AE-MLP", cfg),
    ]
    
    # DOMINANT tier (flow AE only)
    for conv_cls, adj_name in [
        (GCNConv,         "Complete"),
        (GCNConv,         "CEPII-KNN"),
        (GCNConv,         "Random"),
        (GATv2Conv,       "CEPII-KNN"),
        (SAGEConv,        "CEPII-KNN"),
        (TransformerConv, "CEPII-KNN"),
    ]:
        name = f"DOM_{conv_cls.__name__.replace('Conv','')}_{adj_name}"
        detectors.append(DOMINANTDetector(name, cfg, conv_cls, adj_name))
    
    # GUIDE tier — α=0.30 baseline; sweep handled separately in run_experiment()
    for conv_cls, adj_name in [
        (GCNConv,         "CEPII-KNN"),
        (GATv2Conv,       "CEPII-KNN"),
        (TransformerConv, "CEPII-KNN"),
    ]:
        name = f"GUIDE_{conv_cls.__name__.replace('Conv','')}_{adj_name}_a30"
        detectors.append(GUIDEDetector(name, cfg, conv_cls, adj_name, alpha=0.30))
    
    return detectors
```

### 5.3 Naming Schema

| Entity | Convention | Example |
|---|---|---|
| Config constants | `UPPER_SNAKE` | `HIDDEN_DIM`, `TRAIN_END` |
| Tensor variables | `lower_snake` with shape comment | `node_feat_norm  # (T, N, F)` |
| Model classes | `PascalCase` + Detector/AE suffix | `GUIDEDetector`, `ZScoreDetector` |
| Model instances | `lower_snake` | `guide_model`, `zscore_detector` |
| Score arrays | `scores_<model>` or just `scores` inside a function | `scores_guide`, `scores` |
| Label arrays | `labels` (full), `labels_per_type` (dict) | consistent everywhere |
| Index variables | `t` (context), `t_tgt = t+1` (target), `vi` (val/test position) | canonical convention |
| Results dict key | f-string: `f"{conv.__name__}_{adj}_{extra}"` | `"TransformerConv_CEPII-KNN_a30"` |

---

## Appendix A: The Six Anomaly Types — Mathematical Specification

Let $y_{e,t,c}$ = scaled trade flow on edge $e$, timestep $t$, category $c$.

| Type | Injection Rule | Expected Detector |
|---|---|---|
| **T1 — Spike** | $y_{e,t,c} \leftarrow y_{e,t,c} \times 30$ | Z-score, EWMA, flow AE |
| **T2 — Crash** | $y_{e,t,c} \leftarrow y_{e,t,c} \times 0.02$ | Z-score, EWMA, flow AE |
| **T3 — Cessation** | $y_{e,t,c} \leftarrow 0$ | STL (structural break) |
| **T4 — CatSwap** | $y_{e,t,c_1}, y_{e,t,c_2} \leftarrow y_{e,t,c_2}, y_{e,t,c_1}$ for $c_1 \neq c_2$ | GUIDE (attr branch) |
| **T5 — NodeShock** | $\forall e: \text{src}(e)=i: y_{e,t,:} \leftarrow y_{e,t,:} \times 30$ | GUIDE (neighborhood aware) |
| **T6 — DirSwap** | $y_{i \to j, t:t+3,:}, y_{j \to i, t:t+3,:} \leftarrow$ swapped | GUIDE (directed structure) |

T5 and T6 are the anomaly types that **require** relational reasoning. A univariate method on edge $i \to j$ alone cannot detect T5 without observing that all edges leaving $i$ are simultaneously anomalous.

---

## Appendix B: Invariant Checklist (Run Before Each Experiment)

```
□  DataRegistry.validate() passes (no NaNs in node features, correct shapes)
□  Normalizer statistics computed exclusively from train_idx
□  TARGET_SCALE matches between TrainBundle and ScoreBundle (same frozen value)
□  AnomalyInjector assertion: pre-test data byte-identical to target_clean
□  All model.fit() calls use TrainBundle (target_scaled = clean copy)
□  All model.score() calls use ScoreBundle (target_scaled = perturbed copy)
□  Evaluator.evaluate() receives scores.shape == (n_test, E, C)
□  No model.fit() inside model.score()
□  AUROC sanity check: no method has AUROC < 0.45 without a diagnosed reason
□  Leaderboard printed before any visualization (so results are always visible)
□  Seed fixed: torch.manual_seed(42) + np.random.seed(42) at start of run_experiment()
□  Val MSE logged separately; never used to select between models for final evaluation
```
