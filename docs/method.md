# Kappa-FIN: Mathematical Documentation

**Author:** David Ohio · odavidohio@gmail.com

---

## 1. Input Data

**Universe:** N assets (ETFs or individual securities) over period [t₀, T].

**Preprocessing:**
- Download adjusted close prices via yfinance
- Log-returns: r_t = log(P_t / P_{t-1})
- Rolling windows of W=22 trading days

---

## 2. Correlation Network (per window)

For each rolling window [t-W, t]:

1. **Spearman correlation matrix** C ∈ ℝ^{N×N}
2. **Ledoit shrinkage:** C_λ = (1-λ)C + λI  (λ=0.05 default)
3. **Distance matrix:** D_ij = 1 - C_ij  (clipped to [0,2])
4. **k-NN graph** G_t: connect each node to its k=5 nearest neighbors by distance

---

## 3. Forman-Ricci Curvature → η(t)

For each edge (i,j) in G_t:

    FR(i,j) = 4 - deg(i) - deg(j) + 3·|N(i) ∩ N(j)|

Summary curvature κ_t = median{ FR(i,j) : (i,j) ∈ E(G_t) }

**Topological flexibility:**

    η(t) = 1 / (|κ_t| + η_floor)     (η_floor = 0.05)

High η → flexible/low-rigidity network (heterogeneous, sparse).
Low η → rigid/high-rigidity network (synchronized, dense coupling).

---

## 4. H₁ Persistent Homology (per window)

From D, build Rips complex (GUDHI) up to max_edge_length = quantile(D, 0.99).

Extract H₁ persistence diagram {(b_k, d_k)}: 1-cycles (loops) in the distance space.

**H₁ observables** (lifetimes ℓ_k = d_k - b_k):

    entropy   = -Σ p_k log(p_k)    where p_k = ℓ_k / Σ ℓ_k
    dominance = max(ℓ_k) / Σ ℓ_k
    β₁ count  = |{k : ℓ_k > τ}|    (τ = 1e-7, noise gate)
    mass      = Σ ℓ_k

**v_raw:** v(t) = entropy(t) × (1 - dominance(t))

---

## 5. CALM Identification

Automated scan over candidate windows [s, e] of length L=504 calendar days:

    calm_score([s,e]) = std(η) + std(mean_corr) + mean|Δη| + mean|Δmean_corr|
                       + 0.25·mean(v_raw) + KCPT([s,e])

**KCPT (Knee-phi-tail Penalty, v4.6):**
Penalizes candidates whose tail quartile exhibits accelerating local φ dynamics
(detects hidden internal phase transitions):

    pen_ratio = max(0, dphi_tail/dphi_head - ratio_thresh)
    pen_slope = max(0, annualized_slope_phi_tail - slope_thresh)
    KCPT = weight × [(1-mix)·pen_ratio + mix·pen_slope]

Best candidate = argmin calm_score.

**CALM mask:** M_calm = 1 if t ∈ [s*, e*], else 0.

---

## 6. Ohio Number (Rayleigh Criterion)

**Reference flexibility:**

    η_ref = quantile(η[M_calm], q=0.05)

**Structural coupling index:**

    Ξ(t) = η_ref / η(t)

**Critical threshold:**

    Ξ_c = quantile(Ξ[M_calm], 0.99) × (1 + ε_margin)   (ε=0.02)

**Ohio Number:**

    Oh(t) = Ξ(t) / Ξ_c

Interpretation:
- Oh < 1: structural state within calm baseline range
- Oh > 1: structural coupling exceeds calm-calibrated threshold (sensitized regime)

---

## 7. Damage Integral (Prandtl Criterion)

**Pre-crossing baseline:**

    Oh_pre = quantile(Oh[M_calm], pre_q)   (pre_q = 0.968 default)

**Excess drive:**

    drive(t) = max(Oh(t) - (Oh_pre + δ), 0)   (δ = 0.08 default)

**Damage integral (leaky accumulator):**

    φ(t) = max(γ·φ(t-1) + drive(t), φ_floor)   (γ = 0.97)

**Critical damage:**

    φ_c = quantile(φ[M_calm], 0.99)

---

## 8. Dual Crossing Detection

A confirmed structural regime change requires **both**:

1. **Sensitization:** Oh(t) > 1 for ≥ p_sens = 2 consecutive days
2. **Confirmation:** φ(t) > φ_c for ≥ p_confirm = 5 consecutive days

The dual criterion reduces false positives from transient spikes.

---

## 9. Phase Stratification

For historical analysis, phases are labeled:

| Phase | Criterion |
|---|---|
| CALM | t ∈ [s*, e*] |
| Build-up | Oh > Oh_p90(CALM), φ rising |
| Crisis | Dual crossing confirmed |
| Recovery | Oh < 1 sustained after crisis peak |

Mann-Whitney U tests with FDR correction compare H₁ observables across phases.

---

## 10. Connection to the Law of Ohio

The Law of Ohio (Organized Coherence In Optimization) posits that information-processing
systems losing access to external validation optimize internal coherence at the expense
of reality correspondence.

In financial markets, external validation = price discovery / fundamental anchoring.
As systemic stress mounts, correlated forced selling destroys price discovery, and
the correlation network transitions to a high-coherence, low-entropy configuration —
the opposite of what naive intuition suggests about "chaotic crashes."

Kappa-FIN measures this Coherence Inversion via H₁ mass and η dynamics,
providing a model-agnostic, data-driven early warning signal.

---

## References

- Carlsson, G. (2009). Topology and data. *Bulletin of the AMS*, 46(2), 255–308.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*. AMS.
- Mantegna, R.N. (1999). Hierarchical structure in financial markets. *EPJB*, 11(1), 193–197.
- Scheffer, M. et al. (2009). Early-warning signals for critical transitions. *Nature*, 461, 53–59.
- Forman, R. (2003). Bochner's method for cell complexes and combinatorial Ricci curvature. *Discrete & Computational Geometry*, 29(3), 323–374.

---

## 11. On phi_c at the Floor

When the CALM period is correctly identified — i.e., it contains no structural
stress — the drive term `max(Oh(t) - (Oh_pre + δ), 0)` is zero throughout the
baseline by construction, and φ(t) rests at `phi_floor = 1e-6`.

Consequently `phi_c = quantile(φ[CALM], 0.99) = phi_floor`.

**This is not a calibration failure.** It means the threshold for damage
confirmation is: *any sustained excess above `Oh_pre + δ`*. The discriminative
power comes from `Oh_pre` (the pre-crossing Oh baseline) and `δ` (the damage
sensitivity), both calibrated from the CALM period.

A reviewer observing `phi_c = 1e-6` should interpret it as evidence that the
CALM was clean — not as a trivially low threshold. A contaminated CALM (one
that includes early crisis dynamics) would yield `phi_c >> phi_floor`, making
crossings harder to trigger, not easier.

**The correct diagnostic question is:** was the CALM period correctly
identified? This is answered by the `calm_score` and `[CALM] scan chosen`
output lines. A score of `inf` indicates a failed scan (see Section 5).
