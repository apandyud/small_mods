import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from scipy.stats import chi2, binomtest
import pandas as pd


# ---------- Util: illesztés, ΔEM, b/c táblák ----------

def align_on_items(
    df_base: pd.DataFrame,
    df_rule: pd.DataFrame,
    key_cols: Tuple[str, ...] = ("qid",),
) -> pd.DataFrame:
    """
    Join baseline and rule results egy közös kulcson (pl. item_id, opcionálisan variant).
    Elvárja, hogy mindkét DF-ben legyen 'em' oszlop 0/1 értékekkel.
    """
    cols_b = list(key_cols) + ["exact_match"]
    cols_r = list(key_cols) + ["exact_match"]
    left = df_base[cols_b].rename(columns={"exact_match": "em_base"})
    right = df_rule[cols_r].rename(columns={"exact_match": "em_rule"})
    df = pd.merge(left, right, on=list(key_cols), how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError("Nincs közös kulcs az összehasonlításhoz (üres illesztés).")
    return df


def confusion_counts(df_aligned: pd.DataFrame) -> Dict[str, int]:
    """
    McNemar-hoz szükséges 2x2 contingency:
      a: 1→1, d: 0→0, b: 0→1 (javult), c: 1→0 (romlott)
    """
    em_b, em_r = df_aligned["em_base"], df_aligned["em_rule"]
    a = int(((em_b == 1) & (em_r == 1)).sum())
    b = int(((em_b == 0) & (em_r == 1)).sum())
    c = int(((em_b == 1) & (em_r == 0)).sum())
    d = int(((em_b == 0) & (em_r == 0)).sum())
    return {"a": a, "b": b, "c": c, "d": d}


def delta_em(df_aligned: pd.DataFrame) -> float:
    """ΔEM százalékpontban (rule - base) az illesztett mintán."""
    return (df_aligned["em_rule"].mean() - df_aligned["em_base"].mean()) * 100.0


# ---------- McNemar teszt (kontinuitás-korrekció + egzakt) ----------

@dataclass
class McNemarResult:
    b: int
    c: int
    p_chi2_cc: float
    p_exact_binom: float


def mcnemar_test(b: int, c: int) -> McNemarResult:
    """
    McNemar teszt két módon:
      - Chi-négyzet 1 df, kontinuitás-korrekcióval: ((|b-c|-1)^2) / (b+c)
      - Egzakt binomiális: H0: p=0.5 a discordáns párokban, kétoldali
    """
    if b + c == 0:
        # Nincs discordáns pár → nincs bizonyíték hatásra
        return McNemarResult(b=b, c=c, p_chi2_cc=1.0, p_exact_binom=1.0)

    # Chi-square with continuity correction
    chi2_cc = ((abs(b - c) - 1) ** 2) / (b + c)
    p_chi2 = 1.0 - chi2.cdf(chi2_cc, df=1)

    # Exact binomial (two-sided) on min(b,c) successes out of (b+c) with p=0.5
    k = min(b, c)
    n = b + c
    p_exact = binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue

    return McNemarResult(b=b, c=c, p_chi2_cc=p_chi2, p_exact_binom=p_exact)


# ---------- Bootstrap CI a ΔEM-re ----------

@dataclass
class BootstrapResult:
    delta_point: float
    ci_low: float
    ci_high: float
    deltas: np.ndarray


def bootstrap_delta_em(
    df_aligned: pd.DataFrame,
    key_col: str = "qid",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: Optional[int] = 42,
) -> BootstrapResult:
    """
    Item-szintű bootstrap: item_id-ket mintavételezünk visszatevéssel,
    minden mintán újraszámoljuk ΔEM-et (százalékpont). Percentilis CI-t ad vissza.
    """
    rng = np.random.default_rng(random_state)
    items = df_aligned[key_col].unique()
    m = len(items)
    deltas = np.empty(n_boot, dtype=float)

    # Gyors lookup táblák
    base_map = df_aligned.set_index(key_col)["em_base"].to_dict()
    rule_map = df_aligned.set_index(key_col)["em_rule"].to_dict()

    for i in range(n_boot):
        sample = rng.choice(items, size=m, replace=True)
        base_vals = np.fromiter((base_map[it] for it in sample), dtype=float)
        rule_vals = np.fromiter((rule_map[it] for it in sample), dtype=float)
        deltas[i] = (rule_vals.mean() - base_vals.mean()) * 100.0

    alpha = 1 - ci
    low = np.percentile(deltas, 100 * (alpha / 2))
    high = np.percentile(deltas, 100 * (1 - alpha / 2))
    point = delta_em(df_aligned)
    return BootstrapResult(delta_point=point, ci_low=low, ci_high=high, deltas=deltas)


# ---------- Permutációs teszt a discordáns párokon ----------

@dataclass
class PermTestResult:
    delta_point: float
    p_value: float


def permutation_test_delta(
    df_aligned: pd.DataFrame,
    n_perm: int = 10000,
    random_state: Optional[int] = 123,
) -> PermTestResult:
    """
    Permutációs (randomization) teszt a ΔEM-re:
    Csak a discordáns párok (0→1 vagy 1→0) kapnak véletlen előjelet.
    Ez H0 alatt szimmetrikus legyen (nincs különbség a két módszer között).
    """
    rng = np.random.default_rng(random_state)
    # Discordáns párok kivonatolása
    discord = df_aligned[df_aligned["em_base"] != df_aligned["em_rule"]].copy()
    if discord.empty:
        # nincs különbség → nincs bizonyíték
        return PermTestResult(delta_point=0.0, p_value=1.0)

    # Valós ΔEM
    delta_obs = delta_em(df_aligned)

    # Permutációk: random felcseréljük a 0/1-et a két oszlop között discordánsoknál
    m = len(discord)
    greater_eq = 0
    for _ in range(n_perm):
        flip = rng.integers(0, 2, size=m, dtype=int)  # 0: no flip, 1: swap
        em_b = discord["em_base"].to_numpy().copy()
        em_r = discord["em_rule"].to_numpy().copy()
        # swap ahol flip==1
        em_b_new = np.where(flip == 1, em_r, em_b)
        em_r_new = np.where(flip == 1, em_b, em_r)

        # Teljes mintán kell számolni → készítünk másolatot az egészből
        df_perm = df_aligned.copy()
        df_perm.loc[discord.index, "em_base"] = em_b_new
        df_perm.loc[discord.index, "em_rule"] = em_r_new
        delta_perm = delta_em(df_perm)

        if abs(delta_perm) >= abs(delta_obs):
            greater_eq += 1

    p = (greater_eq + 1) / (n_perm + 1)  # add-one smoothing
    return PermTestResult(delta_point=delta_obs, p_value=p)


# ---------- Stabilitás több prompt-variánson ----------

@dataclass
class StabilityResult:
    per_variant: List[Dict[str, Any]]
    pass_rate: float


def evaluate_on_variants(
    df_base: pd.DataFrame,
    df_rule: pd.DataFrame,
    variant_col: str = "variant",
    alpha: float = 0.05,
    min_delta_pp: float = 0.5,  # százalékpont küszöb
    use_exact_mcnemar: bool = True,
) -> StabilityResult:
    """
    Variánsonként lefuttatja a McNemar-tesztet és kiszámolja a ΔEM-et.
    Visszaadja a replikáció arányát: hány variánson volt (p<alpha és ΔEM>=küszöb).
    """
    if variant_col not in df_base.columns or variant_col not in df_rule.columns:
        raise ValueError("A stabilitás-vizsgálathoz mindkét DF-ben kell 'variant' oszlop.")

    results = []
    variants = sorted(set(df_base[variant_col]).intersection(df_rule[variant_col]))

    for v in variants:
        sub_b = df_base[df_base[variant_col] == v]
        sub_r = df_rule[df_rule[variant_col] == v]
        aligned = align_on_items(sub_b, sub_r, key_cols=("qid",))
        counts = confusion_counts(aligned)
        mres = mcnemar_test(counts["b"], counts["c"])
        dpp = delta_em(aligned)

        pval = mres.p_exact_binom if use_exact_mcnemar else mres.p_chi2_cc
        passed = (pval < alpha) and (dpp >= min_delta_pp)

        results.append({
            "variant": v,
            "N": len(aligned),
            "b_improved": counts["b"],
            "c_degraded": counts["c"],
            "delta_pp": dpp,
            "p_mcnemar": pval,
            "pass": passed
        })

    pass_rate = np.mean([r["pass"] for r in results]) if results else 0.0
    return StabilityResult(per_variant=results, pass_rate=pass_rate)


# ---------- Fő "rule gate" döntési függvény ----------

@dataclass
class RuleDecision:
    accepted: bool
    reason: str
    summary: Dict[str, Any]


def rule_gate_decision(
    df_base: pd.DataFrame,
    df_rule: pd.DataFrame,
    alpha: float = 0.05,
    min_delta_pp: float = 0.5,
    use_bootstrap: bool = True,
    bootstrap_kwargs: Optional[Dict[str, Any]] = None,
    use_permutation: bool = False,
    permutation_kwargs: Optional[Dict[str, Any]] = None,
    require_variants: bool = False,
    variants_kwargs: Optional[Dict[str, Any]] = None,
) -> RuleDecision:
    """
    Egy helyen meghozza a döntést egy szabály megtartásáról.
    Kritériumok (mind teljesüljenek, ha be vannak kapcsolva):
      1) McNemar p < alpha ÉS ΔEM >= min_delta_pp
      2) (opcionális) Bootstrap 95% CI alsó széle > 0
      3) (opcionális) Permutációs teszt p < alpha
      4) (opcionális) Variáns-stabilitás: pass rate >= 0.8
    """
    # 0) Illesztés
    key_cols = ("qid",) if "variant" not in df_base.columns else ("qid", "variant")
    aligned = align_on_items(df_base, df_rule, key_cols=key_cols)

    # 1) McNemar + ΔEM
    counts = confusion_counts(aligned)
    mres = mcnemar_test(counts["b"], counts["c"])
    dpp = delta_em(aligned)

    passed_base = (mres.p_exact_binom < alpha) and (dpp >= min_delta_pp)

    summary = {
        "N": len(aligned),
        "b_improved": counts["b"],
        "c_degraded": counts["c"],
        "delta_pp": dpp,
        "p_mcnemar_exact": mres.p_exact_binom,
        "p_mcnemar_chi2cc": mres.p_chi2_cc,
        "criteria": {"mcnemar_and_delta": passed_base}
    }

    # 2) Bootstrap CI (opcionális)
    if use_bootstrap:
        boot_kwargs = {"n_boot": 2000, "ci": 0.95, "random_state": 42}
        if bootstrap_kwargs:
            boot_kwargs.update(bootstrap_kwargs)
        # bootstrapot item-szinten értelmezzük → ha variáns oszlop van, egyetlen variánst se keverjünk:
        if "variant" in key_cols:
            # aggregáljunk item szintre: a két em oszlop legyen item-átlag a variánsok felett
            collapsed = aligned.groupby("qid")[["em_base", "em_rule"]].mean().reset_index()
            boot = bootstrap_delta_em(collapsed, key_col="qid", **boot_kwargs)
        else:
            boot = bootstrap_delta_em(aligned, key_col="qid", **boot_kwargs)

        summary["bootstrap"] = {
            "delta_point": boot.delta_point,
            "ci_low": boot.ci_low,
            "ci_high": boot.ci_high,
        }
        summary["criteria"]["bootstrap_CI_excludes_0"] = (boot.ci_low > 0)

    # 3) Permutációs teszt (opcionális)
    if use_permutation:
        perm_kwargs = {"n_perm": 10000, "random_state": 123}
        if permutation_kwargs:
            perm_kwargs.update(permutation_kwargs)
        # permutációt is a teljes illesztésen futtatjuk
        perm_res = permutation_test_delta(aligned, **perm_kwargs)
        summary["permutation"] = {"delta_point": perm_res.delta_point, "p_value": perm_res.p_value}
        summary["criteria"]["perm_test"] = (perm_res.p_value < alpha)

    # 4) Variáns-stabilitás (opcionális)
    if require_variants:
        if "variant" not in df_base.columns or "variant" not in df_rule.columns:
            raise ValueError("Variant-stabilitás kérése mellett mindkét DF-ben kell 'variant' oszlop.")
        vk = {"alpha": alpha, "min_delta_pp": min_delta_pp, "variant_col": "variant"}
        if variants_kwargs:
            vk.update(variants_kwargs)
        stab = evaluate_on_variants(df_base, df_rule, **vk)
        summary["stability"] = {"per_variant": stab.per_variant, "pass_rate": stab.pass_rate}
        summary["criteria"]["stability_pass_rate"] = (stab.pass_rate >= 0.8)

    # Végső döntés: minden bekapcsolt kritérium teljesüljön
    accepted = all(val for val in summary["criteria"].values())

    reason_parts = []
    if not summary["criteria"]["mcnemar_and_delta"]:
        reason_parts.append("McNemar p>=alpha vagy ΔEM túl kicsi")
    if use_bootstrap and not summary["criteria"]["bootstrap_CI_excludes_0"]:
        reason_parts.append("bootstrap CI tartalmazza a 0-t")
    if use_permutation and not summary["criteria"]["perm_test"]:
        reason_parts.append("permutációs p>=alpha")
    if require_variants and not summary["criteria"]["stability_pass_rate"]:
        reason_parts.append("variáns-stabilitás < 0.8")

    reason = "OK" if accepted else ("; ".join(reason_parts) or "Elutasítva")

    return RuleDecision(accepted=accepted, reason=reason, summary=summary)
