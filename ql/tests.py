import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

# ===== 1. Încarcă datele =====
# df = pd.read_csv('csv/v1_results_final.csv')
df = pd.read_csv('csv/v3_results.csv')

# Afișează rapid structura
print("Primele 5 randuri:\n", df.head(), "\n")
print("Valori unice pentru 'n':", sorted(df['n'].unique()), "\n")

# Setează coloana de interes
# coloana = 'execution_time'
# coloana = 'comparators_found'
coloana = 'depth_found'

print(f"Pentru coloana: {coloana}")

# ===== 2. Shapiro-Wilk pentru fiecare grup =====
# H_0: datele provin dintr-o distributie normala
print("=== Shapiro-Wilk: test de normalitate pe fiecare n ===")
for val_n in sorted(df['n'].unique()):
    grup = df.loc[df['n'] == val_n, coloana]
    statistic_w, p_value = stats.shapiro(grup)
    verdict = "Normal" if p_value > 0.05 else "Nu e normal"
    print(f"n={val_n}: W={statistic_w:.20f}, p={p_value:.20f} -> {verdict}")
print()

# ===== 3. Levene: omogenitatea varianțelor =====
# H_0: variantele in toate grupurile sunt egale
grupuri = [df.loc[df['n'] == val_n, coloana].values
           for val_n in sorted(df['n'].unique())]
stat_levene, p_levene = stats.levene(*grupuri)
levene_verdict = "Variante omogene" if p_levene > 0.05 else "Variante ne-omogene"
print(f"=== Levene (omogenitatea variantelor) ===")
print(f"W={stat_levene:.20f}, p={p_levene:.20f} -> {levene_verdict}\n")

# ===== 4. Dacă datele sunt normale si variantele omogene, se face ANOVA =====
# Verifica conditiile (exemplu simplificat):
cond_normale = all(stats.shapiro(df.loc[df['n'] == val_n, coloana])[1] > 0.05
                   for val_n in sorted(df['n'].unique())) # valoarea testului saphiro > 0.05
cond_variante = p_levene > 0.05

if cond_normale and cond_variante:
    print("Toate grupurile par normale si variantele sunt omogene -> facem One-way ANOVA\n")
    # H_0: toate mediile celor 7 grupuri sunt egale
    f_stat, p_anova = stats.f_oneway(*grupuri) # statistica f - cat de mare este variatia dintre grupuri si variatia din interiorul grupurilor
    print(f"ANOVA: F = {f_stat:.20f}, p = {p_anova:.20f}")

    if p_anova < 0.05:
        print("Rezultat semnificativ -> fac post-hoc Tukey HSD\n")
        # Pregatire date pt Tukey
        df_tukey = df[['n', coloana]].copy()
        df_tukey.rename(columns={'n': 'group', coloana: 'value'}, inplace=True)

        tukey = pairwise_tukeyhsd(endog=df_tukey['value'],
                                  groups=df_tukey['group'],
                                  alpha=0.05)
        print(tukey)
    else:
        print("Nu exista diferente semnificative intre medii (p >= 0.05).")
else:
    # ===== 5. Altfel, facem Kruskal-Wallis =====
    # echivalent neparametric al ANOVA, compara distributiile medii-rang (nu mediile propriu-zise)
    print("Cel putin un grup nu e normal sau variantele nu-s omogene -> facem Kruskal-Wallis\n")
    h_stat, p_kruskal = stats.kruskal(*grupuri)
    print(f"Kruskal-Wallis: H = {h_stat:.20f}, p = {p_kruskal:.20f}")

    if p_kruskal < 0.05:
        print("Rezultat semnificativ -> fac post-hoc pairwise Mann-Whitney + Bonferroni\n")
        comparisons = list(itertools.combinations(sorted(df['n'].unique()), 2))
        results = []
        for (n1, n2) in comparisons:
            x1 = df.loc[df['n'] == n1, coloana]
            x2 = df.loc[df['n'] == n2, coloana]
            stat, p_val = stats.mannwhitneyu(x1, x2, alternative='two-sided')
            p_bonf = p_val * len(comparisons)
            results.append((n1, n2, stat, p_val, min(p_bonf, 1.0)))

        print("Post-hoc Mann-Whitney + Bonferroni:")
        for r in results:
            n1, n2, stat, p_val, p_corr = r
            verdict = "sig." if p_corr < 0.05 else "nu e sig."
            print(f"  n={n1} vs n={n2}: U={stat:.20f}, p_unadj={p_val:.20f}, "
                  f"p_bonf={p_corr:.20f} -> {verdict}")
    else:
        print("Nu exista diferente semnificative intre distributii (p >= 0.05).")
