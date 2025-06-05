import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, bootstrap
from scipy.stats import mannwhitneyu

BASE_DIR = Path(__file__).resolve().parent

file_options = {
    "nbWires < 8": BASE_DIR / "csv" / "bingo-mici.csv",
    "nbWires >= 8": BASE_DIR / "csv" / "bingo2.csv",
    "nbWires = 10": BASE_DIR / "csv" / "bingo10.csv",
    "nbWires = 17": BASE_DIR / "csv" / "bingo17.csv",
}

st.set_page_config(page_title="Analiză Metode Sorting Networks", layout="wide")

selected_labels = st.sidebar.multiselect(
    "Fișiere pentru analiză",
    options=list(file_options.keys()),
    default=list(file_options.keys())
)

if not selected_labels:
    st.warning("Cel puțin un fișier trebuie selectat.")
    st.stop()

dfs = []
for label in selected_labels:
    path = file_options[label]
    try:
        df_temp = pd.read_csv(path)
        df_temp["source_file"] = label
        dfs.append(df_temp)
    except Exception as e:
        st.error(f"Eroare la încărcarea fișierului `{path}`: {e}")
        st.stop()

df = pd.concat(dfs, ignore_index=True)
st.sidebar.markdown(f"Fișiere încărcate: `{', '.join(selected_labels)}`")


def shorten_method(row):
    heuristic = "GBF" if row["heuristic"] == "Greedy" else row["heuristic"]
    fitness = row["fitness"].replace("Fitness", "")
    flags = ""
    if row["SUBSUMPTION"]:
        flags += "Sub"
    if row["VARIANCE_RANDOM"]:
        flags += "Var"
    return f"{heuristic}_{fitness}_{flags}" if flags else f"{heuristic}_{fitness}"


df["method"] = df.apply(shorten_method, axis=1)

limit = st.sidebar.selectbox("Valoare limit analizată", sorted(df['limit'].unique()))
use_log_scale = st.sidebar.checkbox("Scară logaritmică pe axa Y", value=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Evoluție timp vs Număr de fire",
    "Bar chart (min/avg/max)",
    "Boxplot distribuție",
    "Scatter plot timp vs comparatori",
    "Heatmap-uri",
    "Clasament metode (limit=300)",
    "Teste statistice & CI"
])

with tab1:
    st.header(f"Evoluție timp mediu vs Număr de fire (limit={limit}) — Timp în milisecunde")
    subset = df[df["limit"] == limit]
    if subset.empty:
        st.warning("Date indisponibile pentru această valoare de limit.")
    else:
        agg = subset.groupby(["nbWires", "method"]).agg(
            time_mean=('totalTimeMs', 'mean')
        ).reset_index()
        fig = px.line(
            agg, x="nbWires", y="time_mean", color="method", markers=True,
            log_y=use_log_scale, height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg)

with tab2:
    st.header(f"Comparație per metodă (limit={limit}) — Timp în milisecunde")
    subset = df[df["limit"] == limit]
    available_nbwires = sorted(subset["nbWires"].unique())
    selected_nbwires = st.multiselect(
        "Valori număr de fire afișate în grafice",
        available_nbwires,
        default=available_nbwires,
        key="tab2_nbwires"
    )
    filtered_subset = subset[subset["nbWires"].isin(selected_nbwires)]
    if filtered_subset.empty:
        st.warning("Date indisponibile pentru valorile selectate.")
    else:
        agg_all = filtered_subset.groupby(["nbWires", "method"]).agg(
            time_min=('totalTimeMs', 'min'),
            time_mean=('totalTimeMs', 'mean'),
            time_max=('totalTimeMs', 'max')
        ).reset_index()
        chart_min, chart_mean, chart_max = st.tabs(["Minim", "Medie", "Maxim"])
        with chart_min:
            st.subheader("Timp minim per număr de fire (ms)")
            fig_min = px.bar(
                agg_all, x="nbWires", y="time_min", color="method", barmode="group", height=500
            )
            st.plotly_chart(fig_min, use_container_width=True)
        with chart_mean:
            st.subheader("Timp mediu per număr de fire (ms)")
            fig_mean = px.bar(
                agg_all, x="nbWires", y="time_mean", color="method", barmode="group", height=500
            )
            st.plotly_chart(fig_mean, use_container_width=True)
        with chart_max:
            st.subheader("Timp maxim per număr de fire (ms)")
            fig_max = px.bar(
                agg_all, x="nbWires", y="time_max", color="method", barmode="group", height=500
            )
            st.plotly_chart(fig_max, use_container_width=True)
    agg_all_full = subset.groupby(["nbWires", "method"]).agg(
        time_min=('totalTimeMs', 'min'),
        time_mean=('totalTimeMs', 'mean'),
        time_max=('totalTimeMs', 'max')
    ).reset_index()
    st.subheader("Tabele per număr de fire — Timp în milisecunde")
    nbw_values = sorted(agg_all_full['nbWires'].unique())
    tab_nbw = st.tabs([f"nbWires = {nbw}" for nbw in nbw_values])
    for nbw, container in zip(nbw_values, tab_nbw):
        with container:
            st.markdown(f"Tabele pentru număr de fire = {nbw}")
            filtered = agg_all_full[agg_all_full['nbWires'] == nbw]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("Minim")
                st.dataframe(
                    filtered[["method", "time_min"]].sort_values("time_min"),
                    hide_index=True,
                    use_container_width=True
                )
            with col2:
                st.markdown("Medie")
                st.dataframe(
                    filtered[["method", "time_mean"]].sort_values("time_mean"),
                    hide_index=True,
                    use_container_width=True
                )
            with col3:
                st.markdown("Maxim")
                st.dataframe(
                    filtered[["method", "time_max"]].sort_values("time_max"),
                    hide_index=True,
                    use_container_width=True
                )

with tab3:
    st.header(f"Distribuție timp pe metodă (limit={limit}) — Timp în milisecunde")
    nbw_values = sorted(df[df["limit"] == limit]['nbWires'].unique())
    tab_list = st.tabs([f"nbWires = {val}" for val in nbw_values])
    for nbw, container in zip(nbw_values, tab_list):
        with container:
            st.subheader(f"nbWires = {nbw}")
            subset = df[(df["limit"] == limit) & (df["nbWires"] == nbw)]
            if subset.empty:
                st.warning("Date indisponibile pentru această valoare denumăr de fire.")
                continue
            available_methods = sorted(subset["method"].unique())
            selected_methods = st.multiselect(
                f"Metode afișate pentrunumăr de fire = {nbw}",
                available_methods,
                default=available_methods,
                key=f"select_methods_{nbw}"
            )
            subset = subset[subset["method"].isin(selected_methods)]
            if subset.empty:
                st.info("Nicio metodă selectată.")
                continue
            fig = px.box(
                subset,
                y="method",
                x="totalTimeMs",
                points="outliers",
                orientation="h",
                log_x=use_log_scale,
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            stats = subset.groupby('method').agg(
                success_rate=('found', 'mean'),
                time_mean=('totalTimeMs', 'mean'),
                time_median=('totalTimeMs', 'median'),
                time_std=('totalTimeMs', 'std'),
                time_min=('totalTimeMs', 'min'),
                time_max=('totalTimeMs', 'max')
            ).reset_index()
            stats = stats.sort_values(by=['success_rate', 'time_median'], ascending=[False, True])
            st.subheader("Statistici per metodă")
            st.caption("Timp exprimat în milisecunde.")
            st.dataframe(stats)

with tab4:
    st.header(f"Scatter plot: timp vs număr de comparatori (limit={limit})")
    subset = df[df["limit"] == limit]
    if subset.empty:
        st.warning("Date indisponibile pentru această valoare de limit.")
    else:
        fig = px.scatter(
            subset,
            x="nbComparators",
            y="totalTimeMs",
            color="method",
            hover_data=["nbWires", "found"],
            log_y=use_log_scale,
            title="Timp de execuție în funcție de numărul de comparatori",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header(f"Heatmap: Metode vsnumărul de fire (limit={limit})")
    subset = df[df["limit"] == limit]
    if subset.empty:
        st.warning("Date indisponibile pentru această valoare de limit.")
    else:
        available_methods = sorted(df["method"].unique())
        selected_methods = st.multiselect(
            "Metode afișate în heatmap",
            available_methods,
            default=available_methods,
            key="heatmap_methods"
        )
        st.subheader("Timp mediu de execuție — milisecunde")
        apply_log = st.checkbox("Transformare logaritmică (log(1 + timp))", value=True)
        agg_time = subset.groupby(["method", "nbWires"]).agg(
            time_mean=('totalTimeMs', 'mean')
        ).reset_index()
        if selected_methods:
            agg_time = agg_time[agg_time["method"].isin(selected_methods)]
        if apply_log:
            agg_time["value"] = np.log1p(agg_time["time_mean"])
            color_label = "log(1 + timp mediu)"
        else:
            agg_time["value"] = agg_time["time_mean"]
            color_label = "Timp mediu (ms)"
        pivot_time = agg_time.pivot(index="method", columns="nbWires", values="value")
        fig_time = px.imshow(
            pivot_time,
            labels=dict(x="nbWires", y="Metodă", color=color_label),
            aspect="auto",
            color_continuous_scale="Viridis",
            height=900
        )
        st.plotly_chart(fig_time, use_container_width=True)
        if apply_log:
            st.caption("Valorile sunt transformate logaritmic.")
        else:
            st.caption("Valorile reprezintă timpul mediu de execuție în milisecunde.")
        st.markdown("---")
        st.subheader("Rată de succes")
        agg_found = subset.groupby(["method", "nbWires"]).agg(
            success_rate=('found', 'mean')
        ).reset_index()
        if selected_methods:
            agg_found = agg_found[agg_found["method"].isin(selected_methods)]
        pivot_found = agg_found.pivot(index="method", columns="nbWires", values="success_rate")
        fig_found = px.imshow(
            pivot_found,
            labels=dict(x="nbWires", y="Metodă", color="Rată de succes"),
            aspect="auto",
            color_continuous_scale="Blues",
            height=900,
            text_auto=".2f"
        )
        st.plotly_chart(fig_found, use_container_width=True)
        st.caption("Valorile reprezintă proporția de rulări cu succes.")

with tab6:
    st.header("Clasament metode (limit = 300) — Rată de succes 100%")
    subset_300 = df[df["limit"] == 300]
    if subset_300.empty:
        st.warning("Nu există date pentru limit = 300.")
        top_methods_300 = []
    else:
        found_agg_300 = subset_300.groupby(["method", "nbWires"]).agg(
            success_rate=('found', 'mean'),
            mean_time=('totalTimeMs', 'mean')
        ).reset_index()
        method_success_300 = found_agg_300.groupby("method").agg(
            min_success=('success_rate', 'min'),
            total_time=('mean_time', 'sum'),
            mean_time=('mean_time', 'mean')
        ).reset_index()
        filtered_300 = method_success_300[method_success_300["min_success"] == 1]
        top_methods_300 = filtered_300["method"].tolist()
        ranked_300 = filtered_300.sort_values(by="total_time", ascending=True).reset_index(drop=True)
        ranked_300.index += 1
        if ranked_300.empty:
            st.info("Nicio metodă cu rată de succes 100% pentru toate valorile pe care le ia numărul de fire cand avem limt=300.")
        else:
            st.caption("Metodele sunt sortate după suma timpilor medii pe toate valorile numărul de fire.")
            st.dataframe(ranked_300, use_container_width=True)
    st.markdown("---")
    st.header("Clasament metode (limit = 2000) — Filtrat după cele mai bune de la 300")
    subset_2000 = df[df["limit"] == 2000]
    if subset_2000.empty:
        st.warning("Nu există date pentru limit = 2000.")
    else:
        found_agg_2000 = subset_2000.groupby(["method", "nbWires"]).agg(
            success_rate=('found', 'mean'),
            mean_time=('totalTimeMs', 'mean')
        ).reset_index()
        method_agg_2000 = found_agg_2000.groupby("method").agg(
            avg_success=('success_rate', 'mean'),
            total_time=('mean_time', 'sum'),
            mean_time=('mean_time', 'mean')
        ).reset_index()
        filtered_2000 = method_agg_2000[
            (method_agg_2000["avg_success"] > 0) &
            (method_agg_2000["method"].isin(top_methods_300))
            ]
        ranked_2000 = filtered_2000.sort_values(by="total_time", ascending=True).reset_index(drop=True)
        ranked_2000.index += 1
        if ranked_2000.empty:
            st.info("Nicio metodă comună cu cele din topul de la limit = 300.")
        else:
            st.caption("Clasament doar pentru metodele performante de la limit = 300 care au reușit să găsească o rețea de sortare cu 13 fire si limit = 2000.")
            st.dataframe(
                ranked_2000[["method", "avg_success", "total_time", "mean_time"]],
                use_container_width=True
            )

with tab7:
    st.header("Teste statistice (Kruskal–Wallis) + Interval încredere 95%")

    subset = df[df["limit"] == limit]

    st.subheader("Rezultate globale Kruskal–Wallis per număr de fire")
    kruskal_rows = []
    for nbw in sorted(subset["nbWires"].unique()):
        sub_nbw = subset[subset["nbWires"] == nbw]
        methods = sub_nbw["method"].unique()
        if len(methods) < 2:
            continue
        grouped = [grp["totalTimeMs"].values for _, grp in sub_nbw.groupby("method")]
        if all(len(g) >= 2 for g in grouped):
            h_stat, p_val = kruskal(*grouped)
            kruskal_rows.append({
                "nbWires": nbw,
                "p-value": f"{p_val:.2e}",
                "Diferențe semnificative": "Da" if p_val < 0.05 else "Nu"
            })

    if kruskal_rows:
        st.dataframe(pd.DataFrame(kruskal_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Nu există suficiente metode pentru a efectua testele Kruskal–Wallis.")

    st.markdown("---")

    nbwires_vals = sorted(subset["nbWires"].unique())
    selected_nbw = st.selectbox("Selectare număr de fire", nbwires_vals)

    sub = subset[subset["nbWires"] == selected_nbw]
    if sub.empty:
        st.warning("Nu există date pentru această combinație.")
    else:
        st.subheader("Intervale de încredere (95%) pentru media timpului per metodă")
        ci_rows = []
        for method, grp in sub.groupby("method"):
            values = grp["totalTimeMs"].values
            if len(values) < 2:
                continue
            boot_res = bootstrap((values,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
            ci_rows.append({
                "method": method,
                "mean_time": np.mean(values),
                "ci_lower": boot_res.confidence_interval.low,
                "ci_upper": boot_res.confidence_interval.high
            })

        ci_df = pd.DataFrame(ci_rows).sort_values("mean_time")
        st.dataframe(ci_df, hide_index=True, use_container_width=True)

        st.subheader("Comparații 2 câte 2: Top metode identidicate vs. metode care folosesc fitness-ul din articol")
        top_methods = [
            "GBF_OutputSize_Fit",
            "GBF_HammingDistance_SubFit",
            "GBF_HammingDistance_Fit"
        ]

        article_methods = [m for m in sub["method"].unique() if "Article" in m]
        comparison_rows = []
        for top in top_methods:
            for article in article_methods:
                if top == article:
                    continue
                g1 = sub[sub["method"] == top]["totalTimeMs"].values
                g2 = sub[sub["method"] == article]["totalTimeMs"].values
                if len(g1) < 2 or len(g2) < 2:
                    continue
                stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                diff = np.median(g1) - np.median(g2)
                conclusion = "Da" if pval < 0.05 else "Nu"
                comparison_rows.append({
                    "Top method": top,
                    "Article method": article,
                    "Median diff (top - article)": round(diff, 2),
                    "p-value": round(pval, 8),
                    "Diferență semnificativă": conclusion
                })
        if comparison_rows:
            result_df = pd.DataFrame(comparison_rows).sort_values("p-value")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nu sunt suficiente date pentru aceste comparații.")
