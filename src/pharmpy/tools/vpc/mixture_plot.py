from pharmpy.deps import altair as alt
from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem.table import NONMEMTableFile

# This code can create the ID plot for mixtures with uncertainty from simulations.
# It currently uses the NONMEM phm files directly


def plot_mixture_ids(orig_phm_path, sim_phm_path):
    tab = NONMEMTableFile(orig_phm_path)

    df = tab.tables[0].data_frame
    pmix = df[['ID', 'SUBPOP', 'PMIX']]
    pmix = pmix.sort_values(["SUBPOP", "PMIX"])
    subpops = pmix['SUBPOP'].unique()
    ids = pmix['ID'].unique()
    pmix = pmix.assign(rank=list(range(1, len(ids) + 1)) * len(subpops))
    pmix = pmix.set_index(["SUBPOP", "rank"])

    simtab = NONMEMTableFile(sim_phm_path)

    full = pd.DataFrame()
    for i, tab in enumerate(simtab.tables):
        df = simtab.tables[i].data_frame
        df = df[['ID', 'SUBPOP', 'PMIX']]
        df = df.assign(sim=i)
        full = pd.concat((full, df))

    full = full.sort_values(["sim", "SUBPOP", "PMIX"])
    full = full.assign(rank=list(range(1, len(ids) + 1)) * len(subpops) * len(simtab.tables))

    minmax = full.groupby(['SUBPOP', 'rank']).agg(
        q5=("PMIX", lambda x: x.quantile(0.05)), q95=("PMIX", lambda x: x.quantile(0.95))
    )

    data = pmix.join(minmax)

    facet = alt.hconcat(title="IPmix with uncertainty 5-95% CI")

    for i in subpops:
        df = data.loc[i].reset_index()

        chart = (
            alt.Chart(df)
            .mark_text()
            .encode(
                x=alt.X("rank:Q", axis=alt.Axis(labels=False, ticks=False), title="ID"),
                y=alt.Y("PMIX:Q", title=f"IPmix({i})").scale(domain=[0.0, 1.0]),
                text="ID:N",
            )
            .interactive()
        )

        area = (
            alt.Chart(df)
            .mark_area(opacity=0.7)
            .encode(
                alt.X("rank:Q"),
                alt.Y("q95:Q"),
                alt.Y2("q5:Q"),
            )
        )

        chart = chart + area

        facet |= chart

    return facet
