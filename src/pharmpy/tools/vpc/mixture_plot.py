from pharmpy.deps import altair as alt
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem.table import NONMEMTableFile

# This code can create the ID plot for mixtures with uncertainty from simulations.
# It currently uses the NONMEM phm files directly


def plot_mixture_ids(orig_phm_path, sim_phm_path):
    tab = NONMEMTableFile(orig_phm_path)

    df = tab.tables[0].data_frame
    df = df.set_index(['ID', 'SUBPOP'])
    pmix = df[['PMIX']]

    simtab = NONMEMTableFile(sim_phm_path)

    full = pd.DataFrame()
    for i, tab in enumerate(simtab.tables):
        df = simtab.tables[i].data_frame
        df = df[['ID', 'SUBPOP', 'PMIX']]
        full = pd.concat((full, df))

    minmax = full.groupby(['ID', 'SUBPOP']).agg(
        q5=("PMIX", lambda x: x.quantile(0.05)), q95=("PMIX", lambda x: x.quantile(0.95))
    )

    data = pmix.join(minmax)

    facet = alt.hconcat(title="IPmix with uncertainty 5-95% CI")

    for i in data.index.unique(level="SUBPOP"):
        idvals = data.loc[:, i, :].sort_values(by="PMIX").reset_index()
        idvals["SCALE"] = np.linspace(0, 100, len(idvals))

        chart = (
            alt.Chart(idvals)
            .mark_text()
            .encode(
                x=alt.X("SCALE:Q", axis=alt.Axis(labels=False, ticks=False), title="ID"),
                y=alt.Y("PMIX:Q", title=f"IPmix({i})").scale(domain=[0.0, 1.0]),
                text="ID:N",
            )
            .interactive()
        )

        area = (
            alt.Chart(idvals)
            .mark_area(opacity=0.4)
            .encode(
                alt.X("SCALE:Q"),
                alt.Y("q95:Q"),
                alt.Y2("q5:Q"),
            )
        )

        chart = chart + area

        facet |= chart

    return facet
