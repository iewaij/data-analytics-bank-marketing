from _function import import_dataset, split_dataset, render_benchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cosmetic options
rc = {
    "figure.figsize": (6.4, 4.8),
    "figure.dpi": 300,
    "axes.titlesize": "large",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.titlelocation": "left",
}

sns.set_theme(context="notebook", style="darkgrid", color_codes=True, rc=rc)


def cat_outcome(df, feature):
    df = df.copy()
    if pd.api.types.is_categorical_dtype(df[feature]) and df[feature].isna().sum() > 0:
        df[feature] = df[feature].cat.add_categories("unknown")
        df[feature] = df[feature].fillna("unknown")
    title = feature.title().replace("_", " ").replace("Of", "of")
    f, axs = plt.subplots(1, 2,  figsize=(10, 6), sharey=True,
                          gridspec_kw=dict(wspace=0.04, width_ratios=[5, 2]))
    ax0 = df["y"].groupby(df[feature],
                          dropna=False).value_counts(normalize=True).unstack().plot.barh(xlabel="",
                                                                                         legend=False, stacked=True, ax=axs[0])
    ax1 = df["y"].groupby(df[feature],
                          dropna=False).value_counts().unstack().plot.barh(xlabel="", legend=False,
                                                                           stacked=True, ax=axs[1])


#
#
#
#
#
#
#
#
#
bank_mkt = import_dataset("data/BankMarketing.csv")
#
#
#
#
#
#
#

corr_heatmap = sns.heatmap(data=bank_mkt.corr(method="pearson"))
plt.savefig("docs/figures/2_17_Heatmap.png", bbox_inches='tight')
#
#
#
#
#
# ,bbox_inches='tight'
