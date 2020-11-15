from _function import import_dataset, split_dataset, render_benchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cosmetic options
rc = {"figure.figsize": (6.4, 4.8),
      "figure.dpi": 300,
      "axes.titlesize": "large",
      "axes.titleweight": "bold",
      "axes.titlepad": 12,
      "axes.titlelocation": "left"}

sns.set_theme(context="notebook", style="darkgrid", color_codes=True, rc=rc)

# Exploratory Data Analysis
def uneven_dist():
    """Generate distribution of positive outcomes."""
    bank_year = import_dataset("../data/BankMarketing.csv")
    bank_year["year"] = 2008
    bank_year.loc[27682:, "year"] = 2009
    bank_year.loc[39118:, "year"] = 2010
    p = bank_year[bank_year.y == True].reset_index()
    ax = sns.histplot(data=p, x="index", stat="count",
                      hue="year", bins=500, palette="deep", legend=True)
    ax.get_legend().set_title("")
    ax.set_ylim(0, 60)
    ax.set(xlabel="Index", ylabel="Count")
    fig = ax.get_figure()
    fig.savefig("../docs/figures/1_uneven_distribution.png")


uneven_dist()
