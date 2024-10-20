"""
Helper for experiment.ipynb
"""

# use pyproject group: experiments_analysis
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import yaml


def load_results(file="../../results.yaml"):
    # Load YAML file
    with open(file) as file:
        data = yaml.safe_load(file)

    # Process metrics per experiment call
    rows = []
    for experiment in data["experiments"]:
        if not experiment["results"]:
            continue
        for result in experiment["results"]:
            if "metrics" not in result:
                continue
            metrics = result["metrics"]
            kwargs = result["kwargs"]
            options = {
                "samplepoints": kwargs.get("samplepoints", 0),
                **{f"option_{k}": v for k, v in kwargs.get("config", {}).items()},
            }
            options.update
            # Annotate metrics with experiment name and date
            row = {
                "experiment": experiment["name"],
                "date": data["date"],
                **metrics,
                **options,
            }
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    df["share_of_weights_loading"] = df["us_per_weights_loading"] / df["us_per_all"]
    return df


def get_data_overview(df, exclude_columns: list[str], quantil=0.95):
    df_filtered = df.drop(columns=exclude_columns)
    metrics = list(df_filtered.keys())

    # Set up the subplots
    fig, axes = plt.subplots(math.ceil(len(metrics) / 2), 2, figsize=(14, 12))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each metric
    for i, metric in enumerate(metrics):
        sns.histplot(
            data=df_filtered[metric].dropna(),
            # x=metric,
            # hue=option,
            # multiple='dodge', # ['layer', 'stack', 'fill', 'dodge']
            ax=axes[i],
            # palette='Set2'
        )
        axes[i].set_title(f"Histogram of {metric}")
        quantile_minus = df[metric].quantile(1 - quantil)
        quantile_plus = df[metric].quantile(quantil)

        if i not in []:  # [0, 4, 7]:
            # Overlay the PDF
            mean = df_filtered[metric].mean()
            std = df_filtered[metric].std()
            xmin, xmax = axes[i].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            pdf = stats.norm.pdf(x, mean, std)
            axes[i].plot(x, pdf, color="red", label="PDF")

            # Add the quantile bars
            axes[i].axvline(
                quantile_plus, color="orange", linestyle="--", label="95th Percentile +"
            )
            axes[i].axvline(
                quantile_minus,
                color="orange",
                linestyle="--",
                label="95th Percentile -",
            )
        else:
            # zoom to the quantile
            axes[i].set_xlim([quantile_minus, quantile_plus])
        axes[i].legend([])

    plt.tight_layout()

    # Create a global legend
    # unique_experiments = df_filtered[option].unique()
    # handles = [plt.Line2D([0], [0], color=sns.color_palette('Set2')[i], lw=4) for i in range(len(unique_experiments))]  # noqa: E501
    # fig.legend(
    #     handles,
    #     unique_experiments,
    #     loc='center',
    #     title='Read Margin',
    #     bbox_to_anchor=(0.52, 0.5)
    #     )  # Adjust location as needed
    plt.suptitle("Effects of the option 'read_margin' on metrics")
    plt.show()
