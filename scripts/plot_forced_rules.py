import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse as cli
import os


def parse_header(path: str) -> int:
    with open(path, "r") as file:
        return int(file.readline().split(" ")[2])

parser = cli.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("graphs", type=str)
parser.add_argument("output", type=str)

args = parser.parse_args()

base_data = pd.read_csv(f"{args.input}/rule_7.csv")

gr = {}
for _, row in base_data.iterrows():
    gr[row["name"]] = [
        parse_header(f"{args.graphs}/{row["name"]}"),
        int(row["num_iter"]),
        int(row["domset_size"]),
    ]


data = []
for i in range(7):
    rdata = pd.read_csv(f"{args.input}/rule_{i}.csv")
    rdata = rdata[rdata.num_iter != "-"]

    rdata["idx"] = rdata.apply(lambda row: int(row["name"][-6:-3]), axis=1)
    rdata["n"] = rdata.apply(lambda row: gr[row["name"]][0], axis=1)
    rdata["frac_iters"] = rdata.apply(
        lambda row: int(row["num_iter"]) / gr[row["name"]][1], axis=1
    )
    rdata["diff_domset"] = rdata.apply(
        lambda row: int(row["domset_size"]) - gr[row["name"]][2], axis=1
    )
    rdata["diff_domset_frac"] = rdata.apply(
        lambda row: int(row["domset_size"]) / gr[row["name"]][2], axis=1
    )
    rdata["rule_index"] = i

    data.append(rdata)

data = pd.concat(data)

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = 12, 7

plot = sns.scatterplot(
    data,
    x="idx",
    y="frac_iters",
    hue="rule_index",
    palette="colorblind",
)

plot.set(
    xlabel=r"\textsc{Input: Index of Instance}",
    ylabel=r"\textsc{Fraction of Number}" "\n" r"\textsc{of Iterations}" "\n" r"\textsc{to None}"
)

handles, labels = plot.get_legend_handles_labels()
labels = [
    r"\textsc{Dms}",
    r"$\textsc{Bfs}_2$",
    r"$\textsc{Bfs}_3$",
    r"$\textsc{Bfs}_4$",
    r"$\textsc{Bfs}^+_2$",
    r"$\textsc{Bfs}^+_3$",
    r"$\textsc{Bfs}^+_4$",
]

plt.legend(handles, labels, ncols=2, title=r"\textsc{ForcedRule}")

plt.savefig(
    f"{args.output}/num_iter.pdf",
    format="pdf",
    bbox_inches="tight"
)



plt.clf()

plot = sns.scatterplot(
    data,
    x="idx",
    y="diff_domset",
    hue="rule_index",
    palette="colorblind",
)

plt.yscale("symlog")

plot.set(
    xlabel=r"\textsc{Input: Index of Instance}",
    ylabel=r"\textsc{Total Difference}" "\n" r"\textsc{of DomsetSize}" "\n" r"\textsc{to None}"
)

handles, labels = plot.get_legend_handles_labels()
labels = [
    r"\textsc{Dms}",
    r"$\textsc{Bfs}_2$",
    r"$\textsc{Bfs}_3$",
    r"$\textsc{Bfs}_4$",
    r"$\textsc{Bfs}^+_2$",
    r"$\textsc{Bfs}^+_3$",
    r"$\textsc{Bfs}^+_4$",
]

plt.legend(handles, labels, ncols=2, title=r"\textsc{ForcedRule}")

plt.savefig(
    f"{args.output}/domset_size.pdf",
    format="pdf",
    bbox_inches="tight"
)


plt.clf()

plot = sns.scatterplot(
    data,
    x="idx",
    y="diff_domset_frac",
    hue="rule_index",
    palette="colorblind",
)

plt.yscale("symlog")

plot.set(
    xlabel=r"\textsc{Input: Index of Instance}",
    ylabel=r"\textsc{Fractional Difference}" "\n" r"\textsc{of DomsetSize}" "\n" r"\textsc{to None}"
)

handles, labels = plot.get_legend_handles_labels()
labels = [
    r"\textsc{Dms}",
    r"$\textsc{Bfs}_2$",
    r"$\textsc{Bfs}_3$",
    r"$\textsc{Bfs}_4$",
    r"$\textsc{Bfs}^+_2$",
    r"$\textsc{Bfs}^+_3$",
    r"$\textsc{Bfs}^+_4$",
]

plt.legend(handles, labels, ncols=2, title=r"\textsc{ForcedRule}")

plt.savefig(
    f"{args.output}/domset_size_frac.pdf",
    format="pdf",
    bbox_inches="tight"
)
