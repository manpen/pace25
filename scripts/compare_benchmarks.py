import pandas as pd
import argparse as cli

parser = cli.ArgumentParser()
parser.add_argument("base", type=str)
parser.add_argument("new", type=str)

args = parser.parse_args()

base = pd.read_csv(args.base)
data = pd.read_csv(args.new)

gr = {}
for _, row in base.iterrows():
    gr[row["name"]] = [
        int(row["read_time_ms"]),
        int(row["num_iter"]),
        int(row["domset_size"]),
        int(row["iter_time_ms"]) / int(row["num_iter"]),
    ]

print(gr)

data["read_su"] = data.apply(
    lambda row: int(gr[row["name"]][0]) / int(row["read_time_ms"]), axis=1
)
data["iter_su"] = data.apply(
    lambda row: int(gr[row["name"]][1]) / int(row["num_iter"]), axis=1
)
data["domset_su"] = data.apply(
    lambda row: int(gr[row["name"]][2]) / int(row["domset_size"]), axis=1
)
data["round_su"] = data.apply(
    lambda row: int(gr[row["name"]][3]) / (int(row["iter_time_ms"]) / int(row["num_iter"])),
    axis=1,
)


def x(n, v):
    if v == 0.0:
        return

    if v < 1.0:
        print(n, " was ", 1.0 / v, " times slower")
    else:
        print(n, " was ", v, "times faster")


x("Median[ReadTime]", data["read_su"].median())
x("Mean[ReadTime]", data["read_su"].mean())
x("Median[Iterations]", 1 / data["iter_su"].median())
x("Mean[Iterations]", 1 / data["iter_su"].mean())
x("Median[DomSize]", data["domset_su"].median())
x("Mean[DomSize]", data["domset_su"].mean())
x("Median[RoundTime]", data["round_su"].median())
x("Mean[RoundTime]", data["round_su"].mean())
