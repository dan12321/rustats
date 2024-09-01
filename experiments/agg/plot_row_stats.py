#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import math

data = pd.read_csv('table_load_then_parse/max_rss_mem_stats.csv')
rows = data["rows"]
avg = data["mean"]

rows = [math.log10(r) for r in rows]

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout="constrained")
axs.plot(rows, avg)
axs.set_xlabel("log_rows")
axs.set_ylabel("avg_max_rss_mem")

plt.show()
