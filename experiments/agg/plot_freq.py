#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('table_load_then_parse/1brc_real_time_freq.csv')
buckets_name = data.columns[0]
buckets = data[buckets_name]
counts = data["count"]

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout="constrained")
axs.plot(buckets, counts)


data = pd.read_csv('table_load_then_parse/1brc_sorted_real_time_freq.csv')
buckets = data[buckets_name]
counts = data["count"]

axs.plot(buckets, counts, color='red')
axs.set_xlabel(buckets_name)
axs.set_ylabel("count")

plt.show()
