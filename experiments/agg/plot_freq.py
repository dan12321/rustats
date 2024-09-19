#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('table_stream/1brc_no_group_real_time_freq.csv')
buckets_name = data.columns[0]
buckets = data[buckets_name]
counts = data["count"]

fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout="constrained")
axs.plot(buckets, counts)


data = pd.read_csv('table_threaded/1brc_4_threads_no_group_real_time_freq.csv')
buckets = data[buckets_name]
counts = data["count"]
axs.plot(buckets, counts, color='red')

data = pd.read_csv('table_threaded/1brc_8_threads_no_group_real_time_freq.csv')
buckets = data[buckets_name]
counts = data["count"]
axs.plot(buckets, counts, color='purple')

axs.set_xlabel(buckets_name)
axs.set_ylabel("count")

plt.show()
