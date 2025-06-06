import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('csv/v1_results_final.csv')
df1['version'] = 'v1'
df2 = pd.read_csv('csv/v2_results.csv')
df2['version'] = 'v2'

df = pd.concat([df1, df2], ignore_index=True)

ns = sorted(df['n'].unique())
indices = list(range(len(ns)))

data_v1_et = [df1[df1['n'] == n]['execution_time'].values for n in ns]
data_v2_et = [df2[df2['n'] == n]['execution_time'].values for n in ns]

plt.figure(figsize=(8, 5))
pos_v1 = [i - 0.2 for i in indices]
pos_v2 = [i + 0.2 for i in indices]
plt.boxplot(data_v1_et, positions=pos_v1, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))
plt.boxplot(data_v2_et, positions=pos_v2, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='green'))
plt.xticks(indices, ns)
plt.xlabel('n')
plt.ylabel('execution_time (seconds)')
plt.title('Comparing the distribution of execution times between v1 and v2')
plt.legend([plt.Rectangle((0,0),1,1, facecolor='lightblue'),
            plt.Rectangle((0,0),1,1, facecolor='lightgreen')],
           ['v1', 'v2'], loc='upper left')
plt.tight_layout()
plt.show()

data_v1_df = [df1[df1['n'] == n]['depth_found'].values for n in ns]
data_v2_df = [df2[df2['n'] == n]['depth_found'].values for n in ns]

plt.figure(figsize=(8, 5))
plt.boxplot(data_v1_df, positions=pos_v1, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red'))
plt.boxplot(data_v2_df, positions=pos_v2, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightyellow'), medianprops=dict(color='orange'))
plt.xticks(indices, ns)
plt.xlabel('n')
plt.ylabel('depth_found')
plt.title('Comparing the depth distribution found between v1 and v2')
plt.legend([plt.Rectangle((0,0),1,1, facecolor='lightcoral'),
            plt.Rectangle((0,0),1,1, facecolor='lightyellow')],
           ['v1', 'v2'], loc='upper left')
plt.tight_layout()
plt.show()

data_v1_cf = [df1[df1['n'] == n]['comparators_found'].values for n in ns]
data_v2_cf = [df2[df2['n'] == n]['comparators_found'].values for n in ns]

plt.figure(figsize=(8, 5))
plt.boxplot(data_v1_cf, positions=pos_v1, widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))
plt.boxplot(data_v2_cf, positions=pos_v2, widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='green'))
plt.xticks(indices, ns)
plt.xlabel('n')
plt.ylabel('comparators_found')
plt.title('Comparing the distribution of the number of comparators between v1 and v2')
plt.legend([plt.Rectangle((0,0),1,1, facecolor='lightblue'),
            plt.Rectangle((0,0),1,1, facecolor='lightgreen')],
           ['v1', 'v2'], loc='upper left')
plt.tight_layout()
plt.show()
