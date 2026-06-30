import os
import glob
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# Set style for professional research figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

data_dir = "/home/niik/Documents/Study/hpc/data"
artifact_dir = "/home/niik/Documents/Study/hpc/graphics"
os.makedirs(artifact_dir, exist_ok=True)

csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
pattern = re.compile(r"bench_([a-zA-Z0-9]+)_p(\d+)_b(\d+)_gpu_r(\d+)_(.+)_(\d{8}_\d{6})\.csv")

# 1. Parse and extract metrics from all files
records = []
time_series_data = {} # store representative time series for plotting

for filepath in sorted(csv_files):
    filename = os.path.basename(filepath)
    match = pattern.match(filename)
    if not match:
        continue
    model, patch, batch, rank, opt_flags, timestamp = match.groups()
    config_name = f"{model}_p{patch}_b{batch}"
    
    # Read rows
    rows = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for r in reader:
            if not r or len(r) < 4:
                continue
            try:
                rows.append({
                    'ts': r[0],
                    'gpu': float(r[1]),
                    'mem': float(r[2]),
                    'power': float(r[3])
                })
            except ValueError:
                pass
                
    if not rows:
        continue
        
    # Deduplicate timestamps by averaging metrics for duplicated seconds
    ts_groups = defaultdict(list)
    for r in rows:
        ts_groups[r['ts']].append(r)
        
    clean_series = []
    for ts in sorted(ts_groups.keys()):
        g = ts_groups[ts]
        clean_series.append({
            'ts': ts,
            'gpu': sum(x['gpu'] for x in g) / len(g),
            'mem': sum(x['mem'] for x in g) / len(g),
            'pwr': sum(x['power'] for x in g) / len(g)
        })
        
    total_dur_s = len(clean_series)
    max_mem = max(x['mem'] for x in clean_series)
    min_mem = min(x['mem'] for x in clean_series)
    
    # Active training phase threshold
    threshold = max(min_mem + 500, max_mem * 0.85)
    
    # Find active phase start index
    start_idx = 0
    for idx, x in enumerate(clean_series):
        if x['mem'] >= threshold:
            start_idx = idx
            break
            
    active_series = clean_series[start_idx:]
    active_dur_s = len(active_series)
    
    if active_dur_s == 0:
        continue
        
    # Calculate active phase metrics
    avg_gpu = sum(x['gpu'] for x in active_series) / active_dur_s
    max_gpu = max(x['gpu'] for x in active_series)
    avg_mem = sum(x['mem'] for x in active_series) / active_dur_s
    avg_pwr = sum(x['pwr'] for x in active_series) / active_dur_s
    max_pwr = max(x['pwr'] for x in active_series)
    active_energy_j = sum(x['pwr'] for x in active_series)
    
    # Store clean series for time series plotting
    time_series_data[filename] = clean_series
    
    # Opt level maps
    opt_map = {
        'base': 'Level 0 (Base)',
        'sdpa': 'Level 1 (SDPA)',
        'sdpa_compile': 'Level 2 (Compile)',
        'sdpa_compile_fused_adamw': 'Level 3 (Fused AdamW)',
        'sdpa_compile_fused_adamw_dataloader': 'Level 4 (All)'
    }
    opt_label = opt_map.get(opt_flags, opt_flags)
    
    records.append({
        'filename': filename,
        'config': config_name,
        'model': model,
        'opt_flags': opt_flags,
        'opt': opt_label,
        'total_dur_s': total_dur_s,
        'active_dur_s': active_dur_s,
        'avg_gpu': avg_gpu,
        'max_gpu': max_gpu,
        'avg_mem': avg_mem,
        'max_mem': max_mem,
        'avg_pwr': avg_pwr,
        'max_pwr': max_pwr,
        'energy_j': active_energy_j
    })

df_all = pd.DataFrame(records)

# 2. Compute stable metrics per config and opt level by taking the best (minimum active duration) run
# to avoid compilation overhead noise in trials.
grouped_runs = df_all.groupby(['config', 'opt'])

summary_records = []
for (config, opt), group in grouped_runs:
    # Find the run with the minimum active duration (best stable training loop)
    best_run = group.loc[group['active_dur_s'].idxmin()]
    
    summary_records.append({
        'config': config,
        'opt': opt,
        'step_time_ms': (best_run['active_dur_s'] * 1000.0) / 500.0,
        'avg_gpu': best_run['avg_gpu'],
        'max_gpu': best_run['max_gpu'],
        'avg_mem': best_run['avg_mem'],
        'max_mem': best_run['max_mem'],
        'avg_pwr': best_run['avg_pwr'],
        'max_pwr': best_run['max_pwr'],
        'energy_kj': best_run['energy_j'] / 1000.0
    })

df_summary = pd.DataFrame(summary_records)

# Calculate baselines
base_step_times = df_summary[df_summary['opt'] == 'Level 0 (Base)'].set_index('config')['step_time_ms'].to_dict()
base_mems = df_summary[df_summary['opt'] == 'Level 0 (Base)'].set_index('config')['max_mem'].to_dict()
base_energies = df_summary[df_summary['opt'] == 'Level 0 (Base)'].set_index('config')['energy_kj'].to_dict()

# Map config labels
config_labels = {
    'small_p16_b32': 'ViT-Small (b512)',
    'base_p14_b16': 'ViT-Base (b256)',
    'large_p16_b4': 'ViT-Large (b128)'
}
df_summary['config_label'] = df_summary['config'].map(config_labels)

# Order and sort
opt_order = ['Level 0 (Base)', 'Level 1 (SDPA)', 'Level 2 (Compile)', 'Level 3 (Fused AdamW)', 'Level 4 (All)']
df_summary['opt'] = pd.Categorical(df_summary['opt'], categories=opt_order, ordered=True)
df_summary = df_summary.sort_values(['config', 'opt']).reset_index(drop=True)

# Calculate shifted values for step-by-step previous comparison
df_summary['prev_step_time_ms'] = df_summary.groupby('config')['step_time_ms'].shift(1)
df_summary['prev_max_mem'] = df_summary.groupby('config')['max_mem'].shift(1)
df_summary['prev_energy_kj'] = df_summary.groupby('config')['energy_kj'].shift(1)

# Calculate cumulative speedup / memory / energy multipliers (relative to Base)
df_summary['speedup'] = df_summary.apply(lambda row: base_step_times[row['config']] / row['step_time_ms'], axis=1)
df_summary['mem_efficiency'] = df_summary.apply(lambda row: base_mems[row['config']] / row['max_mem'], axis=1)
df_summary['energy_efficiency'] = df_summary.apply(lambda row: base_energies[row['config']] / row['energy_kj'], axis=1)

# Calculate incremental speedup / memory / energy multipliers (relative to previous level)
df_summary['speedup_vs_prev'] = (df_summary['prev_step_time_ms'] / df_summary['step_time_ms']).fillna(1.0)
df_summary['mem_efficiency_vs_prev'] = (df_summary['prev_max_mem'] / df_summary['max_mem']).fillna(1.0)
df_summary['energy_efficiency_vs_prev'] = (df_summary['prev_energy_kj'] / df_summary['energy_kj']).fillna(1.0)

# Write summary markdown table
print("Summary results:")
print(df_summary.to_string())

# Save df_summary to csv for reference
df_summary.to_csv(os.path.join(data_dir, "benchmark_summary.csv"), index=False)

# Let's write the plotting logic
colors = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#EF4444'] # Premium Tailwind HSL colors

def annotate_bars(ax, df, metric_col, raw_col, unit):
    """
    Annotates bars with the cumulative multiplier (height) and raw value (dato duro) in parentheses.
    """
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    hue_labels = opt_order
    for hue_idx, container in enumerate(ax.containers):
        opt_val = hue_labels[hue_idx]
        for x_idx, bar in enumerate(container):
            h = bar.get_height()
            if np.isnan(h) or h == 0:
                continue
            config_lbl = x_labels[x_idx]
            row = df[(df['config_label'] == config_lbl) & (df['opt'] == opt_val)]
            if row.empty:
                continue
            row = row.iloc[0]
            cum_val = row[metric_col]
            raw_val = row[raw_col]
            
            # Format raw value depending on metric type
            if unit == 'ms':
                raw_str = f"{raw_val:.1f} {unit}"
            elif unit == 'MB':
                raw_str = f"{int(raw_val)} {unit}"
            elif unit == 'kJ':
                raw_str = f"{raw_val:.1f} {unit}"
            else:
                raw_str = f"{raw_val} {unit}"
                
            label = f"{cum_val:.2f}x\n({raw_str})"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2., h),
                ha='center', va='bottom',
                xytext=(0, 2),
                textcoords='offset points',
                fontsize=6.5, weight='bold',
                linespacing=0.9
            )

# ----------------- PLOT 1: GPU Speedup comparison (Speed) -----------------
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(
    data=df_summary,
    x='config_label',
    y='speedup',
    hue='opt',
    hue_order=opt_order,
    palette=colors,
    edgecolor='black',
    linewidth=1
)
plt.title("Aceleración de GPU por Nivel de Optimización (Relativo a Base 1.0x)", pad=20, weight='bold')
plt.xlabel("Configuración del Modelo ViT", labelpad=15, weight='bold')
plt.ylabel("Multiplicador de Velocidad (x)", labelpad=15, weight='bold')
ax1.set_ylim(0, df_summary['speedup'].max() * 1.30)
plt.legend(title="Nivel de Optimización", frameon=True, facecolor='white', edgecolor='none')
annotate_bars(ax1, df_summary, 'speedup', 'step_time_ms', 'ms')
plt.tight_layout()
plot1_path = os.path.join(artifact_dir, "gpu_speedup.png")
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved Plot 1 (Speedup) to {plot1_path}")

# ----------------- PLOT 2: GPU VRAM Memory Efficiency -----------------
plt.figure(figsize=(12, 6))
ax2 = sns.barplot(
    data=df_summary,
    x='config_label',
    y='mem_efficiency',
    hue='opt',
    hue_order=opt_order,
    palette=colors,
    edgecolor='black',
    linewidth=1
)
plt.title("Eficiencia de Memoria VRAM por Nivel de Optimización (Relativo a Base 1.0x)", pad=20, weight='bold')
plt.xlabel("Configuración del Modelo ViT", labelpad=15, weight='bold')
plt.ylabel("Multiplicador de Eficiencia de VRAM (x)", labelpad=15, weight='bold')
ax2.set_ylim(0, df_summary['mem_efficiency'].max() * 1.30)
plt.legend(title="Nivel de Optimización", frameon=True, facecolor='white', edgecolor='none')
annotate_bars(ax2, df_summary, 'mem_efficiency', 'max_mem', 'MB')
plt.tight_layout()
plot2_path = os.path.join(artifact_dir, "gpu_memory_efficiency.png")
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Saved Plot 2 (Memory Efficiency) to {plot2_path}")

# ----------------- PLOT 3: GPU Energy Efficiency -----------------
plt.figure(figsize=(12, 6))
ax3 = sns.barplot(
    data=df_summary,
    x='config_label',
    y='energy_efficiency',
    hue='opt',
    hue_order=opt_order,
    palette=colors,
    edgecolor='black',
    linewidth=1
)
plt.title("Eficiencia Energética por Nivel de Optimización (Relativo a Base 1.0x)", pad=20, weight='bold')
plt.xlabel("Configuración del Modelo ViT", labelpad=15, weight='bold')
plt.ylabel("Multiplicador de Eficiencia Energética (x)", labelpad=15, weight='bold')
ax3.set_ylim(0, df_summary['energy_efficiency'].max() * 1.30)
plt.legend(title="Nivel de Optimización", frameon=True, facecolor='white', edgecolor='none')
annotate_bars(ax3, df_summary, 'energy_efficiency', 'energy_kj', 'kJ')
plt.tight_layout()
plot3_path = os.path.join(artifact_dir, "gpu_energy_efficiency.png")
plt.savefig(plot3_path, dpi=300)
plt.close()
print(f"Saved Plot 3 (Energy Efficiency) to {plot3_path}")

print("Plotting script executed successfully!")
