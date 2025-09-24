"""
Scan label distributions across prediction/window months and plot patterns.

Outputs (under datamart/analysis):
- label_scan.csv: rows of pm, wm, total, positives, negatives, default_rate
- default_rate_heatmap.png: heatmap of default rate by (pm, wm)
- diff_heatmap.png: heatmap of (positives - negatives) normalized by total
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.spark_utils import create_spark_session, stop_spark_session
from utils.gold_utils import create_label_store
from utils.config import SILVER_PATH


def scan_labels(spark, loan_daily, pm_values, wm_values):
    rows = []
    for pm in pm_values:
        for wm in wm_values:
            ls = create_label_store(loan_daily, pm, wm)
            agg = ls.groupBy('label').count()
            dist = {r['label']: r['count'] for r in agg.collect()}
            pos = dist.get(1, 0)
            neg = dist.get(0, 0)
            total = pos + neg
            rate = (pos / total) if total else 0.0
            rows.append({
                'pm': pm,
                'wm': wm,
                'total': total,
                'positives': pos,
                'negatives': neg,
                'default_rate': rate,
                'diff_norm': (pos - neg) / total if total else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    out_dir = os.path.join('datamart', 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    spark = create_spark_session('LabelScan', log_level='ERROR')
    loan_daily = spark.read.parquet(os.path.join(SILVER_PATH, 'loan_daily'))

    pm_values = [0, 1, 2, 3]
    wm_values = [1, 2, 3, 4, 6, 9, 12]

    df = scan_labels(spark, loan_daily, pm_values, wm_values)
    csv_path = os.path.join(out_dir, 'label_scan.csv')
    df.to_csv(csv_path, index=False)

    # Pivot for heatmaps
    rate_pivot = df.pivot(index='pm', columns='wm', values='default_rate')
    diff_pivot = df.pivot(index='pm', columns='wm', values='diff_norm')

    plt.figure(figsize=(10, 6))
    sns.heatmap(rate_pivot, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Default Rate by (Prediction Months, Window Months)')
    plt.ylabel('Prediction Months (pm)')
    plt.xlabel('Window Months (wm)')
    rate_path = os.path.join(out_dir, 'default_rate_heatmap.png')
    plt.tight_layout()
    plt.savefig(rate_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(diff_pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Normalized (Positives - Negatives) by (pm, wm)')
    plt.ylabel('Prediction Months (pm)')
    plt.xlabel('Window Months (wm)')
    diff_path = os.path.join(out_dir, 'diff_heatmap.png')
    plt.tight_layout()
    plt.savefig(diff_path)
    plt.close()

    print(f"Saved: {csv_path}\nSaved: {rate_path}\nSaved: {diff_path}")

    stop_spark_session(spark)


if __name__ == '__main__':
    main()
