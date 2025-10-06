"""
Analyze Label Store Distribution
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName('LabelDistribution').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

print('='*80)
print('LABEL STORE DISTRIBUTION ANALYSIS')
print('='*80)

# Load gold layer (contains labels)
label_store = spark.read.parquet('datamart/gold/features')

print(f'\nTotal records: {label_store.count():,}')
print(f'Unique customers: {label_store.select("Customer_ID").distinct().count():,}')
print(f'Unique loans: {label_store.select("loan_id").distinct().count():,}')

print('\n--- LABEL DISTRIBUTION ---')
distribution = label_store.groupBy('label').count().orderBy('label')
print('\n+-----+-----+')
print('|label|count|')
print('+-----+-----+')

dist_list = distribution.collect()
total = sum([row['count'] for row in dist_list])

for row in dist_list:
    print(f'|  {row["label"]}  |{row["count"]:5d}|')
print('+-----+-----+')

print('\nDetailed Breakdown:')
for row in dist_list:
    label = row['label']
    count = row['count']
    pct = (count / total * 100) if total > 0 else 0
    label_name = 'No Default' if label == 0 else 'Default'
    print(f'  Label {label} ({label_name:12s}): {count:,} records ({pct:.2f}%)')

# Calculate ratio
no_default = dist_list[0]['count']
default_count = dist_list[1]['count']
ratio = no_default / default_count if default_count > 0 else 0

print(f'\nClass Balance Ratio:')
print(f'  No Default : Default = {ratio:.2f} : 1')
print(f'  ({no_default:,} : {default_count:,})')

print('\n--- SAMPLE RECORDS ---')
print('\nFirst 5 records:')
label_store.show(5, truncate=False)

print('\n--- LABEL DISTRIBUTION BY PREDICTION DATE ---')
date_dist = label_store.groupBy('prediction_date').agg(
    F.count('*').alias('total'),
    F.sum(F.when(F.col('label') == 1, 1).otherwise(0)).alias('defaults')
).orderBy('prediction_date')
print(f'\nRecords by prediction date: {date_dist.count()} unique dates')
date_dist.show(10, truncate=False)

spark.stop()
print('\n' + '='*80)
