import pandas as pd
import numpy as np

print("开始分析 train.csv 和 test.csv")
print("=" * 60)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("【TRAIN.CSV 分析】")
print(f"数据集大小: {train.shape[0]} 行 × {train.shape[1]} 列")
print()

print("1. 目标变量识别:")
print(f"   - 列名: 'label'")
label_counts = train['label'].value_counts()
print(f"   - 类别数量: {len(label_counts)}")
print(f"   - 类别分布:")
for label, count in label_counts.items():
    percentage = count / len(train) * 100
    print(f"     * {label}: {count} ({percentage:.1f}%)")
print()

print("2. 特征分析:")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
if 'label' in categorical_cols:
    categorical_cols.remove('label')

print(f"   - 数值特征 ({len(numeric_cols)}个): {numeric_cols}")
print(f"   - 分类特征 ({len(categorical_cols)}个): {categorical_cols}")
print()

print("3. 缺失值分析:")
missing_total = train.isnull().sum().sum()
print(f"   - 总缺失值: {missing_total}")
if missing_total > 0:
    missing_cols = train.columns[train.isnull().any()].tolist()
    print(f"   - 有缺失值的列: {missing_cols}")
print()

print("【TEST.CSV 分析】")
print(f"数据集大小: {test.shape[0]} 行 × {test.shape[1]} 列")
print()

print("1. 结构验证:")
print(f"   - 列名一致: {'是' if list(train.columns) == list(test.columns) else '否'}")
print(f"   - 包含label列: {'是' if 'label' in test.columns else '否'}")
print()

print("2. 数据集比例:")
print(f"   - 训练集: {train.shape[0]} 样本")
print(f"   - 测试集: {test.shape[0]} 样本")
print(f"   - 测试集/训练集: {test.shape[0]/train.shape[0]*100:.1f}%")
print()

print("【TABULAR-CLASSIFICATION-REPORT 准备信息】")
print("=" * 60)
print()
print("目标变量: label (二分类: yes/no)")
print("特征分类:")
print("  - 数值特征: aptitude_score, performance_score, stability_index, load_ratio, index_weight, activity_duration, composite_rank")
print("  - 分类特征: geological_era, weather_pattern, region, personal_interest, brew_preference, instrument, species, mineral_type")
print()
print("数据质量: 缺失值极少，类别不平衡(76% no vs 24% yes)")
print()
