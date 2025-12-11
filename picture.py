import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格，符合学术论文标准
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'Times New Roman' # 尝试设置为论文常用字体

# 读取处理后的数据
df = pd.read_csv('processed_diabetic_data.csv')

# --- 图表 1: 类别分布 (Class Distribution) ---
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='diabetes2', data=df, palette=['#4e79a7', '#e15759'])
plt.title('Distribution of the Target Variable (Diabetes Type 2)', fontsize=14, fontweight='bold')
plt.xlabel('Diabetes Diagnosis', fontsize=12)
plt.ylabel('Number of Encounters', fontsize=12)
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])

# 在柱子上添加具体数值和百分比
total = len(df)
for p in ax.patches:
    count = p.get_height()

plt.tight_layout()
plt.savefig('Figure1_Class_Distribution.png', dpi=300) # 保存高分辨率图片
plt.show()

# --- 图表 2: 不同年龄段的糖尿病分布 (Age Distribution) ---
plt.figure(figsize=(10, 6))
# 将年龄数字映射回原始标签，为了图表可读性
age_labels = {0: '[0-10)', 1: '[10-20)', 2: '[20-30)', 3: '[30-40)', 4: '[40-50)', 
              5: '[50-60)', 6: '[60-70)', 7: '[70-80)', 8: '[80-90)', 9: '[90-100)'}
df['age_str'] = df['age'].map(age_labels)
# 按年龄排序
order = list(age_labels.values())

sns.countplot(x='age_str', hue='diabetes2', data=df, order=order, palette=['#4e79a7', '#e15759'])
plt.title('Distribution of Encounters by Age and Diabetes Status', fontsize=14, fontweight='bold')
plt.xlabel('Age Group (Years)', fontsize=12)
plt.ylabel('Number of Encounters', fontsize=12)
plt.legend(title='Diabetes Diagnosis', labels=['Negative', 'Positive'])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Figure2_Age_Distribution.png', dpi=300)
plt.show()

# --- 图表 3: 特征相关性热力图 (Correlation Heatmap) ---
plt.figure(figsize=(12, 10))
# 移除临时的 age_str 列，只计算数值列的相关性
correlation_matrix = df.drop(columns=['age_str']).corr()

# 绘制热力图
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) # 只显示下三角，避免重复
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Pearson Correlation Matrix of Clinical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Figure3_Correlation_Heatmap.png', dpi=300)
plt.show()