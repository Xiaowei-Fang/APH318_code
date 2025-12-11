import pandas as pd
import numpy as np
import re # 导入正则表达式库，用于处理复杂的diabetes2规则

def process_diabetic_data(input_filepath='diabetic_data.csv', output_filepath='processed_diabetic_data.csv'):
    """
    读取原始的diabetic_data.csv文件，根据项目说明进行清洗和特征工程，
    最终生成一个包含12个特征的、可用于机器学习的干净数据集。
    """
    print("开始处理数据...")

    # =========================================================================
    # 第一步: 准备工作 - 定义所有规则和映射
    # =========================================================================
    
    # 定义人口统计学特征的编码映射
    race_map = {'Caucasian': 1, 'AfricanAmerican': 2, 'Asian': 3, 'Hispanic': 4, 'Other': 5}
    gender_map = {'Male': 1, 'Female': 0}
    age_map = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }

    # 定义ICD-9代码与新特征的映射关系
    # 将项目文档表格中的所有ICD9代码作为字符串列表存储
    ICD9_MAP = {
        'alcohol': ['303', '305.0', '571.0', '571.1', '571.2', '571.3'],
        'blood_pressure': ['401', '402', '403', '404', '405', '642'],
        'cholesterol': ['272'],
        'heart_disease': ['410', '411', '412', '413', '414', '428'],
        'obesity': ['278'],
        'pregnancy': [str(i) for i in range(630, 650)], # 630-649的所有编码
        'uric_acid': ['274', '790.6'],
        'blurred_vision': ['368']
    }

    # =========================================================================
    # 第二步: 数据加载与初步清洗
    # =========================================================================
    
    # 加载数据集
    df = pd.read_csv(input_filepath)
    
    # 将所有的'?'替换为NumPy的NaN（Not a Number），便于统一处理缺失值
    df.replace('?', np.nan, inplace=True)
    
    # 清洗无效性别记录：根据项目说明，只保留Male和Female
    # 'Unknown/Invalid' 在原始数据中存在，需要移除
    df = df[df['gender'].isin(['Male', 'Female'])]
    
    # 删除关键列中包含缺失值的行，这些行无法用于生成特征
    # 我们关注的列是 'race', 'gender', 'age' 和三个诊断列
    critical_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3']
    df.dropna(subset=critical_cols, inplace=True)
    
    # =========================================================================
    # 第三步: 特征工程 - 生成12个目标特征
    # =========================================================================
    
    # 创建一个新的DataFrame来存储处理后的数据
    processed_df = pd.DataFrame()
    
    # 1. 处理人口统计学特征
    processed_df['race'] = df['race'].map(race_map)
    processed_df['gender'] = df['gender'].map(gender_map)
    processed_df['age'] = df['age'].map(age_map)
    
    # 2. 循环处理基于ICD-9代码的二元特征
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for feature_name, icd9_codes in ICD9_MAP.items():
        # 创建一个初始全为False的Series
        has_disease = pd.Series([False] * len(df), index=df.index)
        
        # 遍历三个诊断列
        for col in diag_cols:
            # 检查当前诊断列中的代码是否以任何一个目标ICD9代码开头
            # | (或) 操作符表示只要在任一诊断列中找到，就标记为True
            has_disease = has_disease | df[col].str.startswith(tuple(icd9_codes))
        
        # 将布尔值(True/False)转换为整数(1/0)
        processed_df[feature_name] = has_disease.astype(int)
        
    # 3. 处理目标变量 'diabetes2' (逻辑最复杂)
    # 规则A: ICD9 code = 250.x0 or 250.x2
    # 使用正则表达式匹配以'250.'开头，且小数点后第二位是0或2的模式
    # 例如：250.10, 250.32
    regex_rule_A = r'^250\.\d[02]'
    condition_A = (df['diag_1'].str.match(regex_rule_A, na=False) |
                   df['diag_2'].str.match(regex_rule_A, na=False) |
                   df['diag_3'].str.match(regex_rule_A, na=False))

    # 规则B: ICD9 code=250 or 250.x AND 至少服用一种指定药物
    # 首先检查诊断码是否以 '250' 开头
    is_diag_250 = (df['diag_1'].str.startswith('250', na=False) |
                   df['diag_2'].str.startswith('250', na=False) |
                   df['diag_3'].str.startswith('250', na=False))
    
    # 接着检查是否服用了三种药物中的任何一种
    is_medicated = ((df['metformin'] != 'No') |
                    (df['glimepiride'] != 'No') |
                    (df['glipizide'] != 'No'))
    
    condition_B = is_diag_250 & is_medicated
    
    # 最终条件：满足规则A或规则B
    final_condition = condition_A | condition_B
    processed_df['diabetes2'] = final_condition.astype(int)
    
    # =========================================================================
    # 第四步: 生成并保存最终数据集
    # =========================================================================
    
    # 确保列的顺序与项目说明一致
    final_columns_order = [
        'race', 'gender', 'age', 'alcohol', 'blood_pressure', 'cholesterol',
        'heart_disease', 'obesity', 'pregnancy', 'uric_acid', 'blurred_vision',
        'diabetes2'
    ]
    processed_df = processed_df[final_columns_order]
    
    # 将处理好的数据保存到新的CSV文件，不包含索引列
    processed_df.to_csv(output_filepath, index=False)
    
    print(f"数据处理完成！共处理 {len(df)} 行有效数据。")
    print(f"结果已保存至: {output_filepath}")
    
    return processed_df

# --- 执行函数 ---
if __name__ == '__main__':
    # 你需要确保 'diabetic_data.csv' 文件与此脚本在同一目录下
    # 或者提供完整的文件路径
    final_data = process_diabetic_data()
    
    # 打印输出结果的前5行以供检查
    print("\n处理后数据的前5行:")
    print(final_data.head())
    
    # 打印每列中1（是）和0（否）的分布情况，用于检查逻辑
    print("\n各特征分布情况:")
    for col in final_data.columns:
        print(f"--- {col} ---")
        print(final_data[col].value_counts())