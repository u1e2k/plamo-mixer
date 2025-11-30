"""
最小データセットでテスト: 白と黒だけ
"""

import pandas as pd
from utils import find_best_mix_optimized, rgb_to_lab, lab_to_rgb, calculate_delta_e

# グレーを目標
target_rgb = (128, 128, 128)
target_lab = rgb_to_lab(*target_rgb)

print(f"目標色 RGB: {target_rgb}")
print(f"目標色 Lab: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}\n")

# 白と黒だけのデータセット
colors_data = {
    'code': ['C62', 'C2'],
    'name': ['フラットホワイト', 'ブラック'],
    'manufacturer': ['Mr.Color', 'Mr.Color'],
    'L': [92.5, 15.3],
    'a': [0.1, 0.2],
    'b': [0.2, 0.1],
    'category': ['basic', 'basic']
}

df = pd.DataFrame(colors_data)

print("使用可能塗料:")
print(df[['code', 'name']])
print()

print("最適化開始...")
import time
start = time.time()

result = find_best_mix_optimized(
    target_lab,
    df,
    max_colors=5,
    exclude_metallic=False,
    exclude_white_black=False
)

elapsed = time.time() - start

print(f"最適化完了 ({elapsed:.1f}秒)\n")

print("【結果】")
print(f"使用色数: {result['n_colors']}色")
print(f"色差 ΔE00: {result['delta_e']:.2f}\n")

print("【配合レシピ】")
for item in result['recipe']:
    print(f"  {item['code']} {item['name']}: {item['ratio']:.0f}% ({item['grams']}g)")

mixed_rgb = lab_to_rgb(*result['mixed_lab'])
print(f"\n目標色 RGB: {target_rgb}")
print(f"混色結果 RGB: {mixed_rgb}")

# 参考: DE76での色差も確認
de76 = calculate_delta_e(target_lab, result['mixed_lab'], method='DE76')
print(f"参考(ΔE76): {de76:.2f}")

if result['n_colors'] >= 2:
    print(f"\n✅ 合格!")
else:
    print(f"\n❌ 不合格: 1色のみです")
