"""
1つのテストケースのみ実行
"""

import pandas as pd
from utils import (
    load_color_database,
    find_best_mix_optimized,
    rgb_to_lab,
    lab_to_rgb
)

print("="*60)
print("テスト: グレー(128, 128, 128)")
print("="*60)

target_rgb = (128, 128, 128)
target_lab = rgb_to_lab(*target_rgb)

print(f"目標色 RGB: {target_rgb}")
print(f"目標色 Lab: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}")
print()

db = load_color_database()
mr_color = db[db['manufacturer'] == 'Mr.Color']

print(f"使用可能塗料: {len(mr_color)}色")
print("最適化開始...")
print()

result = find_best_mix_optimized(
    target_lab,
    mr_color,
    max_colors=5,
    exclude_metallic=False,
    exclude_white_black=False,
    thinner_ratio=0.0
)

print("【結果】")
print(f"使用色数: {result['n_colors']}色")
print(f"色差 ΔE: {result['delta_e']:.2f}")
print()
print("【配合レシピ】")
for item in result['recipe']:
    print(f"  {item['code']} {item['name']}: {item['ratio']:.0f}% ({item['grams']}g)")

mixed_rgb = lab_to_rgb(*result['mixed_lab'])
print(f"\n目標色 RGB: {target_rgb}")
print(f"混色結果 RGB: {mixed_rgb}")

if result['n_colors'] >= 2:
    print(f"\n✅ 合格: 2色以上が使用されています")
else:
    print(f"\n❌ 不合格: 1色のみです")
