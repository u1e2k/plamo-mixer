"""
DE00評価およびガンマ設定の検証スクリプト
- グレー(白+黒)系と青系でDE00を評価
- KM_GAMMAの適用を確認
"""

import numpy as np
from utils import calculate_delta_e, kubelka_munk_mix, simple_lab_mix, KM_GAMMA

print("="*70)
print("DE00評価 & ガンマ検証")
print("="*70)
print(f"KM_GAMMA = {KM_GAMMA}")

# ケース1: 白50%+黒50% → 中間グレー近傍（期待Lは単純平均より低い）
white = (92.5, 0.1, 0.2)
black = (15.3, 0.2, 0.1)
colors = [white, black]
ratios = [0.5, 0.5]
expected = (50.0, 0.0, 0.0)

km_res = kubelka_munk_mix(colors, ratios)
simple_res = simple_lab_mix(colors, ratios)

de00_km = calculate_delta_e(expected, km_res, method='DE00')
de00_simple = calculate_delta_e(expected, simple_res, method='DE00')

print("\nケース1: 白50 + 黒50")
print(f"  KM:     L={km_res[0]:.1f}, a={km_res[1]:.1f}, b={km_res[2]:.1f}, DE00={de00_km:.2f}")
print(f"  Hybrid: L={simple_res[0]:.1f}, a={simple_res[1]:.1f}, b={simple_res[2]:.1f}, DE00={de00_simple:.2f}")

# ケース2: 白90%+黒10% → 明るいグレー（単純平均より暗くなる）
ratios2 = [0.9, 0.1]
expected2 = (65.0, 0.0, 0.0)
km_res2 = kubelka_munk_mix(colors, ratios2)
simple_res2 = simple_lab_mix(colors, ratios2)

de00_km2 = calculate_delta_e(expected2, km_res2, method='DE00')
de00_simple2 = calculate_delta_e(expected2, simple_res2, method='DE00')

print("\nケース2: 白90 + 黒10")
print(f"  KM:     L={km_res2[0]:.1f}, a={km_res2[1]:.1f}, b={km_res2[2]:.1f}, DE00={de00_km2:.2f}")
print(f"  Hybrid: L={simple_res2[0]:.1f}, a={simple_res2[1]:.1f}, b={simple_res2[2]:.1f}, DE00={de00_simple2:.2f}")

# ケース3: 赤50%+青50%（紫系）
red = (48.2, 68.4, 45.6)
blue = (32.4, -12.5, -38.6)
colors3 = [red, blue]
ratios3 = [0.5, 0.5]
# 参考期待値は経験ベース
expected3 = (40.0, 28.0, 3.0)
km_res3 = kubelka_munk_mix(colors3, ratios3)
simple_res3 = simple_lab_mix(colors3, ratios3)

de00_km3 = calculate_delta_e(expected3, km_res3, method='DE00')
de00_simple3 = calculate_delta_e(expected3, simple_res3, method='DE00')

print("\nケース3: 赤50 + 青50")
print(f"  KM:     L={km_res3[0]:.1f}, a={km_res3[1]:.1f}, b={km_res3[2]:.1f}, DE00={de00_km3:.2f}")
print(f"  Hybrid: L={simple_res3[0]:.1f}, a={simple_res3[1]:.1f}, b={simple_res3[2]:.1f}, DE00={de00_simple3:.2f}")

print("\n" + "="*70)
print("検証完了")
