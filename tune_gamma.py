"""
ガンマ値の最適化スクリプト
実際の混色データに最も近いガンマ値を探索
"""

import numpy as np
from utils import load_color_database, calculate_delta_e, lab_to_rgb


def km_mix_with_gamma(colors_lab, ratios, gamma):
    """指定したガンマ値でK-M混色を計算"""
    ratios = np.array(ratios) / sum(ratios)
    colors_lab = np.array(colors_lab)
    
    # 明度計算
    reflectances = (colors_lab[:, 0] / 100.0) ** gamma
    epsilon = 1e-6
    reflectances = np.clip(reflectances, epsilon, 1.0 - epsilon)
    k_s_ratios = (1 - reflectances) ** 2 / (2 * reflectances)
    mixed_k_s = np.sum(k_s_ratios * ratios)
    mixed_R = 1 + mixed_k_s - np.sqrt(mixed_k_s**2 + 2*mixed_k_s)
    mixed_R = np.clip(mixed_R, 0.0, 1.0)
    mixed_L = (mixed_R ** (1/gamma)) * 100
    
    # 彩度計算(簡易)
    mixed_a = np.sum(colors_lab[:, 1] * ratios)
    mixed_b = np.sum(colors_lab[:, 2] * ratios)
    
    return (float(mixed_L), float(mixed_a), float(mixed_b))


# 実測データ(文献値や実験値)
# 形式: (色1_Lab, 色2_Lab, 配合比1, 配合比2, 実測結果_Lab)
test_cases = [
    # 白+黒のグラデーション(経験則: 白90%+黒10% ≈ L60-65)
    {
        'name': '白90% + 黒10%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.9, 0.1],
        'expected': (65, 0, 0),  # 経験則
        'weight': 1.0
    },
    # 白+黒(50:50) → やや暗めのグレー(経験則: L40-45)
    {
        'name': '白50% + 黒50%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.5, 0.5],
        'expected': (42, 0, 0),
        'weight': 1.0
    },
    # 白+黒(10:90) → かなり暗い(経験則: L20-25)
    {
        'name': '白10% + 黒90%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.1, 0.9],
        'expected': (22, 0, 0),
        'weight': 1.0
    },
]


print("=" * 70)
print("ガンマ値の最適化")
print("=" * 70)
print("\nテストケース:")
for tc in test_cases:
    print(f"  - {tc['name']}: 期待値 L={tc['expected'][0]}")

print("\n" + "-" * 70)
print(f"{'ガンマ値':<10} {'平均ΔE':<12} {'詳細':<50}")
print("-" * 70)

best_gamma = None
best_error = float('inf')

# ガンマ値を0.5〜3.0の範囲で探索
for gamma in np.linspace(0.5, 3.0, 26):
    total_error = 0
    total_weight = 0
    details = []
    
    for tc in test_cases:
        result = km_mix_with_gamma(tc['colors'], tc['ratios'], gamma)
        error = calculate_delta_e(tc['expected'], result)
        total_error += error * tc['weight']
        total_weight += tc['weight']
        details.append(f"L={result[0]:.0f}(ΔE={error:.1f})")
    
    avg_error = total_error / total_weight
    
    # 結果表示
    marker = ""
    if avg_error < best_error:
        best_error = avg_error
        best_gamma = gamma
        marker = " ← 最良"
    
    print(f"{gamma:<10.2f} {avg_error:<12.2f} {' / '.join(details):<50}{marker}")

print("-" * 70)
print(f"\n✅ 最適ガンマ値: {best_gamma:.2f}")
print(f"   平均誤差: ΔE = {best_error:.2f}")

print("\n" + "=" * 70)
print("推奨値の検証")
print("=" * 70)

for tc in test_cases:
    result = km_mix_with_gamma(tc['colors'], tc['ratios'], best_gamma)
    print(f"\n{tc['name']}")
    print(f"  期待値: L={tc['expected'][0]}")
    print(f"  計算値: L={result[0]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result)}")
    print(f"  ΔE: {calculate_delta_e(tc['expected'], result):.2f}")

print("\n" + "=" * 70)
print(f"utils.pyのガンマ値を {best_gamma:.2f} に変更することを推奨")
print("=" * 70)
