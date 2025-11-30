"""
最もシンプルな検証: 白と黒だけで計算
"""

from utils import kubelka_munk_mix, calculate_delta_e, rgb_to_lab, lab_to_rgb

# グレー(128, 128, 128)を目標
target_rgb = (128, 128, 128)
target_lab = rgb_to_lab(*target_rgb)

print(f"目標色 RGB: {target_rgb}")
print(f"目標色 Lab: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}\n")

# 白と黒のLab値(データベースから)
white_lab = (92.5, 0.1, 0.2)  # C62 フラットホワイト
black_lab = (15.3, 0.2, 0.1)  # C2 ブラック

print("="*60)
print("白のみ")
print("="*60)
mixed = kubelka_munk_mix([white_lab], [1.0])
delta_e = calculate_delta_e(target_lab, mixed)
rgb = lab_to_rgb(*mixed)
print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
print(f"混色結果 RGB: {rgb}")
print(f"色差 ΔE: {delta_e:.2f}\n")

print("="*60)
print("黒のみ")
print("="*60)
mixed = kubelka_munk_mix([black_lab], [1.0])
delta_e = calculate_delta_e(target_lab, mixed)
rgb = lab_to_rgb(*mixed)
print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
print(f"混色結果 RGB: {rgb}")
print(f"色差 ΔE: {delta_e:.2f}\n")

print("="*60)
print("白50% + 黒50%")
print("="*60)
mixed = kubelka_munk_mix([white_lab, black_lab], [0.5, 0.5])
delta_e = calculate_delta_e(target_lab, mixed)
rgb = lab_to_rgb(*mixed)
print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
print(f"混色結果 RGB: {rgb}")
print(f"色差 ΔE: {delta_e:.2f}\n")

# グリッドサーチで最適値を探す
print("="*60)
print("グリッドサーチ(白0%〜100%、5%刻み)")
print("="*60)

best_delta_e = float('inf')
best_white_ratio = 0

for white_ratio in [i/20 for i in range(21)]:  # 0, 0.05, 0.10, ..., 1.0
    black_ratio = 1.0 - white_ratio
    mixed = kubelka_munk_mix([white_lab, black_lab], [white_ratio, black_ratio])
    delta_e = calculate_delta_e(target_lab, mixed)
    
    if delta_e < best_delta_e:
        best_delta_e = delta_e
        best_white_ratio = white_ratio
        best_mixed = mixed

print(f"最適配合: 白{best_white_ratio*100:.0f}% + 黒{(1-best_white_ratio)*100:.0f}%")
print(f"混色結果 Lab: L={best_mixed[0]:.1f}, a={best_mixed[1]:.1f}, b={best_mixed[2]:.1f}")
print(f"混色結果 RGB: {lab_to_rgb(*best_mixed)}")
print(f"最小色差 ΔE: {best_delta_e:.2f}")

print(f"\n目標色と比較:")
print(f"  目標: L={target_lab[0]:.1f}")
print(f"  結果: L={best_mixed[0]:.1f}")
print(f"  差分: {abs(target_lab[0] - best_mixed[0]):.1f}")
