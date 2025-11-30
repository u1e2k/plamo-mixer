#!/usr/bin/env python3
"""
超シンプルな動作確認 - K-Mモデルが動くか
"""

import sys
sys.path.insert(0, '/workspaces/plamo-mixer')

from utils import kubelka_munk_mix, calculate_delta_e

# 白と黒
white_lab = (92.5, 0.1, 0.2)
black_lab = (15.3, 0.2, 0.1)

print("Step 1: 白のみ")
mixed = kubelka_munk_mix([white_lab], [1.0])
print(f"  結果: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")

print("\nStep 2: 黒のみ")
mixed = kubelka_munk_mix([black_lab], [1.0])
print(f"  結果: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")

print("\nStep 3: 白50% + 黒50%")
mixed = kubelka_munk_mix([white_lab, black_lab], [0.5, 0.5])
print(f"  結果: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")

print("\nStep 4: 5種類の比率を試す")
target_L = 53.6  # グレー(128, 128, 128)のL*
for white_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    black_ratio = 1.0 - white_ratio
    mixed = kubelka_munk_mix([white_lab, black_lab], [white_ratio, black_ratio])
    diff = abs(mixed[0] - target_L)
    print(f"  白{white_ratio*100:.0f}%: L={mixed[0]:.1f} (差{diff:.1f})")

print("\n✅ 基本動作OK")
