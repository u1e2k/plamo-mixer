"""
最適化アルゴリズムのデバッグ

目的関数が正しく動作しているか確認
"""

import pandas as pd
import numpy as np
from utils import (
    load_color_database,
    kubelka_munk_mix,
    calculate_delta_e,
    rgb_to_lab
)


def debug_objective_function():
    """目的関数の動作を確認"""
    print("="*60)
    print("  目的関数デバッグ")
    print("="*60)
    
    # データベース読み込み
    db = load_color_database()
    mr_color = db[db['manufacturer'] == 'Mr.Color']
    
    # グレー(128, 128, 128)を目標に
    target_rgb = (128, 128, 128)
    target_lab = rgb_to_lab(*target_rgb)
    print(f"\n目標色: RGB{target_rgb}")
    print(f"Lab値: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}")
    
    # 全色のLab値を取得
    all_colors_lab = [(row['L'], row['a'], row['b']) for _, row in mr_color.iterrows()]
    n_available = len(all_colors_lab)
    print(f"\n使用可能色数: {n_available}色")
    
    # 白と黒を探す
    white_idx = None
    black_idx = None
    
    for i, (_, row) in enumerate(mr_color.iterrows()):
        if 'ホワイト' in row['name'] or 'White' in row['name'] or row['code'] == 'C62':
            white_idx = i
            print(f"\n白色発見: #{i} {row['code']} {row['name']}")
            print(f"  Lab: L={row['L']:.1f}, a={row['a']:.1f}, b={row['b']:.1f}")
        if 'ブラック' in row['name'] or 'Black' in row['name'] or row['code'] == 'C33':
            black_idx = i
            print(f"\n黒色発見: #{i} {row['code']} {row['name']}")
            print(f"  Lab: L={row['L']:.1f}, a={row['a']:.1f}, b={row['b']:.1f}")
    
    if white_idx is None or black_idx is None:
        print("\n⚠️ 白または黒が見つかりません")
        return
    
    print(f"\n{'='*60}")
    print("テストケース1: 白100%")
    print("="*60)
    
    ratios = np.zeros(n_available)
    ratios[white_idx] = 1.0
    
    used_colors = [all_colors_lab[white_idx]]
    used_ratios = [1.0]
    
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"色差 ΔE: {delta_e:.2f}")
    
    print(f"\n{'='*60}")
    print("テストケース2: 黒100%")
    print("="*60)
    
    ratios = np.zeros(n_available)
    ratios[black_idx] = 1.0
    
    used_colors = [all_colors_lab[black_idx]]
    used_ratios = [1.0]
    
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"色差 ΔE: {delta_e:.2f}")
    
    print(f"\n{'='*60}")
    print("テストケース3: 白50% + 黒50%")
    print("="*60)
    
    used_colors = [all_colors_lab[white_idx], all_colors_lab[black_idx]]
    used_ratios = [0.5, 0.5]
    
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"色差 ΔE: {delta_e:.2f}")
    
    print(f"\n{'='*60}")
    print("テストケース4: 白70% + 黒30%")
    print("="*60)
    
    used_colors = [all_colors_lab[white_idx], all_colors_lab[black_idx]]
    used_ratios = [0.7, 0.3]
    
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"色差 ΔE: {delta_e:.2f}")
    
    print(f"\n{'='*60}")
    print("テストケース5: 白30% + 黒70%")
    print("="*60)
    
    used_colors = [all_colors_lab[white_idx], all_colors_lab[black_idx]]
    used_ratios = [0.3, 0.7]
    
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"色差 ΔE: {delta_e:.2f}")
    
    # 最適比率を探す(グリッドサーチ)
    print(f"\n{'='*60}")
    print("グリッドサーチで最適比率を探索")
    print("="*60)
    
    best_delta_e = float('inf')
    best_ratio = 0
    
    for white_ratio in np.arange(0, 1.01, 0.05):
        black_ratio = 1.0 - white_ratio
        
        used_colors = [all_colors_lab[white_idx], all_colors_lab[black_idx]]
        used_ratios = [white_ratio, black_ratio]
        
        mixed = kubelka_munk_mix(used_colors, used_ratios)
        delta_e = calculate_delta_e(target_lab, mixed)
        
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_ratio = white_ratio
    
    print(f"\n最適配合: 白{best_ratio*100:.0f}% + 黒{(1-best_ratio)*100:.0f}%")
    print(f"最小色差 ΔE: {best_delta_e:.2f}")
    
    # 最適配合の詳細
    used_colors = [all_colors_lab[white_idx], all_colors_lab[black_idx]]
    used_ratios = [best_ratio, 1.0 - best_ratio]
    mixed = kubelka_munk_mix(used_colors, used_ratios)
    
    print(f"混色結果 Lab: L={mixed[0]:.1f}, a={mixed[1]:.1f}, b={mixed[2]:.1f}")
    print(f"目標色 Lab: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}")


if __name__ == "__main__":
    debug_objective_function()
