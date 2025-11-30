"""
混色アルゴリズムの精度検証テスト

既知の混色結果と比較して、アルゴリズムの精度を評価
"""

import pandas as pd
import numpy as np
from utils import (
    load_color_database,
    simple_lab_mix,
    kubelka_munk_mix,
    calculate_delta_e,
    lab_to_rgb
)


def test_white_black_mix():
    """白+黒=グレーのテスト"""
    print("=" * 60)
    print("テスト1: 白50% + 黒50% = 中間グレー")
    print("=" * 60)
    
    db = load_color_database()
    
    # Mr.Color の白と黒を取得
    white = db[db['name'].str.contains('ホワイト', na=False)].iloc[0]
    black = db[db['code'] == 'C2'].iloc[0]  # ブラック
    
    print(f"\n白: {white['code']} {white['name']}")
    print(f"  Lab: L={white['L']:.1f}, a={white['a']:.1f}, b={white['b']:.1f}")
    print(f"黒: {black['code']} {black['name']}")
    print(f"  Lab: L={black['L']:.1f}, a={black['a']:.1f}, b={black['b']:.1f}")
    
    # 期待値: 中間グレー(L≈50, a≈0, b≈0)
    expected_L = 50
    expected_lab = (expected_L, 0, 0)
    
    # 旧実装(単純平均)
    simple_avg = ((white['L'] + black['L']) / 2,
                  (white['a'] + black['a']) / 2,
                  (white['b'] + black['b']) / 2)
    
    # 改良版
    colors = [(white['L'], white['a'], white['b']),
              (black['L'], black['a'], black['b'])]
    ratios = [0.5, 0.5]
    
    result_simple = simple_lab_mix(colors, ratios)
    result_km = kubelka_munk_mix(colors, ratios)
    
    print("\n【結果】")
    print(f"単純平均:     L={simple_avg[0]:.1f}, a={simple_avg[1]:.1f}, b={simple_avg[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*simple_avg)}")
    print(f"  ΔE: {calculate_delta_e(expected_lab, simple_avg):.2f}")
    
    print(f"\n改良版(K-M近似): L={result_simple[0]:.1f}, a={result_simple[1]:.1f}, b={result_simple[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_simple)}")
    print(f"  ΔE: {calculate_delta_e(expected_lab, result_simple):.2f}")
    
    print(f"\nK-M完全版:    L={result_km[0]:.1f}, a={result_km[1]:.1f}, b={result_km[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_km)}")
    print(f"  ΔE: {calculate_delta_e(expected_lab, result_km):.2f}")
    
    print(f"\n期待値:       L={expected_L}, a=0, b=0")
    
    # 実際の塗料では白+黒は単純平均より暗くなる(減法混色の特性)
    print("\n💡 解説:")
    print("  実際の塗料では、白+黒=グレーは単純平均(L≈52)より暗くなります")
    print("  K-M理論では L≈35-40 程度になるのが正常です")


def test_primary_color_mix():
    """赤+青=紫のテスト"""
    print("\n\n" + "=" * 60)
    print("テスト2: 赤50% + 青50% = 紫")
    print("=" * 60)
    
    db = load_color_database()
    
    # Mr.Color のレッドとブルー
    red = db[db['code'] == 'C1'].iloc[0]  # レッド
    blue = db[db['code'] == 'C5'].iloc[0]  # ブルー
    
    print(f"\n赤: {red['code']} {red['name']}")
    print(f"  Lab: L={red['L']:.1f}, a={red['a']:.1f}, b={red['b']:.1f}")
    print(f"青: {blue['code']} {blue['name']}")
    print(f"  Lab: L={blue['L']:.1f}, a={blue['a']:.1f}, b={blue['b']:.1f}")
    
    colors = [(red['L'], red['a'], red['b']),
              (blue['L'], blue['a'], blue['b'])]
    ratios = [0.5, 0.5]
    
    # 単純平均
    simple_avg = ((red['L'] + blue['L']) / 2,
                  (red['a'] + blue['a']) / 2,
                  (red['b'] + blue['b']) / 2)
    
    result_simple = simple_lab_mix(colors, ratios)
    result_km = kubelka_munk_mix(colors, ratios)
    
    print("\n【結果】")
    print(f"単純平均:     L={simple_avg[0]:.1f}, a={simple_avg[1]:.1f}, b={simple_avg[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*simple_avg)}")
    
    print(f"\n改良版(K-M近似): L={result_simple[0]:.1f}, a={result_simple[1]:.1f}, b={result_simple[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_simple)}")
    
    print(f"\nK-M完全版:    L={result_km[0]:.1f}, a={result_km[1]:.1f}, b={result_km[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_km)}")
    
    print("\n💡 解説:")
    print("  赤+青の混色では:")
    print("  - 明度は両者の平均より暗くなる(減法混色)")
    print("  - 彩度は低下する(濁りが発生)")
    print("  - a*は正(赤方向)、b*は負(青方向)が混ざり、紫系になる")


def test_white_dominance():
    """白の支配性テスト: 白90% + 黒10%"""
    print("\n\n" + "=" * 60)
    print("テスト3: 白90% + 黒10% = 明るいグレー")
    print("=" * 60)
    
    db = load_color_database()
    
    white = db[db['name'].str.contains('ホワイト', na=False)].iloc[0]
    black = db[db['code'] == 'C2'].iloc[0]
    
    print(f"\n白: L={white['L']:.1f}")
    print(f"黒: L={black['L']:.1f}")
    
    colors = [(white['L'], white['a'], white['b']),
              (black['L'], black['a'], black['b'])]
    ratios = [0.9, 0.1]
    
    # 単純平均だと L = 0.9*87 + 0.1*15 ≈ 79.8
    simple_avg_L = 0.9 * white['L'] + 0.1 * black['L']
    
    result_simple = simple_lab_mix(colors, ratios)
    result_km = kubelka_munk_mix(colors, ratios)
    
    print("\n【結果】")
    print(f"単純平均:        L={simple_avg_L:.1f} (明るい)")
    print(f"改良版(K-M近似): L={result_simple[0]:.1f}")
    print(f"K-M完全版:       L={result_km[0]:.1f}")
    
    print("\n💡 解説:")
    print("  実際の塗料では、わずか10%の黒で大きく暗くなります")
    print("  これを「黒の支配性」と呼び、減法混色の重要な特性です")
    print(f"  単純平均では L≈{simple_avg_L:.0f} ですが、")
    print(f"  K-M理論では L≈{result_km[0]:.0f} まで暗くなるのが正常です")


def test_three_color_mix():
    """3色混合: 赤+黄+青のテスト"""
    print("\n\n" + "=" * 60)
    print("テスト4: 赤33% + 黄33% + 青33% = 茶色/灰色")
    print("=" * 60)
    
    db = load_color_database()
    
    red = db[db['code'] == 'C1'].iloc[0]
    yellow = db[db['code'] == 'C4'].iloc[0]
    blue = db[db['code'] == 'C5'].iloc[0]
    
    print(f"\n赤:  {red['name']} - L={red['L']:.1f}, a={red['a']:.1f}, b={red['b']:.1f}")
    print(f"黄:  {yellow['name']} - L={yellow['L']:.1f}, a={yellow['a']:.1f}, b={yellow['b']:.1f}")
    print(f"青:  {blue['name']} - L={blue['L']:.1f}, a={blue['a']:.1f}, b={blue['b']:.1f}")
    
    colors = [(red['L'], red['a'], red['b']),
              (yellow['L'], yellow['a'], yellow['b']),
              (blue['L'], blue['a'], blue['b'])]
    ratios = [1/3, 1/3, 1/3]
    
    result_simple = simple_lab_mix(colors, ratios)
    result_km = kubelka_munk_mix(colors, ratios)
    
    print("\n【結果】")
    print(f"改良版(K-M近似): L={result_simple[0]:.1f}, a={result_simple[1]:.1f}, b={result_simple[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_simple)}")
    print(f"  彩度: {np.sqrt(result_simple[1]**2 + result_simple[2]**2):.1f}")
    
    print(f"\nK-M完全版:    L={result_km[0]:.1f}, a={result_km[1]:.1f}, b={result_km[2]:.1f}")
    print(f"  RGB: {lab_to_rgb(*result_km)}")
    print(f"  彩度: {np.sqrt(result_km[1]**2 + result_km[2]**2):.1f}")
    
    print("\n💡 解説:")
    print("  3原色を混ぜると:")
    print("  - 明度が大幅に低下(暗い茶色/灰色)")
    print("  - 彩度が大幅に低下(濁る)")
    print("  - これが「補色混合」の効果です")


def compare_algorithms():
    """全テストケースで新旧アルゴリズムを比較"""
    print("\n\n" + "=" * 60)
    print("総合評価: 改良版とK-M完全版の比較")
    print("=" * 60)
    
    test_cases = [
        ("白+黒(50:50)", [(87, 0, 0), (15, 0, 0)], [0.5, 0.5]),
        ("白+黒(90:10)", [(87, 0, 0), (15, 0, 0)], [0.9, 0.1]),
        ("赤+青(50:50)", [(48.2, 68.4, 45.6), (32.4, -12.5, -38.6)], [0.5, 0.5]),
        ("赤+黄(50:50)", [(48.2, 68.4, 45.6), (85.2, 5.8, 78.3)], [0.5, 0.5]),
    ]
    
    print("\n{:<20} {:>15} {:>15}".format("テストケース", "改良版ΔE", "K-M版ΔE"))
    print("-" * 52)
    
    # 改良のベースライン(期待値は経験的に設定)
    expectations = [
        (50, 0, 0),   # 白+黒 → 中間グレー
        (75, 0, 0),   # 白90%+黒10% → 明るいグレー
        (40, 28, 3),  # 赤+青 → 紫
        (66, 37, 62), # 赤+黄 → オレンジ
    ]
    
    for (name, colors, ratios), expected in zip(test_cases, expectations):
        result_simple = simple_lab_mix(colors, ratios)
        result_km = kubelka_munk_mix(colors, ratios)
        
        delta_simple = calculate_delta_e(expected, result_simple)
        delta_km = calculate_delta_e(expected, result_km)
        
        print(f"{name:<20} {delta_simple:>15.2f} {delta_km:>15.2f}")
    
    print("\n✅ ΔEが小さいほど期待値に近く、精度が高い")


if __name__ == "__main__":
    test_white_black_mix()
    test_primary_color_mix()
    test_white_dominance()
    test_three_color_mix()
    compare_algorithms()
    
    print("\n\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. 実際の塗料で混色実験を行い、実測値と比較")
    print("2. 補正係数を微調整してΔEを最小化")
    print("3. スペクトルデータがあればさらに高精度化可能")
