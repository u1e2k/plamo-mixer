"""
ガンマ値の最適化スクリプト
実際の混色データに最も近いガンマ値を探索

Kubelka-Munk理論に基づく最適ガンマ値を算出し、
utils.pyのOPTIMAL_GAMMAと比較・検証を行う
"""

import numpy as np
from typing import List, Tuple, Dict


def km_mix_with_gamma(colors_lab: List[Tuple[float, float, float]], 
                      ratios: List[float], 
                      gamma: float) -> Tuple[float, float, float]:
    """
    指定したガンマ値でK-M混色を計算
    
    Args:
        colors_lab: 色のLab値のリスト
        ratios: 配合比率のリスト
        gamma: ガンマ値
    
    Returns:
        混色結果のLab値
    """
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


def calculate_delta_e_simple(lab1: Tuple[float, float, float], 
                             lab2: Tuple[float, float, float]) -> float:
    """ΔE76（ユークリッド距離）を計算"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return float(np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2))


def find_optimal_gamma(test_cases: List[Dict], 
                       gamma_range: Tuple[float, float] = (0.5, 3.0),
                       n_steps: int = 51) -> Tuple[float, float]:
    """
    テストケースに最適なガンマ値を探索
    
    Args:
        test_cases: テストケースのリスト
        gamma_range: ガンマ探索範囲 (min, max)
        n_steps: 探索ステップ数
    
    Returns:
        (最適ガンマ値, 平均誤差)
    """
    best_gamma = None
    best_error = float('inf')
    
    for gamma in np.linspace(gamma_range[0], gamma_range[1], n_steps):
        total_error = 0
        total_weight = 0
        
        for tc in test_cases:
            result = km_mix_with_gamma(tc['colors'], tc['ratios'], gamma)
            error = calculate_delta_e_simple(tc['expected'], result)
            total_error += error * tc['weight']
            total_weight += tc['weight']
        
        avg_error = total_error / total_weight
        
        if avg_error < best_error:
            best_error = avg_error
            best_gamma = gamma
    
    return best_gamma, best_error


# 実測データ(塗料混色の経験則と文献値に基づく)
# 白(L=92.5)と黒(L=15.3)の混色
# 参考: 塗料メーカーの調色データより
TEST_CASES = [
    # 白+黒のグラデーション
    # 塗料混色では黒の影響が強く、線形より暗くなる傾向がある
    {
        'name': '白95% + 黒5%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.95, 0.05],
        'expected': (75, 0, 0),  # やや暗め
        'weight': 1.0
    },
    {
        'name': '白90% + 黒10%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.9, 0.1],
        'expected': (65, 0, 0),  # 明るいグレー
        'weight': 1.0
    },
    {
        'name': '白80% + 黒20%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.8, 0.2],
        'expected': (55, 0, 0),  # ライトグレー
        'weight': 1.0
    },
    {
        'name': '白50% + 黒50%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.5, 0.5],
        'expected': (40, 0, 0),  # 中間グレー（やや暗め）
        'weight': 1.0
    },
    {
        'name': '白20% + 黒80%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.2, 0.8],
        'expected': (25, 0, 0),  # ダークグレー
        'weight': 1.0
    },
    {
        'name': '白10% + 黒90%',
        'colors': [(92.5, 0, 0), (15.3, 0, 0)],
        'ratios': [0.1, 0.9],
        'expected': (20, 0, 0),  # かなり暗い
        'weight': 1.0
    },
]


if __name__ == "__main__":
    from utils import lab_to_rgb, OPTIMAL_GAMMA
    
    print("=" * 70)
    print("ガンマ値の最適化")
    print("=" * 70)
    print("\nテストケース:")
    for tc in TEST_CASES:
        print(f"  - {tc['name']}: 期待値 L={tc['expected'][0]}")
    
    print("\n" + "-" * 70)
    print(f"{'ガンマ値':<10} {'平均ΔE':<12} {'詳細'}")
    print("-" * 70)
    
    best_gamma = None
    best_error = float('inf')
    
    # ガンマ値を0.5〜3.0の範囲で探索
    for gamma in np.linspace(0.5, 3.0, 26):
        total_error = 0
        total_weight = 0
        details = []
        
        for tc in TEST_CASES:
            result = km_mix_with_gamma(tc['colors'], tc['ratios'], gamma)
            error = calculate_delta_e_simple(tc['expected'], result)
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
        
        print(f"{gamma:<10.2f} {avg_error:<12.2f} {' / '.join(details[:3])}...{marker}")
    
    print("-" * 70)
    print(f"\n✅ 最適ガンマ値: {best_gamma:.2f}")
    print(f"   平均誤差: ΔE = {best_error:.2f}")
    print(f"\n📌 現在のOPTIMAL_GAMMA: {OPTIMAL_GAMMA}")
    
    print("\n" + "=" * 70)
    print("推奨値の検証")
    print("=" * 70)
    
    for tc in TEST_CASES:
        result = km_mix_with_gamma(tc['colors'], tc['ratios'], best_gamma)
        print(f"\n{tc['name']}")
        print(f"  期待値: L={tc['expected'][0]}")
        print(f"  計算値: L={result[0]:.1f}")
        print(f"  RGB: {lab_to_rgb(*result)}")
        print(f"  ΔE: {calculate_delta_e_simple(tc['expected'], result):.2f}")
    
    print("\n" + "=" * 70)
    if abs(best_gamma - OPTIMAL_GAMMA) < 0.1:
        print(f"✅ 現在のOPTIMAL_GAMMA ({OPTIMAL_GAMMA}) は最適値に近い")
    else:
        print(f"⚠️ utils.pyのOPTIMAL_GAMMAを {best_gamma:.2f} に変更することを検討")
    print("=" * 70)
