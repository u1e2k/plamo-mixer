"""
ガンマ最適化のpytestテスト

Kubelka-Munk理論に基づく混色計算の精度を検証
"""

import pytest
import numpy as np
from tune_gamma import (
    km_mix_with_gamma,
    calculate_delta_e_simple,
    find_optimal_gamma,
    TEST_CASES
)
from utils import (
    kubelka_munk_mix,
    simple_lab_mix,
    calculate_delta_e,
    lab_to_rgb,
    rgb_to_lab,
    OPTIMAL_GAMMA
)


class TestGammaFunctions:
    """ガンマ関連関数のテスト"""
    
    def test_km_mix_with_gamma_single_color(self):
        """単色の場合はそのまま返される"""
        white = (92.5, 0, 0)
        result = km_mix_with_gamma([white], [1.0], gamma=2.0)
        
        assert result[0] == pytest.approx(92.5, abs=0.1)
        assert result[1] == pytest.approx(0, abs=0.1)
        assert result[2] == pytest.approx(0, abs=0.1)
    
    def test_km_mix_with_gamma_equal_ratio(self):
        """等量混合のテスト"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        result = km_mix_with_gamma([white, black], [0.5, 0.5], gamma=2.0)
        
        # 中間グレーより暗くなるべき（減法混色）
        linear_avg = (92.5 + 15.3) / 2  # 53.9
        assert result[0] < linear_avg
        assert result[0] > black[0]
    
    def test_km_mix_gamma_effect(self):
        """ガンマ値の効果テスト - 高いガンマ = 暗い結果"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        result_low = km_mix_with_gamma([white, black], [0.5, 0.5], gamma=1.0)
        result_high = km_mix_with_gamma([white, black], [0.5, 0.5], gamma=3.0)
        
        # 高いガンマ値は暗い結果を生む
        assert result_high[0] < result_low[0]
    
    def test_calculate_delta_e_identical(self):
        """同じ色のΔEは0"""
        lab = (50, 10, -10)
        assert calculate_delta_e_simple(lab, lab) == pytest.approx(0, abs=1e-6)
    
    def test_calculate_delta_e_different(self):
        """異なる色のΔEは正"""
        lab1 = (50, 0, 0)
        lab2 = (60, 0, 0)
        
        delta_e = calculate_delta_e_simple(lab1, lab2)
        assert delta_e == pytest.approx(10.0, abs=0.01)


class TestGammaOptimization:
    """ガンマ最適化のテスト"""
    
    def test_find_optimal_gamma_returns_valid_range(self):
        """最適ガンマ値が有効範囲内"""
        best_gamma, best_error = find_optimal_gamma(TEST_CASES)
        
        assert 0.5 <= best_gamma <= 3.0
        assert best_error >= 0
    
    def test_find_optimal_gamma_improves_error(self):
        """最適化により誤差が改善される"""
        # 極端なガンマ値での誤差
        extreme_gamma = 5.0
        total_error = 0
        for tc in TEST_CASES:
            result = km_mix_with_gamma(tc['colors'], tc['ratios'], extreme_gamma)
            total_error += calculate_delta_e_simple(tc['expected'], result)
        avg_error_extreme = total_error / len(TEST_CASES)
        
        # 最適化されたガンマ値での誤差
        best_gamma, best_error = find_optimal_gamma(TEST_CASES)
        
        assert best_error < avg_error_extreme
    
    def test_optimal_gamma_consistency(self):
        """最適化結果が一貫している"""
        gamma1, error1 = find_optimal_gamma(TEST_CASES)
        gamma2, error2 = find_optimal_gamma(TEST_CASES)
        
        assert gamma1 == gamma2
        assert error1 == error2


class TestKubelkaMunkMix:
    """Kubelka-Munk混色関数のテスト"""
    
    def test_kubelka_munk_mix_uses_optimal_gamma(self):
        """デフォルトでOPTIMAL_GAMMAが使用される"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        # 明示的にガンマを指定
        result_explicit = kubelka_munk_mix([white, black], [0.5, 0.5], gamma=OPTIMAL_GAMMA)
        # デフォルト
        result_default = kubelka_munk_mix([white, black], [0.5, 0.5])
        
        assert result_explicit[0] == pytest.approx(result_default[0], abs=0.01)
    
    def test_kubelka_munk_mix_custom_gamma(self):
        """カスタムガンマ値が使用される"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        result_gamma_1 = kubelka_munk_mix([white, black], [0.5, 0.5], gamma=1.0)
        result_gamma_3 = kubelka_munk_mix([white, black], [0.5, 0.5], gamma=3.0)
        
        assert result_gamma_1[0] != result_gamma_3[0]
    
    def test_kubelka_munk_mix_empty_ratios(self):
        """空の配合比率の処理"""
        result = kubelka_munk_mix([(50, 0, 0)], [0])
        assert result == (50, 0, 0)


class TestSimpleLabMix:
    """simple_lab_mix関数のテスト"""
    
    def test_simple_lab_mix_uses_optimal_gamma(self):
        """デフォルトでOPTIMAL_GAMMAが使用される"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        result_explicit = simple_lab_mix([white, black], [0.5, 0.5], gamma=OPTIMAL_GAMMA)
        result_default = simple_lab_mix([white, black], [0.5, 0.5])
        
        assert result_explicit[0] == pytest.approx(result_default[0], abs=0.01)
    
    def test_simple_lab_mix_hybrid_approach(self):
        """ハイブリッドアプローチのテスト"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        # simple_lab_mixはK-Mと線形の補間
        result = simple_lab_mix([white, black], [0.5, 0.5])
        km_result = kubelka_munk_mix([white, black], [0.5, 0.5])
        linear_avg = (92.5 + 15.3) / 2
        
        # 結果はK-Mと線形の間にあるべき
        assert min(km_result[0], linear_avg) <= result[0] <= max(km_result[0], linear_avg)


class TestColorConversions:
    """色変換関数のテスト"""
    
    def test_lab_to_rgb_white(self):
        """白のLab→RGB変換"""
        rgb = lab_to_rgb(100, 0, 0)
        # colormathライブラリの丸め誤差を考慮
        assert rgb[0] >= 254
        assert rgb[1] >= 254
        assert rgb[2] >= 254
    
    def test_lab_to_rgb_black(self):
        """黒のLab→RGB変換"""
        rgb = lab_to_rgb(0, 0, 0)
        assert rgb[0] == 0
        assert rgb[1] == 0
        assert rgb[2] == 0
    
    def test_rgb_to_lab_and_back(self):
        """RGB→Lab→RGB往復変換"""
        # グレースケールでテスト（色変換の丸め誤差が少ない）
        original_rgb = (128, 128, 128)
        lab = rgb_to_lab(*original_rgb)
        recovered_rgb = lab_to_rgb(*lab)
        
        # グレーは往復変換で誤差が少ない
        assert abs(recovered_rgb[0] - original_rgb[0]) <= 1
        assert abs(recovered_rgb[1] - original_rgb[1]) <= 1
        assert abs(recovered_rgb[2] - original_rgb[2]) <= 1
    
    def test_rgb_to_lab_gray(self):
        """グレーのRGB→Lab変換"""
        # グレースケールはa*,b*がほぼ0になる
        lab = rgb_to_lab(128, 128, 128)
        
        assert abs(lab[1]) < 1  # a* ≈ 0
        assert abs(lab[2]) < 1  # b* ≈ 0


class TestDeltaE:
    """色差計算のテスト"""
    
    def test_delta_e_symmetric(self):
        """ΔEの対称性"""
        lab1 = (50, 10, -10)
        lab2 = (60, 5, -5)
        
        assert calculate_delta_e(lab1, lab2) == pytest.approx(
            calculate_delta_e(lab2, lab1), abs=0.001
        )
    
    def test_delta_e_triangle_inequality(self):
        """三角不等式"""
        lab1 = (30, 0, 0)
        lab2 = (50, 0, 0)
        lab3 = (70, 0, 0)
        
        d12 = calculate_delta_e(lab1, lab2)
        d23 = calculate_delta_e(lab2, lab3)
        d13 = calculate_delta_e(lab1, lab3)
        
        assert d13 <= d12 + d23 + 0.001  # 数値誤差を考慮


class TestIntegration:
    """統合テスト"""
    
    def test_gray_mixing_produces_valid_result(self):
        """グレー混色が有効な結果を生成"""
        white = (92.5, 0, 0)
        black = (15.3, 0, 0)
        
        result = kubelka_munk_mix([white, black], [0.5, 0.5])
        
        # L値は白と黒の間にあるべき
        assert black[0] <= result[0] <= white[0]
        
        # a, bはほぼ0のまま
        assert abs(result[1]) < 1
        assert abs(result[2]) < 1
    
    def test_color_mixing_reduces_saturation(self):
        """補色混合で彩度が低下"""
        red = (48.2, 68.4, 45.6)
        cyan = (48.2, -68.4, -45.6)  # 赤の補色
        
        result = kubelka_munk_mix([red, cyan], [0.5, 0.5])
        
        # 彩度（a*とb*）が減少
        original_chroma = np.sqrt(red[1]**2 + red[2]**2)
        result_chroma = np.sqrt(result[1]**2 + result[2]**2)
        
        assert result_chroma < original_chroma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
