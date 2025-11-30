"""
レシピが1色しか出ない問題の検証テスト

明らかに2色以上が必要なケースで正しく混色レシピが出るか確認
"""

import pandas as pd
from utils import (
    load_color_database,
    find_best_mix_optimized,
    rgb_to_lab,
    lab_to_rgb,
    calculate_delta_e
)


def test_case(name, target_rgb, expected_min_colors=2):
    """テストケースを実行"""
    print(f"\n{'='*60}")
    print(f"テスト: {name}")
    print(f"{'='*60}")
    
    # RGB→Lab変換
    target_lab = rgb_to_lab(*target_rgb)
    print(f"目標色 RGB: {target_rgb}")
    print(f"目標色 Lab: L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}")
    
    # データベース読み込み
    db = load_color_database()
    
    # Mr.Colorのみで計算
    mr_color = db[db['manufacturer'] == 'Mr.Color']
    
    # 最適化
    result = find_best_mix_optimized(
        target_lab,
        mr_color,
        max_colors=5,
        exclude_metallic=False,
        exclude_white_black=False,
        thinner_ratio=0.0
    )
    
    # 結果表示
    print(f"\n【結果】")
    print(f"使用色数: {result['n_colors']}色")
    print(f"色差 ΔE: {result['delta_e']:.2f}")
    print(f"\n【配合レシピ】")
    for item in result['recipe']:
        print(f"  {item['code']} {item['name']}: {item['ratio']:.0f}% ({item['grams']}g)")
    
    # 混色結果のRGB
    mixed_rgb = lab_to_rgb(*result['mixed_lab'])
    print(f"\n目標色 RGB: {target_rgb}")
    print(f"混色結果 RGB: {mixed_rgb}")
    
    # 検証
    if result['n_colors'] >= expected_min_colors:
        print(f"\n✅ 合格: {expected_min_colors}色以上が使用されています")
        return True
    else:
        print(f"\n❌ 不合格: {expected_min_colors}色以上が必要ですが{result['n_colors']}色でした")
        return False


def main():
    """全テストケースを実行"""
    print("="*60)
    print("  混色レシピ複数色テスト")
    print("="*60)
    
    results = []
    
    # テスト1: グレー(白+黒)
    results.append(test_case(
        "グレー(白50% + 黒50%)",
        (128, 128, 128),
        expected_min_colors=2
    ))
    
    # テスト2: 明るいグレー(白80% + 黒20%)
    results.append(test_case(
        "明るいグレー(白80% + 黒20%)",
        (200, 200, 200),
        expected_min_colors=2
    ))
    
    # テスト3: 暗いグレー(白20% + 黒80%)
    results.append(test_case(
        "暗いグレー(白20% + 黒80%)",
        (60, 60, 60),
        expected_min_colors=2
    ))
    
    # テスト4: パープル(赤50% + 青50%)
    results.append(test_case(
        "パープル(赤50% + 青50%)",
        (128, 0, 128),
        expected_min_colors=2
    ))
    
    # テスト5: オレンジ(赤70% + 黄30%)
    results.append(test_case(
        "オレンジ(赤70% + 黄30%)",
        (255, 128, 0),
        expected_min_colors=2
    ))
    
    # テスト6: 緑系(黄50% + 青50%)
    results.append(test_case(
        "緑系(黄50% + 青50%)",
        (100, 150, 80),
        expected_min_colors=2
    ))
    
    # 総括
    print("\n" + "="*60)
    print("  総括")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"合格: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 全テスト合格!")
    else:
        print(f"\n⚠️ {total - passed}件のテストが失敗しました")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
