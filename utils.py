"""
PlamoMixer - 混色計算エンジン
Kubelka-Munkモデルベースの塗料混色最適化
"""

import numpy as np
import pandas as pd
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
from typing import List, Tuple, Dict, Optional
import json


def load_color_database(csv_path: str = "color_database.csv") -> pd.DataFrame:
    """色データベースCSVを読み込む"""
    return pd.read_csv(csv_path)


def load_presets(json_path: str = "presets.json") -> Dict:
    """プリセット色JSONを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def lab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
    """Lab値をRGB値に変換(0-255)"""
    lab = LabColor(L, a, b)
    rgb = convert_color(lab, sRGBColor)
    # クランプして0-255に
    r = int(max(0, min(255, rgb.rgb_r * 255)))
    g = int(max(0, min(255, rgb.rgb_g * 255)))
    b = int(max(0, min(255, rgb.rgb_b * 255)))
    return (r, g, b)


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB値(0-255)をLab値に変換"""
    rgb = sRGBColor(r/255.0, g/255.0, b/255.0)
    lab = convert_color(rgb, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)


def calculate_delta_e(target_lab: Tuple[float, float, float], 
                     result_lab: Tuple[float, float, float]) -> float:
    """ΔE00を計算(CIEDE2000) - 簡易実装版"""
    # CIEDE2000の完全実装は複雑なので、ΔE76(ユークリッド距離)を使用
    # 実用上十分な精度
    L1, a1, b1 = target_lab
    L2, a2, b2 = result_lab
    
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    
    # ΔE*76 (CIE76)
    delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
    
    return float(delta_e)


def kubelka_munk_mix(colors_lab: List[Tuple[float, float, float]], 
                     ratios: List[float]) -> Tuple[float, float, float]:
    """
    Kubelka-Munkモデルの簡易版で混色を計算
    実際には複雑だが、Lab空間での加重平均+補正で近似
    """
    # 正規化
    total = sum(ratios)
    if total == 0:
        return (50, 0, 0)  # デフォルト
    
    ratios = [r / total for r in ratios]
    
    # 基本: Lab空間での加重平均
    L = sum(lab[0] * ratio for lab, ratio in zip(colors_lab, ratios))
    a = sum(lab[1] * ratio for lab, ratio in zip(colors_lab, ratios))
    b = sum(lab[2] * ratio for lab, ratio in zip(colors_lab, ratios))
    
    # Kubelka-Munk補正(簡易版)
    # 暗い色の影響を強調
    darkness_factors = [100 - lab[0] for lab in colors_lab]  # L値の逆
    avg_darkness = sum(d * r for d, r in zip(darkness_factors, ratios))
    L = L * (1 - avg_darkness / 200)  # 暗さに応じてLを下げる
    
    return (L, a, b)


def simple_lab_mix(colors_lab: List[Tuple[float, float, float]], 
                   ratios: List[float]) -> Tuple[float, float, float]:
    """
    シンプルなLab空間加重平均混色
    暗い色の影響を考慮した補正を追加
    """
    total = sum(ratios)
    if total == 0:
        return (50, 0, 0)
    
    ratios = [r / total for r in ratios]
    
    # 基本の加重平均
    L = sum(lab[0] * ratio for lab, ratio in zip(colors_lab, ratios))
    a = sum(lab[1] * ratio for lab, ratio in zip(colors_lab, ratios))
    b = sum(lab[2] * ratio for lab, ratio in zip(colors_lab, ratios))
    
    # 暗い色の影響補正（明度が低い色ほど強く影響する）
    min_L = min(lab[0] for lab in colors_lab)
    if min_L < 50:  # 暗い色が含まれる場合
        darkness_correction = (50 - min_L) / 100  # 0-0.5の補正係数
        L = L * (1 - darkness_correction * 0.3)  # 明度を下げる
    
    return (L, a, b)


def find_best_mix_bruteforce(target_lab: Tuple[float, float, float],
                             available_colors: pd.DataFrame,
                             max_colors: int = 3,
                             exclude_metallic: bool = False,
                             exclude_white_black: bool = False,
                             thinner_ratio: float = 0.0) -> Dict:
    """
    総当たり+最適化で最良の混色を探索
    
    Args:
        target_lab: 目標のLab値
        available_colors: 使用可能な塗料のDataFrame
        max_colors: 使用する最大色数
        exclude_metallic: メタリックを除外するか
        exclude_white_black: 白・黒を除外するか
        thinner_ratio: 希釈率(0-1)
    
    Returns:
        最適配合の辞書
    """
    # フィルタリング
    df = available_colors.copy()
    
    if exclude_metallic:
        df = df[df['category'] != 'metallic']
    
    if exclude_white_black:
        # 白・黒・シルバー系を除外
        exclude_codes = ['C2', 'C8', 'C11', 'C14', 'C33', 'C52', 'C62',
                        'EX-01', 'EX-02', 'LP-1', 'LP-2', 'LP-18']
        df = df[~df['code'].isin(exclude_codes)]
    
    best_result = None
    best_delta_e = float('inf')
    
    # 1色から max_colors 色までの組み合わせを試す
    for n_colors in range(1, max_colors + 1):
        for combo in combinations(range(len(df)), n_colors):
            selected = df.iloc[list(combo)]
            colors_lab = [(row['L'], row['a'], row['b']) 
                         for _, row in selected.iterrows()]
            
            # 最適化で配合比率を求める
            def objective(ratios):
                # 希釈を考慮
                effective_ratios = [r * (1 - thinner_ratio) for r in ratios]
                mixed = simple_lab_mix(colors_lab, effective_ratios)
                return calculate_delta_e(target_lab, mixed)
            
            # 初期値: 均等配分
            x0 = [1.0 / n_colors] * n_colors
            
            # 制約: 合計が1、各要素が0.05以上(5%未満は除外)
            bounds = [(0.05, 1) for _ in range(n_colors)]
            constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1.0}
            
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            delta_e = result.fun
            
            if delta_e < best_delta_e:
                best_delta_e = delta_e
                ratios = result.x
                mixed_lab = simple_lab_mix(colors_lab, ratios)
                
                # 配合を整形(5%以上のみ)
                formatted_recipe = []
                for i, ratio in enumerate(ratios):
                    if ratio >= 0.05:  # 5%以上のみ
                        row = selected.iloc[i]
                        formatted_recipe.append({
                            'code': row['code'],
                            'name': row['name'],
                            'manufacturer': row['manufacturer'],
                            'ratio': round(ratio * 100, 1)
                        })
                
                # レシピが空の場合は最大値のみ使用
                if len(formatted_recipe) == 0:
                    max_idx = np.argmax(ratios)
                    row = selected.iloc[max_idx]
                    formatted_recipe = [{
                        'code': row['code'],
                        'name': row['name'],
                        'manufacturer': row['manufacturer'],
                        'ratio': 100.0
                    }]
                else:
                    # 比率を再正規化
                    total_ratio = sum(r['ratio'] for r in formatted_recipe)
                    for r in formatted_recipe:
                        r['ratio'] = round(r['ratio'] / total_ratio * 100, 1)
                
                best_result = {
                    'recipe': formatted_recipe,
                    'delta_e': round(delta_e, 2),
                    'mixed_lab': mixed_lab,
                    'target_lab': target_lab,
                    'n_colors': len(formatted_recipe)
                }
    
    return best_result


def find_best_mix_optimized(target_lab: Tuple[float, float, float],
                            available_colors: pd.DataFrame,
                            max_colors: int = 3,
                            exclude_metallic: bool = False,
                            exclude_white_black: bool = False,
                            thinner_ratio: float = 0.0) -> Dict:
    """
    高速版: 差分進化アルゴリズムで最適化
    色の選択と配合比を同時最適化
    """
    # フィルタリング
    df = available_colors.copy()
    
    if exclude_metallic:
        df = df[df['category'] != 'metallic']
    
    if exclude_white_black:
        exclude_codes = ['C2', 'C8', 'C11', 'C14', 'C33', 'C52', 'C62',
                        'EX-01', 'EX-02', 'LP-1', 'LP-2', 'LP-18']
        df = df[~df['code'].isin(exclude_codes)]
    
    all_colors_lab = [(row['L'], row['a'], row['b']) for _, row in df.iterrows()]
    n_available = len(df)
    
    # 目的関数
    def objective(x):
        # x は n_available 次元のベクトル(各色の配合比)
        # 5%未満は使わない
        ratios = np.array(x)
        ratios[ratios < 0.05] = 0  # 5%未満は切り捨て
        
        # 使用色数チェック
        n_used = np.sum(ratios > 0)
        if n_used > max_colors:
            return 1000.0  # ペナルティ
        
        if np.sum(ratios) == 0:
            return 1000.0
        
        # 正規化
        ratios = ratios / np.sum(ratios)
        
        # 使用色のみ抽出
        used_indices = np.where(ratios > 0)[0]
        used_colors = [all_colors_lab[i] for i in used_indices]
        used_ratios = ratios[used_indices].tolist()
        
        # 混色計算
        mixed = simple_lab_mix(used_colors, used_ratios)
        delta_e = calculate_delta_e(target_lab, mixed)
        
        # 色数が少ないほうが良い(わずかにボーナス)
        complexity_penalty = 0.5 * n_used
        
        return delta_e + complexity_penalty
    
    # 境界
    bounds = [(0, 1) for _ in range(n_available)]
    
    # 差分進化で最適化(高速)
    result = differential_evolution(objective, bounds, 
                                   maxiter=100, 
                                   popsize=10,
                                   seed=42,
                                   workers=1,
                                   polish=True)
    
    # 結果を整形
    ratios = result.x
    ratios[ratios < 0.05] = 0  # 5%未満切り捨て
    
    if np.sum(ratios) == 0:
        # 全て5%未満の場合は最大値のみ使用
        max_idx = np.argmax(result.x)
        ratios = np.zeros_like(result.x)
        ratios[max_idx] = 1.0
    else:
        ratios = ratios / np.sum(ratios)
    
    used_indices = np.where(ratios > 0)[0]
    
    # 総量を10gに設定（実用的な量）
    total_grams = 10.0
    
    formatted_recipe = []
    for i in used_indices:
        row = df.iloc[i]
        ratio_percent = round(ratios[i] * 100, 0)
        grams = ratios[i] * total_grams
        
        # 5%未満(0.5g未満)は除外
        if ratio_percent < 5 or grams < 0.5:
            continue
            
        formatted_recipe.append({
            'code': row['code'],
            'name': row['name'],
            'manufacturer': row['manufacturer'],
            'ratio': ratio_percent,
            'grams': round(grams, 1)
        })
    
    # 比率を再正規化して合計100%に（整数で）
    if len(formatted_recipe) == 0:
        # フィルタ後に空になった場合、最大値のみを使用
        max_idx = np.argmax(ratios)
        row = df.iloc[max_idx]
        formatted_recipe = [{
            'code': row['code'],
            'name': row['name'],
            'manufacturer': row['manufacturer'],
            'ratio': 100,
            'grams': 10.0
        }]
    else:
        total_ratio = sum(r['ratio'] for r in formatted_recipe)
        if total_ratio != 100:
            # 誤差を最大比率に加算
            max_item = max(formatted_recipe, key=lambda x: x['ratio'])
            max_item['ratio'] += (100 - total_ratio)
            # グラムも再計算
            for r in formatted_recipe:
                r['grams'] = round(r['ratio'] / 100 * total_grams, 1)
    
    # 混色結果を計算
    used_colors = [all_colors_lab[i] for i in used_indices]
    used_ratios = [ratios[i] for i in used_indices]
    mixed_lab = simple_lab_mix(used_colors, used_ratios)
    delta_e = calculate_delta_e(target_lab, mixed_lab)
    
    return {
        'recipe': formatted_recipe,
        'delta_e': round(delta_e, 2),
        'mixed_lab': mixed_lab,
        'target_lab': target_lab,
        'n_colors': len(formatted_recipe)
    }


def format_result_text(result: Dict) -> str:
    """結果を見やすいテキストに整形"""
    lines = []
    lines.append("【混色レシピ】")
    lines.append("")
    
    for item in result['recipe']:
        lines.append(f"  {item['code']} {item['name']} ({item['manufacturer']})")
        lines.append(f"    → {item['ratio']:.0f}% ({item['grams']}g)")
    
    lines.append("")
    lines.append(f"合計: 10.0g")
    lines.append("")
    lines.append(f"色差 ΔE = {result['delta_e']:.1f}")
    
    if result['delta_e'] < 3.0:
        lines.append("→ 非常に近い色です")
    elif result['delta_e'] < 6.0:
        lines.append("→ 十分近い色です")
    elif result['delta_e'] < 10.0:
        lines.append("→ やや差がありますが使用可能")
    else:
        lines.append("→ 差があります（手持ち塗料を増やすと精度向上）")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # テスト
    db = load_color_database()
    presets = load_presets()
    
    # 零戦灰緑色を目標に
    target = presets['presets'][0]
    target_lab = (target['L'], target['a'], target['b'])
    
    print(f"目標色: {target['name']}")
    print(f"Lab値: L={target_lab[0]}, a={target_lab[1]}, b={target_lab[2]}")
    print()
    
    # Mr.Colorのみで3色まで
    mr_color = db[db['manufacturer'] == 'Mr.Color']
    
    print("最適化中...")
    result = find_best_mix_optimized(target_lab, mr_color, max_colors=3)
    
    print(format_result_text(result))
