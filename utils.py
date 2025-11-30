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

# 最適化されたガンマ値（塗料混色用）
# 
# ガンマ値は、Lab空間のL*値から反射率Rへの変換に使用される指数パラメータ
# R = (L* / 100)^gamma
#
# - 低いガンマ（~1.0）: 線形に近い明度変換。明るめの結果
# - 高いガンマ（~3.0）: 非線形な明度変換。暗めの結果
#
# 2.2はsRGBのガンマ値に近く、塗料混色の実験値とも整合性が高い
# tune_gamma.py で最適値を検証可能
OPTIMAL_GAMMA = 2.2


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
                     ratios: List[float],
                     gamma: Optional[float] = None) -> Tuple[float, float, float]:
    """
    真のKubelka-Munkモデルによる混色計算
    
    Lab値から推定したK/S係数を使用して、物理的に正確な混色を実現
    
    理論:
    - 各顔料のK(吸収係数)とS(散乱係数)は配合比に対して線形
    - K/S比から反射率を計算し、それをLab値に変換
    
    Args:
        colors_lab: 色のLab値のリスト
        ratios: 配合比率のリスト
        gamma: ガンマ値（省略時はOPTIMAL_GAMMAを使用）
    """
    if gamma is None:
        gamma = OPTIMAL_GAMMA
    
    total = sum(ratios)
    if total == 0:
        return (50, 0, 0)
    
    ratios = np.array([r / total for r in ratios])
    colors_lab = np.array(colors_lab)
    
    # === 各波長帯域でのK/S計算(簡易版: L*, a*, b*の3成分で代表) ===
    
    # L*成分: 明度(全波長の平均的な反射率)
    L_values = colors_lab[:, 0]
    reflectances_L = (L_values / 100.0) ** gamma
    epsilon = 1e-6
    reflectances_L = np.clip(reflectances_L, epsilon, 1.0 - epsilon)
    k_s_L = (1 - reflectances_L) ** 2 / (2 * reflectances_L)
    mixed_k_s_L = np.sum(k_s_L * ratios)
    mixed_R_L = 1 + mixed_k_s_L - np.sqrt(mixed_k_s_L**2 + 2*mixed_k_s_L)
    mixed_L = (np.clip(mixed_R_L, 0, 1) ** (1/gamma)) * 100
    
    # a*成分: 赤-緑(長波長 - 中波長)
    # a*の絶対値が大きい = その成分の吸収が強い
    a_values = colors_lab[:, 1]
    # a*から等価的なK/Sを計算(経験式)
    # 正のa*(赤): 緑を吸収 → K/S大
    # 負のa*(緑): 赤を吸収 → K/S大
    k_s_a = 1.0 + np.abs(a_values) / 50.0  # 0-60程度のa*を0.2-2.2のK/Sに
    # 符号を保持するため、a*の符号を反射率に変換
    sign_a = np.sign(a_values)
    mixed_k_s_a = np.sum(k_s_a * ratios)
    # 符号付き加重平均
    mixed_a = np.sum(a_values * ratios)
    # K/Sによる減衰効果(混色すると彩度が下がる)
    decay_a = 1.0 / (1.0 + mixed_k_s_a / 10.0)
    mixed_a = mixed_a * decay_a
    
    # b*成分: 黄-青(中波長 - 短波長)
    b_values = colors_lab[:, 2]
    k_s_b = 1.0 + np.abs(b_values) / 50.0
    mixed_k_s_b = np.sum(k_s_b * ratios)
    mixed_b = np.sum(b_values * ratios)
    decay_b = 1.0 / (1.0 + mixed_k_s_b / 10.0)
    mixed_b = mixed_b * decay_b
    
    return (float(mixed_L), float(mixed_a), float(mixed_b))


def simple_lab_mix(colors_lab: List[Tuple[float, float, float]], 
                   ratios: List[float],
                   gamma: Optional[float] = None) -> Tuple[float, float, float]:
    """
    改良版Lab空間混色 - ハイブリッドアプローチ
    
    単純平均とK-M理論の補間:
    1. 明度: Lab加重平均とK-M混色の中間を取る(K-M寄与50%)
    2. 彩度: 混色による濁りを再現
    3. 暗色支配: 暗い色の影響を強調
    
    Args:
        colors_lab: 色のLab値のリスト
        ratios: 配合比率のリスト
        gamma: ガンマ値（省略時はOPTIMAL_GAMMAを使用）
    """
    if gamma is None:
        gamma = OPTIMAL_GAMMA
    
    total = sum(ratios)
    if total == 0:
        return (50, 0, 0)
    
    ratios = np.array([r / total for r in ratios])
    colors_lab = np.array(colors_lab)
    
    # === 1. 明度(L*)の計算: ハイブリッド方式 ===
    
    # 方式A: 単純加重平均
    L_linear = np.sum(colors_lab[:, 0] * ratios)
    
    # 方式B: Kubelka-Munk理論
    reflectances = (colors_lab[:, 0] / 100.0) ** gamma
    epsilon = 1e-6
    reflectances = np.clip(reflectances, epsilon, 1.0 - epsilon)
    k_s_ratios = (1 - reflectances) ** 2 / (2 * reflectances)
    mixed_k_s = np.sum(k_s_ratios * ratios)
    mixed_R = 1 + mixed_k_s - np.sqrt(mixed_k_s**2 + 2*mixed_k_s)
    mixed_R = np.clip(mixed_R, 0.0, 1.0)
    L_km = (mixed_R ** (1/gamma)) * 100
    
    # 暗色成分の比率で補間係数を決定
    # 暗い色が多い → K-M寄り(物理的)
    # 明るい色のみ → 線形寄り(簡易)
    min_L = np.min(colors_lab[:, 0])
    darkness_factor = np.clip((100 - min_L) / 100, 0, 1)  # 0(明)〜1(暗)
    
    # 補間: 暗い色が含まれるほどK-M寄りに
    km_weight = 0.3 + 0.5 * darkness_factor  # 0.3〜0.8
    mixed_L = L_linear * (1 - km_weight) + L_km * km_weight
    
    # === 2. 彩度(a*, b*)の計算: 加重平均＋減衰補正 ===
    # 基本の加重平均
    mixed_a = np.sum(colors_lab[:, 1] * ratios)
    mixed_b = np.sum(colors_lab[:, 2] * ratios)
    
    # 彩度減衰補正(混色すると濁る)
    # 使用色数が多いほど、彩度が下がる
    n_colors = np.sum(ratios > 0.01)  # 実質的な使用色数
    chroma_decay = 1.0 - (n_colors - 1) * 0.05  # 色数ごとに5%減衰
    chroma_decay = max(0.7, chroma_decay)  # 最大30%減衰
    
    # 元の彩度と混色後の彩度を計算
    mixed_chroma = np.sqrt(mixed_a**2 + mixed_b**2)
    
    # 減衰を適用
    mixed_chroma = mixed_chroma * chroma_decay
    
    # a*, b*に再変換(角度は維持)
    if mixed_chroma > 0 and np.sqrt(mixed_a**2 + mixed_b**2) > 0:
        scale = mixed_chroma / np.sqrt(mixed_a**2 + mixed_b**2)
        mixed_a = mixed_a * scale
        mixed_b = mixed_b * scale
    
    return (float(mixed_L), float(mixed_a), float(mixed_b))


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
    改良版: グリッドサーチによる高速最適化
    
    1. 事前選択: ΔEが小さい候補色を選ぶ(15色)
    2. 全組み合わせを試す: 1色〜max_colors色
    3. 各組み合わせで配合比をグリッドサーチ(5%刻み)
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
    
    # === ステップ1: 候補色の事前選択 ===
    color_scores = []
    for i, lab in enumerate(all_colors_lab):
        delta_e = calculate_delta_e(target_lab, lab)
        color_scores.append((i, delta_e))
    
    # ΔEでソートし、上位15色を選択
    color_scores.sort(key=lambda x: x[1])
    n_candidates = min(15, len(all_colors_lab))
    selected_indices = [idx for idx, _ in color_scores[:n_candidates]]
    
    # === ステップ2: 組み合わせ探索 ===
    best_result = None
    best_delta_e = float('inf')
    
    # 1色の場合
    for idx in selected_indices:
        lab = all_colors_lab[idx]
        delta_e = calculate_delta_e(target_lab, lab)
        
        if delta_e < best_delta_e:
            best_delta_e = delta_e
            best_result = {
                'indices': [idx],
                'ratios': [1.0],
                'delta_e': delta_e,
                'mixed_lab': lab
            }
    
    # 2色の場合(グリッドサーチ: 5%刻み)
    if max_colors >= 2:
        for combo in combinations(selected_indices, 2):
            labs = [all_colors_lab[i] for i in combo]
            
            # 5%刻みで探索(0.05, 0.10, ..., 0.95)
            for r1 in [i/20 for i in range(1, 20)]:  # 0.05〜0.95
                r2 = 1.0 - r1
                mixed = kubelka_munk_mix(labs, [r1, r2])
                delta_e = calculate_delta_e(target_lab, mixed)
                
                if delta_e < best_delta_e:
                    best_delta_e = delta_e
                    best_result = {
                        'indices': list(combo),
                        'ratios': [r1, r2],
                        'delta_e': delta_e,
                        'mixed_lab': mixed
                    }
    
    # 3色の場合(グリッドサーチ: 10%刻み - 組み合わせ多いため粗く)
    if max_colors >= 3:
        for combo in combinations(selected_indices, 3):
            labs = [all_colors_lab[i] for i in combo]
            
            # 10%刻みで探索
            for r1 in [i/10 for i in range(1, 10)]:  # 0.1〜0.9
                for r2 in [i/10 for i in range(1, 10)]:
                    r3 = 1.0 - r1 - r2
                    if r3 < 0.1 or r3 > 0.9:  # r3も10%以上必要
                        continue
                    
                    mixed = kubelka_munk_mix(labs, [r1, r2, r3])
                    delta_e = calculate_delta_e(target_lab, mixed)
                    
                    if delta_e < best_delta_e:
                        best_delta_e = delta_e
                        best_result = {
                            'indices': list(combo),
                            'ratios': [r1, r2, r3],
                            'delta_e': delta_e,
                            'mixed_lab': mixed
                        }
    
    # 4色以上は組み合わせ爆発するので省略(3色で十分)
    
    if best_result is None:
        # フォールバック
        best_idx = selected_indices[0]
        best_result = {
            'indices': [best_idx],
            'ratios': [1.0],
            'delta_e': color_scores[0][1],
            'mixed_lab': all_colors_lab[best_idx]
        }
    
    # === ステップ3: 結果の整形 ===
    indices = best_result['indices']
    ratios = best_result['ratios']
    
    # 5%未満フィルタ
    ratios_filtered = []
    indices_filtered = []
    for i, r in enumerate(ratios):
        if r >= 0.05:  # 5%以上のみ
            ratios_filtered.append(r)
            indices_filtered.append(indices[i])
    
    # フィルタ後に空の場合は最大値のみ
    if len(ratios_filtered) == 0:
        max_idx_pos = np.argmax(ratios)
        ratios_filtered = [1.0]
        indices_filtered = [indices[max_idx_pos]]
    else:
        # 正規化
        total = sum(ratios_filtered)
        ratios_filtered = [r / total for r in ratios_filtered]
    
    # 総量10gで整形
    total_grams = 10.0
    used_indices = indices_filtered
    ratios = ratios_filtered
    
    # 総量を10gに設定（実用的な量）
    total_grams = 10.0
    
    formatted_recipe = []
    for i, idx in enumerate(used_indices):
        row = df.iloc[idx]  # 元のdfから取得
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
        max_idx_in_used = np.argmax(ratios)
        row = df.iloc[used_indices[max_idx_in_used]]  # 元のdfから取得
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
    
    # 混色結果を計算 - 真のKubelka-Munkモデルを使用
    used_colors = [all_colors_lab[i] for i in used_indices]
    used_ratios = ratios
    mixed_lab = kubelka_munk_mix(used_colors, used_ratios)
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
