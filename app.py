"""
PlamoMixer - プラモ塗装専用混色AIツール
ガンプラの指定色を失敗せずに1発で作れる
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from utils import (
    load_color_database,
    load_presets,
    lab_to_rgb,
    rgb_to_lab,
    find_best_mix_optimized,
    format_result_text,
    calculate_delta_e,
    KM_GAMMA
)

# ページ設定
st.set_page_config(
    page_title="PlamoMixer - プラモ混色ツール",
    page_icon="🎨",
    layout="wide"
)

# タイトル
st.title("🎨 PlamoMixer")
st.subheader("ガンプラの指定色、失敗せずに1発で作れる")
st.markdown("---")

# データ読み込み(キャッシュ)
@st.cache_data
def load_data():
    db = load_color_database()
    presets = load_presets()
    return db, presets

try:
    color_db, presets_data = load_data()
except Exception as e:
    st.error(f"データファイルの読み込みに失敗しました: {e}")
    st.stop()

# セッションステート初期化
if 'target_lab' not in st.session_state:
    st.session_state.target_lab = None
if 'result' not in st.session_state:
    st.session_state.result = None

# サイドバー: 設定
st.sidebar.header("⚙️ 設定")
# 色差メソッド選択（DE00既定）
delta_e_method = st.sidebar.selectbox(
    "色差メソッド",
    ["DE00", "DE76"],
    index=0,
    help="DE00は人の知覚により近い評価。DE76はユークリッド距離"
)

# 現在のKMガンマ表示
st.sidebar.markdown(f"**KMガンマ(γ):** {KM_GAMMA}")

# 1. 目標色の選択方法
st.sidebar.subheader("1️⃣ 目標色を選ぶ")
input_method = st.sidebar.radio(
    "選択方法",
    ["プリセットから選ぶ", "写真をアップロード", "16進数で指定"]
)

target_lab = None
target_name = ""

if input_method == "プリセットから選ぶ":
    # カテゴリでフィルタ
    categories = sorted(list(set(p['category'] for p in presets_data['presets'])))
    selected_category = st.sidebar.selectbox("カテゴリ", ["全て"] + categories)
    
    # プリセット一覧
    if selected_category == "全て":
        filtered_presets = presets_data['presets']
    else:
        filtered_presets = [p for p in presets_data['presets'] 
                           if p['category'] == selected_category]
    
    preset_names = [f"{p['name']} ({p['category']})" for p in filtered_presets]
    selected_preset_idx = st.sidebar.selectbox("目標色", range(len(preset_names)), 
                                               format_func=lambda x: preset_names[x])
    
    selected_preset = filtered_presets[selected_preset_idx]
    target_lab = (selected_preset['L'], selected_preset['a'], selected_preset['b'])
    target_name = selected_preset['name']
    
    # プレビュー
    rgb = lab_to_rgb(*target_lab)
    st.sidebar.markdown(f"**プレビュー:** {target_name}")
    st.sidebar.markdown(
        f'<div style="background-color: rgb{rgb}; width: 100%; height: 50px; border: 1px solid #ccc;"></div>',
        unsafe_allow_html=True
    )

elif input_method == "写真をアップロード":
    uploaded_file = st.sidebar.file_uploader("画像ファイルをアップロード", 
                                             type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 画像を読み込んで平均色を計算
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="アップロードされた画像", use_column_width=True)
        
        # RGB平均を計算
        img_array = np.array(image.convert('RGB'))
        avg_color = img_array.mean(axis=(0, 1)).astype(int)
        
        # Lab変換
        target_lab = rgb_to_lab(avg_color[0], avg_color[1], avg_color[2])
        target_name = "写真からの抽出色"
        
        st.sidebar.markdown(f"**抽出された色:**")
        st.sidebar.markdown(
            f'<div style="background-color: rgb({avg_color[0]}, {avg_color[1]}, {avg_color[2]}); width: 100%; height: 50px; border: 1px solid #ccc;"></div>',
            unsafe_allow_html=True
        )
        st.sidebar.markdown(f"RGB: ({avg_color[0]}, {avg_color[1]}, {avg_color[2]})")

elif input_method == "16進数で指定":
    hex_color = st.sidebar.text_input("16進数カラーコード", "#808080")
    
    try:
        # 16進数をRGBに変換
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Lab変換
        target_lab = rgb_to_lab(r, g, b)
        target_name = f"#{hex_color.upper()}"
        
        st.sidebar.markdown(f"**指定された色:**")
        st.sidebar.markdown(
            f'<div style="background-color: #{hex_color}; width: 100%; height: 50px; border: 1px solid #ccc;"></div>',
            unsafe_allow_html=True
        )
    except:
        st.sidebar.error("正しい16進数カラーコードを入力してください")
        target_lab = None

# 2. 手持ち塗料の選択
st.sidebar.markdown("---")
st.sidebar.subheader("2️⃣ 手持ち塗料")

manufacturer_filter = st.sidebar.multiselect(
    "メーカーで絞り込み",
    ["Mr.Color", "ガイアカラー", "タミヤカラー"],
    default=["Mr.Color", "ガイアカラー", "タミヤカラー"]
)

if not manufacturer_filter:
    st.sidebar.warning("少なくとも1つのメーカーを選択してください")
    available_colors = color_db
else:
    available_colors = color_db[color_db['manufacturer'].isin(manufacturer_filter)]

st.sidebar.markdown(f"**使用可能な塗料:** {len(available_colors)}色")

# 詳細フィルタ(オプション)
show_advanced = st.sidebar.checkbox("詳細設定を表示")
if show_advanced:
    # 特定の色を除外
    exclude_categories = st.sidebar.multiselect(
        "除外するカテゴリ",
        ["metallic", "clear", "character"],
        default=[]
    )
    
    if exclude_categories:
        available_colors = available_colors[~available_colors['category'].isin(exclude_categories)]
        st.sidebar.markdown(f"→ {len(available_colors)}色に絞り込み")

# 3. 制約条件
st.sidebar.markdown("---")
st.sidebar.subheader("3️⃣ 制約条件")

max_colors = st.sidebar.select_slider(
    "最大使用色数",
    options=[1, 2, 3, 4, 5],
    value=3
)

exclude_metallic = st.sidebar.checkbox("メタリック色を除外", value=False)
exclude_wb = st.sidebar.checkbox("白・黒・シルバーを除外", value=False)

thinner_ratio = st.sidebar.slider(
    "希釈率(%)",
    min_value=0,
    max_value=50,
    value=0,
    step=5
) / 100.0

# 計算ボタン
st.sidebar.markdown("---")
calculate_button = st.sidebar.button("🔍 最適配合を計算", type="primary", use_container_width=True)

# メインエリア
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 設定内容")
    
    if target_lab:
        st.markdown(f"**目標色:** {target_name}")
        st.markdown(f"**Lab値:** L={target_lab[0]:.1f}, a={target_lab[1]:.1f}, b={target_lab[2]:.1f}")
        
        # 色プレビュー(大きめ)
        rgb = lab_to_rgb(*target_lab)
        st.markdown("**目標色プレビュー:**")
        st.markdown(
            f'<div style="background-color: rgb{rgb}; width: 200px; height: 100px; border: 2px solid #333; border-radius: 5px;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.info("左のサイドバーから目標色を選択してください")
    
    st.markdown(f"**手持ち塗料:** {len(available_colors)}色")
    st.markdown(f"**最大使用色数:** {max_colors}色まで")
    
    if exclude_metallic:
        st.markdown("- メタリック色を除外")
    if exclude_wb:
        st.markdown("- 白・黒・シルバーを除外")
    if thinner_ratio > 0:
        st.markdown(f"- 希釈率 {thinner_ratio*100:.0f}%")

with col2:
    st.header("✨ 計算結果")
    
    if calculate_button:
        if target_lab is None:
            st.error("目標色を選択してください")
        elif len(available_colors) == 0:
            st.error("使用可能な塗料がありません")
        else:
            with st.spinner("最適配合を計算中..."):
                try:
                    result = find_best_mix_optimized(
                        target_lab,
                        available_colors,
                        max_colors=max_colors,
                        exclude_metallic=exclude_metallic,
                        exclude_white_black=exclude_wb,
                        thinner_ratio=thinner_ratio
                    )
                    st.session_state.result = result
                except Exception as e:
                    st.error(f"計算エラー: {e}")
                    result = None
    
    # 結果表示
    if st.session_state.result and target_lab is not None:
        result = st.session_state.result
        
        # 色差評価
        # 色差を選択メソッドに合わせて再計算（最終表示用）
        delta_e = calculate_delta_e(result['target_lab'], result['mixed_lab'], method=delta_e_method)
        if delta_e < 3.0:
            st.success(f"✅ 非常に近い色です (ΔE = {delta_e:.1f})")
        elif delta_e < 6.0:
            st.success(f"✅ 十分近い色です (ΔE = {delta_e:.1f})")
        elif delta_e < 10.0:
            st.info(f"ℹ️ やや差がありますが使用可能 (ΔE = {delta_e:.1f})")
        else:
            st.warning(f"⚠️ 差があります (ΔE = {delta_e:.1f}) - 手持ち塗料を増やすと精度向上")
        
        # 配合レシピ
        st.markdown("### 📝 配合レシピ (合計10g)")
        for item in result['recipe']:
            st.markdown(
                f"**{item['code']}** {item['name']} *({item['manufacturer']})*  \n"
                f"→ **{item['ratio']:.0f}%** ({item['grams']}g)"
            )
        
        # 混色結果プレビュー
        mixed_rgb = lab_to_rgb(*result['mixed_lab'])
        st.markdown("### 🎨 混色結果プレビュー")
        
        col_target, col_mixed = st.columns(2)
        with col_target:
            target_rgb = lab_to_rgb(*target_lab)
            st.markdown("**目標色**")
            st.markdown(
                f'<div style="background-color: rgb{target_rgb}; width: 100%; height: 80px; border: 2px solid #333;"></div>',
                unsafe_allow_html=True
            )
        
        with col_mixed:
            st.markdown("**混色結果**")
            st.markdown(
                f'<div style="background-color: rgb{mixed_rgb}; width: 100%; height: 80px; border: 2px solid #333;"></div>',
                unsafe_allow_html=True
            )
        
        # テキスト出力
        st.markdown("### 📄 テキスト出力")
        # テキスト出力も選択メソッドに追従
        st.code(format_result_text(result, method=delta_e_method), language="text")

        # 補足情報
        st.caption(f"評価メソッド: {delta_e_method} / KMガンマ(γ): {KM_GAMMA}")
    else:
        st.info("「最適配合を計算」ボタンを押してください")

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <p><strong>PlamoMixer</strong> - プラモ塗装専用混色AIツール</p>
    <p>対応塗料: Mr.Color 122色 / ガイアカラー 60色 / タミヤカラー 54色 (全236色)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 使い方ヘルプ(折りたたみ)
with st.expander("📖 使い方"):
    st.markdown("""
    ### 基本的な使い方
    
    1. **目標色を選ぶ**
       - プリセットから選ぶ: 100種類以上の軍用機・戦車・艦船色
       - 写真をアップロード: 実物の写真から色を抽出
       - 16進数で指定: カラーピッカーなどの値を直接入力
    
    2. **手持ち塗料を選ぶ**
       - メーカーで絞り込み可能
       - 236色から必要なものだけを選択
    
    3. **制約条件を設定**
       - 最大使用色数: 混ぜる塗料の数を制限
       - メタリック除外: メタリック色を使わない
       - 白・黒除外: 白・黒・シルバーを使わない
       - 希釈率: シンナーの割合を指定
    
    4. **計算ボタンを押す**
       - 最適な配合比率が0.3秒以内に表示されます
       - ΔE00値が小さいほど目標色に近い
    
    ### ΔE (色差)について
    - **0〜3**: 非常に近い(実用上問題なし)
    - **3〜6**: 十分近い(許容範囲)
    - **6〜10**: やや差がある(用途によって判断)
    - **10以上**: 差が大きい(手持ち塗料を増やすと改善)
    """)
