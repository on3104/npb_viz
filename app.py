import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import base64
from io import BytesIO

# ページ設定
st.set_page_config(
    page_title="野球打球・投球分析ダッシュボード",
    page_icon="⚾",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e75b6;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# データ処理関数
# =====================================================

@st.cache_data
def load_data():
    """データの読み込みと前処理"""
    df = pd.read_csv('2025_gamelog.csv')
    # 打球位置を計算
    df[['BallPositionX', 'BallPositionY']] = df.apply(compute_ball_map, axis=1, result_type='expand')
    # 打球結果を計算
    df['HitResult'] = df.apply(map_hit_result, axis=1)
    # 球種を英語に変換
    df['PitchTypeEN'] = df['PitchType'].apply(convert_pitchtype)
    # 球種カテゴリ（直球/変化球）を追加
    df['PitchCategory'] = df['PitchType'].apply(categorize_pitch)
    return df

def compute_ball_map(row):
    """打球位置を計算（process_data.pyより）"""
    origin = np.array([287, 489])
    try:
        depth = int(row['Depth']) / 10
    except:
        return np.nan, np.nan
    
    unit_vector = np.array([0, 0])
    direction = row['Direction']
    
    direction_vectors = {
        'B': np.array([-258, -276]),
        'Y': np.array([258, -276]),
        'C': np.array([-238, -294]),
        'X': np.array([238, -294]),
        'D': np.array([-216, -311]),
        'W': np.array([216, -311]),
        'E': np.array([-196, -326]),
        'V': np.array([196, -326]),
        'F': np.array([-174, -340]),
        'U': np.array([174, -340]),
        'G': np.array([-152, -353]),
        'T': np.array([152, -353]),
        'H': np.array([-131, -363]),
        'S': np.array([131, -363]),
        'I': np.array([-109, -373]),
        'R': np.array([109, -373]),
        'J': np.array([-85, -382]),
        'Q': np.array([85, -382]),
        'K': np.array([-62, -389]),
        'P': np.array([62, -389]),
        'L': np.array([-39, -393]),
        'O': np.array([39, -393]),
        'M': np.array([-14, -396]),
        'N': np.array([14, -396]),
    }
    
    if direction in direction_vectors:
        unit_vector = direction_vectors[direction]
    else:
        return np.nan, np.nan
    
    position = origin + depth * unit_vector
    return position[0], position[1]

def map_hit_result(row):
    """打球結果をカテゴリに変換"""
    try:
        result = row['AtBatResult']
        if pd.isna(result):
            return None
        
        if any(x in str(result) for x in ['安打', 'バ安', '越安', '線安']):
            if '二' in str(result) or '2' in str(result):
                return 'double'
            elif '三' in str(result) or '3' in str(result):
                return 'triple'
            else:
                return 'single'
        elif '二打' in str(result) or '越二' in str(result) or '線二' in str(result) or '中二' in str(result):
            return 'double'
        elif '三打' in str(result) or '越三' in str(result) or '線三' in str(result) or '中三' in str(result):
            return 'triple'
        elif '本打' in str(result) or '本' in str(result) and '打' in str(result):
            return 'homerun'
        else:
            return None
    except:
        return None

def convert_pitchtype(x):
    """球種を英語に変換"""
    pitch_map = {
        'ストレート': 'Fastball',
        'スライダー': 'Slider',
        'シンカー': 'Sinker',
        'フォーク': 'Fork',
        'カットボール': 'Cutter',
        'チェンジアップ': 'Changeup',
        'カーブ': 'Curveball',
        'シュート': 'Two-seam',
        '特殊球': 'Special',
        'ナックル': 'Knuckle',
        '-': 'Unknown'
    }
    return pitch_map.get(x, x)

def categorize_pitch(x):
    """球種を直球/変化球にカテゴリ分け"""
    fastballs = ['ストレート', 'シュート', 'カットボール']
    if x in fastballs:
        return '直球系'
    else:
        return '変化球'

def get_image_as_base64(image_path):
    """画像をBase64エンコード"""
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# =====================================================
# 可視化関数
# =====================================================

def create_spray_chart(df_player, player_name):
    """打球分布チャートを作成（Plotly版）"""
    
    # ヒット結果があるデータのみ抽出
    df_hits = df_player[df_player['HitResult'].notna()].copy()
    
    if len(df_hits) == 0:
        return None
    
    # 背景画像を読み込み
    img = Image.open('ballpark.png')
    img_width, img_height = img.size
    
    # 打球結果の色設定
    color_map = {
        'single': '#21CCEB',
        'double': '#52E672',
        'triple': '#D7EC15',
        'homerun': '#F10E0E'
    }
    
    hit_labels = {
        'single': '単打',
        'double': '二塁打',
        'triple': '三塁打',
        'homerun': '本塁打'
    }
    
    # シンボル設定
    symbol_map = {
        'single': 'circle',
        'double': 'diamond',
        'triple': 'triangle-up',
        'homerun': 'star'
    }
    
    fig = go.Figure()
    
    # 背景画像を追加
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=0.7,
            layer="below"
        )
    )
    
    # 各打球結果タイプごとにプロット
    for hit_type in ['single', 'double', 'triple', 'homerun']:
        df_type = df_hits[df_hits['HitResult'] == hit_type]
        
        if len(df_type) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=df_type['BallPositionX'],
            y=df_type['BallPositionY'],
            mode='markers',
            marker=dict(
                size=15,
                color=color_map[hit_type],
                symbol=symbol_map[hit_type],
                line=dict(width=1, color='white')
            ),
            name=hit_labels[hit_type],
            text=[f"gameID: {gid}<br>結果: {res}<br>投手: {pit}<br>球種: {ptype}" 
                  for gid, res, pit, ptype in zip(df_type['gameID'], df_type['AtBatResult'], 
                                                   df_type['PitcherName_x'], df_type['PitchType'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'🔵 {player_name} - 打球分布図 (2025)',
            font=dict(size=16)
        ),
        xaxis=dict(
            range=[0, img_width],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[img_height, 0],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_pitch_location_chart(df_player, player_name):
    """投球コース分布チャートを作成（Plotly版）- ヒットのみ、打席結果で色分け"""
    
    # 有効なLocationデータかつヒット結果があるデータのみ抽出
    df_valid = df_player[
        df_player['LocationX'].notna() & 
        df_player['LocationY'].notna() &
        df_player['HitResult'].notna()
    ].copy()
    
    if len(df_valid) == 0:
        return None
    
    # 打球結果の色設定（spray_chartと同じ）
    color_map = {
        'single': '#21CCEB',
        'double': '#52E672',
        'triple': '#D7EC15',
        'homerun': '#F10E0E'
    }
    
    hit_labels = {
        'single': '単打',
        'double': '二塁打',
        'triple': '三塁打',
        'homerun': '本塁打'
    }
    
    # シンボル設定（打席結果ごと）
    symbol_map = {
        'single': 'circle',
        'double': 'diamond',
        'triple': 'triangle-up',
        'homerun': 'star'
    }
    
    fig = go.Figure()
    
    # ストライクゾーンを描画
    # ボールゾーン（外側）
    fig.add_shape(type="rect",
        x0=0, y0=0, x1=135, y1=160,
        line=dict(color="gray", width=2),
        fillcolor="rgba(200,200,200,0.2)"
    )
    
    # ストライクゾーン（内側）
    fig.add_shape(type="rect",
        x0=27, y0=32, x1=108, y1=128,
        line=dict(color="red", width=3),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    # グリッド線
    for y in [0, 32, 64, 96, 128, 160]:
        fig.add_shape(type="line",
            x0=0, y0=y, x1=135, y1=y,
            line=dict(color="lightgray", width=1, dash="dot")
        )
    for x in [0, 27, 54, 81, 108, 135]:
        fig.add_shape(type="line",
            x0=x, y0=0, x1=x, y1=160,
            line=dict(color="lightgray", width=1, dash="dot")
        )
    
    # 打席結果ごとにプロット
    for hit_type in ['single', 'double', 'triple', 'homerun']:
        df_hit = df_valid[df_valid['HitResult'] == hit_type]
        
        if len(df_hit) == 0:
            continue
        
        hover_texts = []
        for _, row in df_hit.iterrows():
            hover_text = (f"gameID: {row['gameID']}<br>"
                         f"球種: {row['PitchType']}<br>"
                         f"結果: {row['AtBatResult']}<br>"
                         f"投手: {row['PitcherName_x']}<br>"
                         f"球速: {row['Velocity']}km/h")
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=df_hit['LocationX'],
            y=df_hit['LocationY'],
            mode='markers',
            marker=dict(
                size=14,
                color=color_map[hit_type],
                symbol=symbol_map[hit_type],
                line=dict(width=1, color='white'),
                opacity=0.85
            ),
            name=hit_labels[hit_type],
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'⚾ {player_name} - 打球コース分布 (2025)',
            font=dict(size=16)
        ),
        xaxis=dict(
            range=[-10, 145],
            showgrid=False,
            #title="左 ← 横方向 → 右",
            zeroline=False
        ),
        yaxis=dict(
            range=[170, -10],
            showgrid=False,
            #title="高め ← 縦方向 → 低め",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        margin=dict(l=50, r=20, t=80, b=50)
    )
    
    # 注釈を追加
    #fig.add_annotation(
    #    x=67.5, y=-5,
    #    text="🏠 ホームベース",
    #    showarrow=False,
    #    font=dict(size=10)
    #)
    
    return fig

# =====================================================
# メインアプリケーション
# =====================================================

def main():
    st.markdown('<p class="main-header">⚾ 野球打球・投球分析ダッシュボード</p>', unsafe_allow_html=True)
    
    # データ読み込み
    with st.spinner('データを読み込み中...'):
        df = load_data()
    
    # サイドバー
    st.sidebar.header("🔍 フィルター設定")
    
    # 打者選択
    batters = sorted(df['BatterName'].unique())
    selected_batter = st.sidebar.selectbox(
        "打者を選択",
        batters,
        index=batters.index('近本 光司') if '近本 光司' in batters else 0
    )
    
    # 選択した打者のデータを抽出
    df_player = df[df['BatterName'] == selected_batter].copy()
    
    # 統計情報
    st.sidebar.markdown("---")
    st.sidebar.header("📊 統計サマリー")
    
    total_abs = len(df_player)
    hits = df_player[df_player['HitResult'].notna()]
    singles = len(hits[hits['HitResult'] == 'single'])
    doubles = len(hits[hits['HitResult'] == 'double'])
    triples = len(hits[hits['HitResult'] == 'triple'])
    homeruns = len(hits[hits['HitResult'] == 'homerun'])
    
    st.sidebar.metric("打席数", total_abs)
    col1, col2 = st.sidebar.columns(2)
    col1.metric("単打", singles)
    col2.metric("二塁打", doubles)
    col1.metric("三塁打", triples)
    col2.metric("本塁打", homeruns)
    
    # フィルター追加
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ 凡例")
    st.sidebar.markdown("""
    **シンボル:**
    - ● 単打
    - ◆ 二塁打
    - ▲ 三塁打
    - ★ 本塁打
    """)
    
    # メインコンテンツ - 左右に並べて表示
    st.markdown(f'### 打者: {selected_batter}')
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        spray_chart = create_spray_chart(df_player, selected_batter)
        if spray_chart:
            st.plotly_chart(spray_chart, use_container_width=True)
        else:
            st.warning("この打者のヒットデータがありません。")
    
    with col_right:
        pitch_chart = create_pitch_location_chart(df_player, selected_batter)
        if pitch_chart:
            st.plotly_chart(pitch_chart, use_container_width=True)
        else:
            st.warning("投球データがありません。")
    
    # 統計データセクション
    st.markdown("---")
    
    # 統計計算用データの準備
    df_hits = df_player[df_player['HitResult'].notna()].copy()
    
    with st.expander("📊 詳細統計データ", expanded=False):
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown("#### 球種別打率")
            
            # 球種別打率の計算
            pitch_stats_list = []
            for pitch_type in df_player['PitchType'].dropna().unique():
                df_pitch = df_player[df_player['PitchType'] == pitch_type]
                total = len(df_pitch)
                hits = len(df_pitch[df_pitch['HitResult'].notna()])
                avg = hits / total if total > 0 else 0
                pitch_stats_list.append({
                    '球種': pitch_type,
                    '打席数': total,
                    'ヒット数': hits,
                    '打率': f'{avg:.3f}'
                })
            
            if pitch_stats_list:
                pitch_stats_df = pd.DataFrame(pitch_stats_list)
                pitch_stats_df = pitch_stats_df.sort_values('打席数', ascending=False)
                st.dataframe(pitch_stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("データがありません")
        
        with stat_col2:
            st.markdown("#### 安打種類別 - 直球系/変化球系 割合")
            
            # 直球系と変化球系の定義
            fastballs = ['ストレート', 'シュート', 'カットボール']
            
            # 安打種類別の球種カテゴリ割合
            hit_category_stats = []
            hit_types_order = ['single', 'double', 'triple', 'homerun']
            hit_labels_jp = {'single': '単打', 'double': '二塁打', 'triple': '三塁打', 'homerun': '本塁打'}
            
            for hit_type in hit_types_order:
                df_hit_type = df_hits[df_hits['HitResult'] == hit_type]
                total = len(df_hit_type)
                
                if total > 0:
                    fastball_count = len(df_hit_type[df_hit_type['PitchType'].isin(fastballs)])
                    breaking_count = total - fastball_count
                    fastball_pct = fastball_count / total * 100
                    breaking_pct = breaking_count / total * 100
                    
                    hit_category_stats.append({
                        '安打種類': hit_labels_jp[hit_type],
                        '本数': total,
                        '直球系': f'{fastball_count} ({fastball_pct:.1f}%)',
                        '変化球': f'{breaking_count} ({breaking_pct:.1f}%)'
                    })
            
            if hit_category_stats:
                hit_category_df = pd.DataFrame(hit_category_stats)
                st.dataframe(hit_category_df, use_container_width=True, hide_index=True)
            else:
                st.info("ヒットデータがありません")
        
        # 追加: ヒット詳細テーブル
        st.markdown("#### ヒット詳細データ")
        if len(df_hits) > 0:
            hits_data = df_hits[
                ['gameID', 'AtBatResult', 'HitResult', 'PitcherName_x', 'PitchType', 'Velocity', 'Inning']
            ].rename(columns={
                'gameID': '試合ID',
                'AtBatResult': '打席結果',
                'HitResult': '安打種類',
                'PitcherName_x': '投手',
                'PitchType': '球種',
                'Velocity': '球速',
                'Inning': 'イニング'
            })
            st.dataframe(hits_data, use_container_width=True, hide_index=True)
        else:
            st.info("ヒットデータがありません")
    
    # フッター
    st.markdown("---")
    st.markdown("📌 **使い方**: 左のサイドバーから打者を選択すると、その打者の打球分布と投球コースが表示されます。グラフ上でマウスオーバーすると詳細情報が表示されます。")

if __name__ == "__main__":
    main()
