import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import re
# Importujemy funkcje analityczne do przeliczania danych z CSV
from src.analytics import (
    calculate_shannon_polarization, 
    calculate_weighted_arousal, 
    calculate_discussion_depth
)

st.set_page_config(
    page_title="Social Radar: Content Arbitrage Engine", 
    layout="wide", 
    page_icon="🎯"
)

# --- STYLIZACJA CSS ---
st.markdown("""
    <style>
    .stMetric { 
        background-color: #1e2130; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #30363d; 
    }
    .camp-box {
        background: #161b22; 
        padding: 12px; 
        border-radius: 8px; 
        border-top: 3px solid #58a6ff; 
        min-height: 140px;
        margin-bottom: 10px;
    }
    .argument-box {
        padding-top: 8px; 
        border-top: 1px solid #30363d; 
        margin-top: 8px;
        font-size: 0.8rem;
        color: #d1d5db;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Social Radar: Viral Topic Detection Engine")
st.markdown("*Content Arbitrage Engine powered by AI discourse analysis*")

# --- FUNKCJE POMOCNICZE ---

@st.cache_data
def load_csv_data(file_path):
    """Wczytuje CSV z faktami (komentarze) i opcjonalnie łączy z wymiarem wideo oraz campów."""
    try:
        df_comms = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
        comms_mapping = {
            'Comment Key': 'comment_key',
            'Comment ID': 'comment_id',
            'Video Key': 'video_key',
            'Comment': 'text',
            'Author': 'author',
            'Comment Date': 'comment_published_at',
            'Likes': 'like_count',
            'Reply Count': 'reply_count',
            'Is Reply': 'is_reply',
            'Engagement Score': 'engagement_score',
            'Stance': 'stance',
            'Arousal Score': 'arousal_score',
            'Emotion': 'emotion_tag',
            'Core Argument': 'core_argument'
        }
        df_comms = df_comms.rename(columns=comms_mapping)
        
        # Szukamy powiązanego pliku _videos.csv
        videos_csv = Path(str(file_path).replace("_comments", "_videos"))
        if videos_csv.exists():
            df_vids = pd.read_csv(videos_csv, sep=";", encoding="utf-8-sig")
            vids_mapping = {
                'Video Key': 'video_key',
                'Video ID': 'video_id',
                'Title': 'title',
                'Channel Key': 'channel_key',
                'Published Date': 'published_at',
                'Views': 'views',
                'Likes': 'video_likes',
                'Subscribers': 'subscribers',
                'Views Per Hour (VPH)': 'views_per_hour',
                'Total Comments': 'total_comments',
                'Duration': 'duration',
                'Tags': 'tags',
                'Description': 'description'
            }
            df_vids = df_vids.rename(columns=vids_mapping)
            
            # Pobieramy również plik _channels.csv i łączymy po channel_key, aby odzyskać channel_title
            channels_csv = Path(str(file_path).replace("_comments", "_channels"))
            if channels_csv.exists():
                df_ch = pd.read_csv(channels_csv, sep=";", encoding="utf-8-sig")
                df_ch = df_ch.rename(columns={
                    'Channel Key': 'channel_key',
                    'Channel Title': 'channel_title'
                })
                if 'channel_key' in df_vids.columns and 'channel_key' in df_ch.columns:
                    df_vids = df_vids.merge(df_ch[['channel_key', 'channel_title']], on='channel_key', how='left')

            # Łączymy tabelę faktów (komentarze) z wymiarem wideo po video_key
            if 'video_key' in df_comms.columns and 'video_key' in df_vids.columns:
                df_comms = df_comms.merge(df_vids, on='video_key', how='left')

        # Szukamy powiązanego pliku _camps.csv i dołączamy go, aby móc korzystać z gotowych udziałów i definicji
        camps_csv = Path(str(file_path).replace("_comments", "_camps"))
        if camps_csv.exists():
            df_camps = pd.read_csv(camps_csv, sep=";", encoding="utf-8-sig")
            if 'video_key' in df_comms.columns and 'Video Key' in df_camps.columns:
                df_comms = df_comms.merge(df_camps, left_on='video_key', right_on='Video Key', how='left')
                if 'Video Key' in df_comms.columns:
                    df_comms = df_comms.drop(columns=['Video Key'])
                
        # Konwersja typów numerycznych
        df_comms['like_count'] = pd.to_numeric(df_comms['like_count'], errors='coerce').fillna(0)
        df_comms['reply_count'] = pd.to_numeric(df_comms.get('reply_count', 0), errors='coerce').fillna(0)
        df_comms['arousal_score'] = pd.to_numeric(df_comms['arousal_score'], errors='coerce').fillna(0)
        df_comms['engagement_score'] = pd.to_numeric(df_comms.get('engagement_score', 0), errors='coerce').fillna(0)
        df_comms['is_reply'] = df_comms['is_reply'].astype(bool) if 'is_reply' in df_comms.columns else (df_comms['reply_count'] > 0)
        return df_comms
    except Exception as e:
        st.error(f"Error loading and joining CSV data: {e}")
        return pd.DataFrame()

def extract_total_comments(coverage_text):
    """Wyciąga całkowitą liczbę komentarzy z tekstu np. '6.6% (100/1512)'"""
    if pd.isna(coverage_text):
        return 0
    match = re.search(r'/(\d+)\)', str(coverage_text))
    return int(match.group(1)) if match else 0

def extract_camp_desc(full_text, camp_id):
    """Precyzyjnie wyciąga opis tylko dla danego CAMP_X, usuwając tagi."""
    if pd.isna(full_text) or full_text == "":
        return "No definition available."
    
    text = str(full_text)
    pattern = rf"{camp_id}:?\s*(.*?)(?=CAMP_[A-E]|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        clean_text = match.group(1).strip()
        return clean_text.rstrip(';').strip()
    
    return text

def get_video_stats(df):
    """Grupuje komentarze po filmach i liczy statystyki analityczne przy użyciu danych z _camps.csv."""
    video_stats = []
    for title, group in df.groupby('title'):
        i_pol = calculate_shannon_polarization(group['stance'])
        e_arousal = calculate_weighted_arousal(group)
        d_depth = calculate_discussion_depth(group) if 'is_reply' in group.columns else 0.0
        
        # Pobieramy całkowitą liczbę komentarzy z _videos.csv jeśli jest dostępna, w przeciwieństwie len(group)
        total_comments = int(group['total_comments'].iloc[0]) if 'total_comments' in group.columns and pd.notna(group['total_comments'].iloc[0]) else len(group)
        
        # Zbuduj podsumowanie camp_shares bezpośrednio z kolumn _camps.csv
        shares_parts = []
        for camp_letter, camp_name in [('A', 'Camp A'), ('B', 'Camp B'), ('C', 'Camp C'), ('D', 'Camp D'), ('E', 'Camp E')]:
            share_col = f"Percent Share of Camp {camp_letter}"
            if share_col in group.columns:
                val = group[share_col].iloc[0]
                if pd.notna(val) and val > 0:
                    shares_parts.append(f"CAMP_{camp_letter}: {val:.1f}%")
        camp_shares = "; ".join(shares_parts) if shares_parts else "No identified camps"
        
        toi_score = (0.4 * i_pol + 0.4 * e_arousal + 0.2 * min(d_depth/5, 1.0))
        
        if i_pol >= 0.7 and e_arousal >= 0.5: quadrant = "🔥 Viral Arbitrage"
        elif i_pol < 0.7 and e_arousal >= 0.5: quadrant = "⚡ Community Hype"
        elif i_pol >= 0.7 and e_arousal < 0.5: quadrant = "⚖️ Niche Debate"
        else: quadrant = "💤 Low Opportunity"

        video_stats.append({
            'topic_name': title,
            'channel_title': group['channel_title'].iloc[0],
            'shannon_polarization': i_pol,
            'weighted_arousal': e_arousal,
            'discussion_depth': d_depth,
            'toi_score': toi_score,
            'total_comments': total_comments,
            'opportunity_quadrant': quadrant,
            'coverage': f"100% ({len(group)}/{total_comments})",
            'camp_shares': camp_shares,
            'link': group['link'].iloc[0] if 'link' in group.columns else "#"
        })
    return pd.DataFrame(video_stats).sort_values('total_comments', ascending=False)

# --- LOGIKA GŁÓWNA ---

# Szukamy raportów w podkatalogu 'exports/'
csv_files = sorted(
    list(Path("exports").glob("discourse_*comments*.csv")),
    key=lambda x: x.stat().st_mtime,
    reverse=True
)

if not csv_files:
    st.warning("⚠️ No CSV reports found. Run `run_pipeline.py` first.")
    st.stop()

selected_csv = st.sidebar.selectbox(
    "Select Report",
    csv_files,
    format_func=lambda path: path.stem.replace("discourse_", "").replace("_comments", "").replace("_", " ").title()
)

# Szukamy odpowiadającego pliku wideo (_videos.csv), jeśli istnieje
videos_csv = Path(str(selected_csv).replace("_comments", "_videos"))
df_videos = pd.DataFrame()
if videos_csv.exists():
    try:
        df_videos = pd.read_csv(videos_csv, sep=";", encoding="utf-8-sig")
        df_videos = df_videos.rename(columns={
            'Video Key': 'video_key',
            'Video ID': 'video_id',
            'Title': 'title',
            'Channel Key': 'channel_key',
            'Published Date': 'published_at',
            'Views': 'views',
            'Likes': 'likes',
            'Subscribers': 'subscribers',
            'Views Per Hour (VPH)': 'views_per_hour',
            'Total Comments': 'total_comments'
        })
        # Dołączamy channel_title z _channels.csv, jeśli istnieje
        channels_csv = Path(str(selected_csv).replace("_comments", "_channels"))
        if channels_csv.exists():
            df_ch = pd.read_csv(channels_csv, sep=";", encoding="utf-8-sig")
            df_ch = df_ch.rename(columns={
                'Channel Key': 'channel_key',
                'Channel Title': 'channel_title'
            })
            if 'channel_key' in df_videos.columns and 'channel_key' in df_ch.columns:
                df_videos = df_videos.merge(df_ch[['channel_key', 'channel_title']], on='channel_key', how='left')
    except Exception:
        pass

df_comments = load_csv_data(selected_csv)

if not df_comments.empty:
    df_clusters = get_video_stats(df_comments)

    # Dołączamy dane z wymiaru wideo (np. Views, VPH), jeśli są dostępne
    if not df_videos.empty and 'title' in df_videos.columns:
        df_clusters = df_clusters.merge(
            df_videos[['title', 'views', 'views_per_hour', 'subscribers']],
            left_on='topic_name',
            right_on='title',
            how='left',
            suffixes=('', '_vid')
        )
        if 'title_vid' in df_clusters.columns:
            df_clusters = df_clusters.drop(columns=['title_vid'])

    # 1. KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Most Comments", f"{df_clusters['total_comments'].max():,}")
    m2.metric("Mean Polarization", f"{df_clusters['shannon_polarization'].mean():.2f}")
    m3.metric("Mean Arousal", f"{df_clusters['weighted_arousal'].mean():.2f}")
    m4.metric("Analyzed Comments", f"{len(df_comments):,}")

    st.markdown("---")

    # 2. Quadrant Chart
    st.subheader("📍 Topic Opportunity Quadrant")
    fig = px.scatter(
        df_clusters,
        x="shannon_polarization", 
        y="weighted_arousal",
        color="opportunity_quadrant",
        size=[15]*len(df_clusters),
        hover_name="topic_name",
        hover_data=["channel_title", "toi_score", "total_comments"],
        labels={"shannon_polarization": "Polarization (I_pol)", "weighted_arousal": "Arousal (E_arousal)"},
        template="plotly_dark",
        color_discrete_map={
            "🔥 Viral Arbitrage": "#ff4b4b",
            "⚡ Community Hype": "#ffaa00",
            "⚖️ Niche Debate": "#00aaff",
            "💤 Low Opportunity": "#6d6d6d"
        }
    )
    fig.add_hline(y=0.50, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_vline(x=0.70, line_dash="dash", line_color="white", opacity=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Video Ranking
    st.subheader("🏆 Popular Topics (Sorted by Comment Count)")

    for idx, row in df_clusters.reset_index().iterrows():
        with st.expander(f"#{idx+1} | 💬 {row['total_comments']:,} comments | {row['topic_name']}"):
            st.caption(f"📊 {row['coverage']} | 📢 Shares: {row['camp_shares']}")
            
            c_info, c_camps = st.columns([1, 4])  # Więcej miejsca na campy
            
            with c_info:
                st.markdown(f"**Channel:**\n{row['channel_title']}")
                if 'views' in row and pd.notna(row['views']):
                    st.markdown(f"**Views:** `{int(row['views']):,}`")
                if 'views_per_hour' in row and pd.notna(row['views_per_hour']):
                    st.markdown(f"**VPH (Velocity):** `{row['views_per_hour']:.1f}/h`")
                st.markdown(f"**Polarization:** `{row['shannon_polarization']:.2f}`")
                st.markdown(f"**Arousal:** `{row['weighted_arousal']:.2f}`")
                st.link_button("Watch Video", row['link'])
            
            with c_camps:
                video_comments = df_comments[df_comments['title'] == row['topic_name']]
                if not video_comments.empty:
                    row_data = video_comments.iloc[0]
                    
                    active_camps = []
                    for camp_letter in ['A', 'B', 'C', 'D', 'E']:
                        def_col = f"Camp {camp_letter}"
                        share_col = f"Percent Share of Camp {camp_letter}"
                        if def_col in row_data and share_col in row_data:
                            c_def = row_data[def_col]
                            c_share = row_data[share_col]
                            if pd.notna(c_def) and str(c_def).strip() != "" and pd.notna(c_share) and c_share > 0:
                                active_camps.append((f"CAMP_{camp_letter}", c_def, c_share))
                    
                    if active_camps:
                        camp_cols = st.columns(len(active_camps))
                        for i, (camp_id, desc, share) in enumerate(active_camps):
                            with camp_cols[i]:
                                st.markdown(f"""
                                <div class="camp-box">
                                    <b style="color: #58a6ff; font-size: 1.1rem;">{camp_id} ({share:.1f}%)</b><br>
                                    <p style="color: #8b949e; font-size: 0.85rem; margin-top: 8px;">{desc}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                camp_name_mapped = camp_id
                                top_arg = video_comments[video_comments['stance'] == camp_name_mapped]['core_argument'].mode()
                                if not top_arg.empty:
                                    st.markdown(f"""
                                    <div class="argument-box">
                                        📌 <i>{top_arg[0]}</i>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No distinct camps identified for this video.")
                else:
                    st.info("No distinct camps identified for this video.")

    with st.expander("📂 View Raw Comment Data"):
        st.dataframe(df_comments, use_container_width=True)
else:
    st.error("The selected CSV file is empty or corrupted.")