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
    """Wczytuje CSV z faktami (komentarze) i łączy z wymiarem wideo oraz campów po camp_key."""
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
            'Camp Key': 'camp_key',
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
                'Total Likes': 'video_likes',
                'Views Per Hour (VPH)': 'views_per_hour',
                'Total Comments': 'total_comments',
                'Duration': 'duration',
                'Tags': 'tags',
                'Description': 'description'
            }
            df_vids = df_vids.rename(columns=vids_mapping)
            
            # Pobieramy również plik _channels.csv i łączymy po channel_key
            channels_csv = Path(str(file_path).replace("_comments", "_channels"))
            if channels_csv.exists():
                df_ch = pd.read_csv(channels_csv, sep=";", encoding="utf-8-sig")
                df_ch = df_ch.rename(columns={
                    'Channel Key': 'channel_key',
                    'Channel Title': 'channel_title'
                })
                if 'channel_key' in df_vids.columns and 'channel_key' in df_ch.columns:
                    df_vids = df_vids.merge(df_ch[['channel_key', 'channel_title']], on='channel_key', how='left')

            if 'video_key' in df_comms.columns and 'video_key' in df_vids.columns:
                df_comms = df_comms.merge(df_vids, on='video_key', how='left')

                # Szukamy powiązanego pliku _camps.csv i łączymy po camp_key
        camps_csv = Path(str(file_path).replace("_comments", "_camps"))
        if camps_csv.exists():
            df_camps = pd.read_csv(camps_csv, sep=";", encoding="utf-8-sig")
            camps_mapping = {
                'Camp Key': 'camp_key',
                'Video Key': 'video_key',
                'Camp Title': 'camp_title',
                'Perspective Group': 'perspective_group',
                'Camp Category': 'camp_category',
                'Camp Description': 'camp_description'
            }
            df_camps = df_camps.rename(columns=camps_mapping)

            if 'camp_key' in df_comms.columns and 'camp_key' in df_camps.columns:
                df_comms = df_comms.merge(df_comms, on=['camp_key', 'video_key'], how='left') if 'camp_key' in df_comms.columns else df_comms
                
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
    """Precyzyjnie wyciąga opis tylko dla danej litery campu, usuwając tagi."""
    if pd.isna(full_text) or full_text == "":
        return "No definition available."
    
    text = str(full_text)
    pattern = rf"{camp_id}:?\s*(.*?)(?=[A-E]|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        clean_text = match.group(1).strip()
        return clean_text.rstrip(';').strip()
    
    return text

def get_video_stats(df):
    """Grupuje komentarze po filmach i liczy statystyki analityczne przy użyciu campów."""
    video_stats = []
    for title, group in df.groupby('title'):
        stance_series = group['camp_key'] if 'camp_key' in group.columns else group.get('camp_title', pd.Series([1]*len(group)))
        i_pol = calculate_shannon_polarization(stance_series)
        e_arousal = calculate_weighted_arousal(group)
        d_depth = calculate_discussion_depth(group) if 'is_reply' in group.columns else 0.0
        
        total_comments = int(group['total_comments'].iloc[0]) if 'total_comments' in group.columns and pd.notna(group['total_comments'].iloc[0]) else len(group)
        analyzed_count = len(group)
        coverage_pct = (analyzed_count / total_comments * 100) if total_comments > 0 else 100.0
        
        toi_score = (0.4 * i_pol + 0.4 * e_arousal + 0.2 * min(d_depth/5, 1.0))
        
        if i_pol >= 0.7 and e_arousal >= 0.5: quadrant = "🔥 Viral Arbitrage"
        elif i_pol < 0.7 and e_arousal >= 0.5: quadrant = "⚡ Community Hype"
        elif i_pol >= 0.7 and e_arousal < 0.5: quadrant = "⚖️ Niche Debate"
        else: quadrant = "💤 Low Opportunity"

        video_stats.append({
            'topic_name': title,
            'video_key': group['video_key'].iloc[0] if 'video_key' in group.columns else 0,
            'channel_title': group['channel_title'].iloc[0] if 'channel_title' in group.columns else "Unknown",
            'shannon_polarization': i_pol,
            'weighted_arousal': e_arousal,
            'discussion_depth': d_depth,
            'toi_score': toi_score,
            'total_comments': total_comments,
            'opportunity_quadrant': quadrant,
            'coverage': f"{coverage_pct:.1f}% ({analyzed_count}/{total_comments})",
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
            'Total Likes': 'likes',
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
            df_videos[['title', 'views', 'views_per_hour']],
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

        # Wczytujemy również plik _camps.csv osobno do wyświetlania kart campów dla każdego filmu
    camps_csv_path = Path(str(selected_csv).replace("_comments", "_camps"))
    df_all_camps = pd.DataFrame()
    if camps_csv_path.exists():
        try:
            df_all_camps = pd.read_csv(camps_csv_path, sep=";", encoding="utf-8-sig")
            df_all_camps = df_all_camps.rename(columns={
                'Camp Key': 'camp_key',
                'Video Key': 'video_key',
                'Camp Title': 'camp_title',
                'Perspective Group': 'perspective_group',
                'Camp Category': 'camp_category',
                'Camp Description': 'camp_description'
            })
        except Exception:
            pass

    for idx, row in df_clusters.reset_index().iterrows():
        with st.expander(f"#{idx+1} | 💬 {row['total_comments']:,} comments | {row['topic_name']}"):
            st.caption(f"📊 {row['coverage']}")
            
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
                video_key = row.get('video_key', None)
                video_camps = df_all_camps[df_all_camps['video_key'] == video_key] if not df_all_camps.empty and video_key else pd.DataFrame()
                video_comments = df_comments[df_comments['title'] == row['topic_name']]

                if not video_camps.empty:
                    camp_cols = st.columns(len(video_camps))
                    for i, (_, camp_row) in enumerate(video_camps.iterrows()):
                        c_id = camp_row['camp_key']
                        c_title = camp_row.get('camp_title', f"Camp {i+1}")
                        c_cat = camp_row.get('camp_category', 'General')
                        c_desc = camp_row.get('camp_description', '')
                        
                        with camp_cols[i]:
                            st.markdown(f"""
                            <div class="camp-box">
                                <span style="background-color: #388bfd22; color: #58a6ff; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; border: 1px solid #388bfd55;">🏷️ {c_cat}</span><br>
                                <b style="color: #ffffff; font-size: 1.05rem; margin-top: 4px; display: inline-block;">{c_title}</b><br>
                                <p style="color: #8b949e; font-size: 0.85rem; margin-top: 6px;">{c_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Szukamy najczęstszego argumentu w komentarzach przypisanych do tego campu po camp_key
                            if not video_comments.empty and 'camp_key' in video_comments.columns:
                                camp_comms = video_comments[video_comments['camp_key'] == c_id]
                                if not camp_comms.empty and 'core_argument' in camp_comms.columns:
                                    top_arg = camp_comms['core_argument'].mode()
                                    if not top_arg.empty:
                                        st.markdown(f"""
                                        <div class="argument-box">
                                            📌 <i>{top_arg[0]}</i>
                                        </div>
                                        """, unsafe_allow_html=True)
                else:
                    st.info("No distinct camps identified for this video.")

    with st.expander("📂 View Raw Comment Data"):
        st.dataframe(df_comments, use_container_width=True)
else:
    st.error("The selected CSV file is empty or corrupted.")