import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
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
    """Wczytuje CSV i mapuje nazwy kolumn na standard używany w kodzie."""
    try:
        df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
        column_mapping = {
            'Video': 'title',
            'Channel': 'channel_title',
            'Published Date': 'published_at',
            'Comment': 'text',
            'Author': 'author',
            'Likes': 'like_count',
            'Reply Count': 'reply_count',
            'Stance': 'stance',
            'Camp Definition': 'camp_definition',
            'Arousal Score': 'arousal_score',
            'Emotion': 'emotion_tag',
            'Core Argument': 'core_argument',
            'Unresolved Question': 'unresolved_question',
            'Link': 'link'
        }
        df = df.rename(columns=column_mapping)
        # Podstawowa walidacja typów
        df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
        df['arousal_score'] = pd.to_numeric(df['arousal_score'], errors='coerce').fillna(0)
        df['is_reply'] = df['reply_count'] > 0
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def extract_camp_desc(full_text, camp_id):
    """Wyciąga opis konkretnego obozu ze zbiorczego ciągu definicji."""
    if pd.isna(full_text) or full_text == "":
        return "No definition available."
    try:
        parts = str(full_text).split(';')
        for p in parts:
            if f"{camp_id}:" in p:
                return p.split(f"{camp_id}:")[1].strip()
        return full_text # Jeśli nie znaleźliśmy separatora, zwracamy całość
    except:
        return full_text

def get_video_stats(df):
    """Grupuje komentarze po filmach i liczy statystyki analityczne."""
    video_stats = []
    for title, group in df.groupby('title'):
        i_pol = calculate_shannon_polarization(group['stance'])
        e_arousal = calculate_weighted_arousal(group)
        d_depth = calculate_discussion_depth(group)
        
        # Pobranie metadanych zapisanych w CSV
        coverage = group['Comment Analysis Coverage'].iloc[0] if 'Comment Analysis Coverage' in group.columns else "N/A"
        camp_shares = group['Camp Shares'].iloc[0] if 'Camp Shares' in group.columns else "N/A"
        
        # Obliczenie uproszczonego TOI (Topic Opportunity Index)
        toi_score = (0.4 * i_pol + 0.4 * e_arousal + 0.2 * min(d_depth/5, 1.0))
        
        # Klasyfikacja kwadrantu
        if i_pol >= 0.7 and e_arousal >= 0.5:
            quadrant = "🔥 Viral Arbitrage"
        elif i_pol < 0.7 and e_arousal >= 0.5:
            quadrant = "⚡ Community Hype"
        elif i_pol >= 0.7 and e_arousal < 0.5:
            quadrant = "⚖️ Niche Debate"
        else:
            quadrant = "💤 Low Opportunity"

        video_stats.append({
            'topic_name': title,
            'channel_title': group['channel_title'].iloc[0],
            'shannon_polarization': i_pol,
            'weighted_arousal': e_arousal,
            'discussion_depth': d_depth,
            'toi_score': toi_score,
            'opportunity_quadrant': quadrant,
            'coverage': coverage,
            'camp_shares': camp_shares,
            'link': group['link'].iloc[0] if 'link' in group.columns else "#"
        })
    return pd.DataFrame(video_stats).sort_values('toi_score', ascending=False)

# --- LOGIKA GŁÓWNA ---

# Pasek boczny - wybór raportu
csv_files = sorted(Path(".").glob("discourse_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
if not csv_files:
    st.warning("⚠️ No CSV reports found. Run `run_pipeline.py` first.")
    st.stop()

selected_csv = st.sidebar.selectbox(
    "Select Report",
    csv_files,
    format_func=lambda path: path.stem.replace("discourse_", "").replace("_", " ").title()
)

df_comments = load_csv_data(selected_csv)

if not df_comments.empty:
    df_clusters = get_video_stats(df_comments)

    # 1. KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top TOI Score", f"{df_clusters['toi_score'].max():.3f}")
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
        hover_data=["channel_title", "toi_score"],
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

    # 3. TOI Ranking
    st.subheader("🏆 Highest-Opportunity Topics (TOI)")

    for idx, row in df_clusters.iterrows():
        with st.expander(f"#{idx+1} [TOI: {row['toi_score']:.3f}] - {row['topic_name']}"):
            st.caption(f"📊 {row['coverage']} | 📢 Shares: {row['camp_shares']}")
            
            c_info, c_camps = st.columns([1, 3])
            
            with c_info:
                st.markdown(f"**Channel:**\n{row['channel_title']}")
                st.markdown(f"**Polarization:** `{row['shannon_polarization']:.2f}`")
                st.markdown(f"**Arousal:** `{row['weighted_arousal']:.2f}`")
                st.markdown(f"**Quadrant:**\n{row['opportunity_quadrant']}")
                st.link_button("Watch Video", row['link'])
            
            with c_camps:
                video_comments = df_comments[df_comments['title'] == row['topic_name']]
                # Filtrujemy tylko wypowiedzi sklasyfikowane jako Campy
                camp_data = video_comments[video_comments['stance'].str.match(r'^CAMP_[A-E]$', na=False)]
                
                if not camp_data.empty:
                    # Pobieramy zbiorczą definicję dla tego filmu
                    raw_definitions = str(camp_data['camp_definition'].iloc[0])
                    
                    # Liczymy udziały procentowe
                    counts = camp_data['stance'].value_counts(normalize=True) * 100
                    counts = counts.sort_index() # Sortowanie A, B, C...
                    
                    camp_cols = st.columns(len(counts))
                    for i, (camp_name, share) in enumerate(counts.items()):
                        with camp_cols[i]:
                            # Wyodrębniamy tylko opis dla TEGO obozu
                            clean_desc = extract_camp_desc(raw_definitions, camp_name)
                            
                            st.markdown(f"""
                            <div class="camp-box">
                                <b style="color: #58a6ff; font-size: 1.1rem;">{camp_name} ({share:.1f}%)</b><br>
                                <p style="color: #8b949e; font-size: 0.85rem; margin-top: 8px;">{clean_desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Pobieramy najczęstszy argument dla tego obozu
                            top_arg = video_comments[video_comments['stance'] == camp_name]['core_argument'].mode()
                            if not top_arg.empty:
                                st.markdown(f"""
                                <div class="argument-box">
                                    📌 <i>{top_arg[0]}</i>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("No distinct camps identified for this video.")

            st.success(f"**💡 Suggested Angle:** Why everyone is wrong about {row['topic_name'][:50]}... [The truth behind the conflict]")

    # Opcjonalnie: Tabela na samym dole
    with st.expander("📂 View Raw Comment Data"):
        st.dataframe(df_comments, use_container_width=True)
else:
    st.error("The selected CSV file is empty or corrupted.")