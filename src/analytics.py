import math
import numpy as np
import pandas as pd

def calculate_shannon_polarization(stance_series):
    """
    Calculate normalized Shannon entropy across all detected camps.
    Neutral or spam comments are excluded from the camp distribution.
    """
    counts = stance_series.value_counts()
    camp_counts = counts[counts.index.to_series().str.match(r"^[A-E]$")]
    total_camp_comments = camp_counts.sum()

    if total_camp_comments == 0 or len(camp_counts) <= 1:
        return 0.0

    probabilities = camp_counts / total_camp_comments
    entropy = 0.0
    for probability in probabilities:
        entropy -= probability * math.log2(probability)
    normalized_entropy = entropy / math.log2(len(camp_counts))
    return round(max(0.0, min(1.0, normalized_entropy)), 4)

def calculate_weighted_arousal(df_comments):
    """
    Calculate weighted affective arousal:
    E_arousal = SUM(arousal_i * ln(likes_i + 2)) / SUM(ln(likes_i + 2))
    """
    if df_comments.empty:
        return 0.0
        
    safe_like_counts = np.maximum(
        df_comments["like_count"].fillna(0).astype(float),
        0
    )
    weights = np.log(safe_like_counts + 2)
    arousal_scores = df_comments["arousal_score"].fillna(0.0).astype(float)
    weighted_sum = (arousal_scores * weights).sum()
    total_weight = weights.sum()
    
    if total_weight == 0:
        return 0.0
        
    return round(float(weighted_sum / total_weight), 4)

def calculate_discussion_depth(df_comments):
    """Calculate discussion depth as replies divided by top-level threads."""
    replies = df_comments["is_reply"].sum()
    threads = len(df_comments) - replies
    if threads <= 0:
        return 0.0
    return round(float(replies / threads), 3)

def compute_toi_cluster_metrics(df_videos, df_comments):
    """
    Aggregate per-video metrics and calculate the Topic Opportunity Index (TOI).
    TOI = 0.35 * I_pol + 0.30 * E_arousal + 0.20 * OF + 0.15 * D_depth
    """
    clusters = []
    
    for _, vid in df_videos.iterrows():
        v_id = vid["video_id"]
        v_comms = df_comments[df_comments["video_id"] == v_id]
        
        if v_comms.empty:
            continue
            
        i_pol = calculate_shannon_polarization(v_comms["stance"])
        e_arousal = calculate_weighted_arousal(v_comms)
        d_depth = calculate_discussion_depth(v_comms)
        of = vid["outlier_factor"]
        if pd.isna(of):
            of = 0.0
        
                        # Extract the dominant thesis for each camp.
        camp_a_args = v_comms[v_comms["stance"] == "A"]["core_argument"].value_counts()
        camp_b_args = v_comms[v_comms["stance"] == "B"]["core_argument"].value_counts()
        
        thesis_a = camp_a_args.index[0] if not camp_a_args.empty else "Camp A position"
        thesis_b = camp_b_args.index[0] if not camp_b_args.empty else "Camp B position"
        
        clusters.append({
            "cluster_id": f"clust_{v_id}",
            "video_id": v_id,
            "topic_name": vid["title"],
            "camp_a_thesis": thesis_a,
            "camp_b_thesis": thesis_b,
            "shannon_polarization": i_pol,
            "weighted_arousal": e_arousal,
            "discussion_depth": d_depth,
            "outlier_factor": of
        })
        
    df_clust = pd.DataFrame(clusters)
    if df_clust.empty:
        return df_clust
        
    # Normalize only open-ended metrics. Polarization and arousal already use 0..1.
    def min_max(col):
        c_min, c_max = col.min(), col.max()
        return (col - c_min) / (c_max - c_min) if c_max > c_min else col * 0.0 + 1.0

    df_clust["norm_of"] = min_max(df_clust["outlier_factor"].fillna(0.0))
    df_clust["norm_depth"] = min_max(df_clust["discussion_depth"].fillna(0.0))
    
    # Calculate TOI.
    df_clust["toi_score"] = (
        0.35 * df_clust["shannon_polarization"] +
        0.30 * df_clust["weighted_arousal"] +
        0.20 * df_clust["norm_of"] +
        0.15 * df_clust["norm_depth"]
    ).round(4)
    
    # Opportunity matrix quadrants:
    # Q1: high polarization and high arousal -> viral arbitrage.
    # Q2: low polarization and high arousal -> community outcry or hype.
    # Q3: high polarization and low arousal -> niche debate.
    # Q4: low polarization and low arousal -> low opportunity.
    def assign_quadrant(row):
        high_pol = row["shannon_polarization"] >= 0.70
        high_arousal = row["weighted_arousal"] >= 0.50
        if high_pol and high_arousal:
            return "🔥 Viral Arbitrage (Polarized Conflict)"
        elif not high_pol and high_arousal:
            return "⚡ Community Outcry / Hype"
        elif high_pol and not high_arousal:
            return "⚖️ Niche Debate"
        else:
            return "💤 Low Opportunity"

    df_clust["opportunity_quadrant"] = df_clust.apply(assign_quadrant, axis=1)
    
    return df_clust.drop(columns=["norm_of", "norm_depth"])