import duckdb

DB_PATH = "social_radar.duckdb"

def get_connection():
    return duckdb.connect(DB_PATH)

def init_db():
    con = get_connection()
    
    # 1. Video dimension.
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Videos (
        video_id VARCHAR PRIMARY KEY,
        title VARCHAR,
        channel_title VARCHAR,
        channel_id VARCHAR,
        published_at TIMESTAMP,
        views BIGINT,
        likes BIGINT,
        comments_count BIGINT,
        channel_median_views DOUBLE,
        outlier_factor DOUBLE,
        channel_subs BIGINT
    );
    """)
    
    # 2. Comment facts and AI analysis.
    con.execute("""
    CREATE TABLE IF NOT EXISTS Fact_Comments (
        comment_id VARCHAR PRIMARY KEY,
        video_id VARCHAR,
        text VARCHAR,
        author VARCHAR,
        published_at TIMESTAMP,
        like_count BIGINT,
        is_reply BOOLEAN,
        stance VARCHAR,              -- CAMP_A, CAMP_B, NEUTRAL_SPAM
        arousal_score DOUBLE,        -- 0.0 to 1.0 (affective arousal)
        emotion_tag VARCHAR,         -- np. Anger, Frustration, Excitement
        core_argument VARCHAR,       -- Syntetyczny argument
        unresolved_question VARCHAR, -- Unresolved player question
        FOREIGN KEY (video_id) REFERENCES Dim_Videos(video_id)
    );
    """)
    
    # 3. Aggregated topic clusters (TOI).
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Clusters (
        cluster_id VARCHAR PRIMARY KEY,
        video_id VARCHAR,
        topic_name VARCHAR,
        camp_a_thesis VARCHAR,
        camp_b_thesis VARCHAR,
        shannon_polarization DOUBLE,
        weighted_arousal DOUBLE,
        discussion_depth DOUBLE,
        outlier_factor DOUBLE,
        toi_score DOUBLE,
        opportunity_quadrant VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES Dim_Videos(video_id)
    );
    """)
    con.close()
    print("✅ DuckDB star schema initialized.")

if __name__ == "__main__":
    init_db()