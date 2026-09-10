import duckdb

DB_PATH = "social_radar.duckdb"

def get_connection():
    return duckdb.connect(DB_PATH)

def init_db():
    con = get_connection()
    
    # Drop existing tables to recreate with correct surrogate key schema if needed
    con.execute("DROP TABLE IF EXISTS Dim_Clusters;")
    con.execute("DROP TABLE IF EXISTS Fact_Comments;")
    con.execute("DROP TABLE IF EXISTS Dim_Videos;")
    con.execute("DROP TABLE IF EXISTS Dim_Channels;")

    # 1. Channels dimension.
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Channels (
        channel_key BIGINT PRIMARY KEY,
        channel_id VARCHAR UNIQUE,
        channel_title VARCHAR,
        custom_url VARCHAR,
        country VARCHAR,
        subscriber_count BIGINT,
        video_count BIGINT,
        total_view_count BIGINT
    );
    """)

    # 2. Videos dimension.
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Videos (
        video_key BIGINT PRIMARY KEY,
        channel_key BIGINT,
        video_id VARCHAR UNIQUE,
        title VARCHAR,
        channel_title VARCHAR,
        published_at TIMESTAMP,
        views BIGINT,
        likes BIGINT,
        comments_count BIGINT,
        views_per_hour DOUBLE,
        channel_subs BIGINT,
        duration VARCHAR,
        tags VARCHAR,
        description VARCHAR,
        FOREIGN KEY (channel_key) REFERENCES Dim_Channels(channel_key)
    );
    """)
    
        # 3. Comment facts and AI analysis.
    con.execute("""
    CREATE TABLE IF NOT EXISTS Fact_Comments (
        comment_key BIGINT,
        video_key BIGINT,
        comment_id VARCHAR UNIQUE,
        text VARCHAR,
        author VARCHAR,
        published_at TIMESTAMP,
        like_count BIGINT,
        reply_count BIGINT,
        is_reply BOOLEAN,
        engagement_score BIGINT,
        stance VARCHAR,              -- CAMP_A..E, NEUTRAL_SPAM
        arousal_score DOUBLE,        -- 0.0 to 1.0
        emotion_tag VARCHAR,
        core_argument VARCHAR,
        FOREIGN KEY (video_key) REFERENCES Dim_Videos(video_key)
    );
    """)

    # 4. Camps dimension (wide format matching _camps.csv).
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Camps (
        video_key BIGINT PRIMARY KEY,
        camp_a VARCHAR,
        percent_share_camp_a DOUBLE,
        camp_b VARCHAR,
        percent_share_camp_b DOUBLE,
        camp_c VARCHAR,
        percent_share_camp_c DOUBLE,
        camp_d VARCHAR,
        percent_share_camp_d DOUBLE,
        camp_e VARCHAR,
        percent_share_camp_e DOUBLE,
        FOREIGN KEY (video_key) REFERENCES Dim_Videos(video_key)
    );
    """)
    
    # 4. Aggregated topic clusters (TOI).
    con.execute("""
    CREATE TABLE IF NOT EXISTS Dim_Clusters (
        cluster_id VARCHAR PRIMARY KEY,
        video_key BIGINT,
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
        FOREIGN KEY (video_key) REFERENCES Dim_Videos(video_key)
    );
    """)
    con.close()
    print("✅ DuckDB star schema with surrogate keys initialized.")

if __name__ == "__main__":
    init_db()