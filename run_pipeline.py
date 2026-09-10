import os
import csv
import re
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from src.enrichment import analyze_discourse_with_ai
from src.analytics import compute_toi_cluster_metrics
from src.db import get_connection, init_db

# 1. Load API keys from .env.
load_dotenv()
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env before running the pipeline.")

# 2. Initialize the YouTube client if key is available.
youtube = None
if youtube_api_key:
    youtube = build("youtube", "v3", developerKey=youtube_api_key)


def execute_with_retry(request_factory, operation_name, max_retries=3):
    """Retry temporary API rate limits with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return request_factory().execute()
        except Exception as error:
            error_text = str(error).lower()
            status_code = getattr(getattr(error, "resp", None), "status", None)
            is_temporary_limit = status_code == 429 or any(
                phrase in error_text
                for phrase in ("rate limit", "too many requests", "ratelimitexceeded")
            )
            is_daily_quota_error = "quotaexceeded" in error_text or "dailylimitexceeded" in error_text

            if not is_temporary_limit or is_daily_quota_error or attempt >= max_retries:
                raise

            delay = 5 * (2 ** attempt)
            print(f"   ⏳ {operation_name} rate limit reached. Retrying in {delay}s...")
            time.sleep(delay)

# === INTERACTIVE INPUT ===
print("=" * 60)
print("🎯 SOCIAL RADAR: DISCOURSE ANALYSIS (YOUTUBE)")
print("=" * 60)

def read_positive_integer(prompt, default, maximum=None):
    """Read a positive integer with a default and optional upper bound."""
    while True:
        value_text = input(f"{prompt} [{default}]: ").strip()
        if not value_text:
            return default
        if value_text.isdigit() and int(value_text) > 0:
            value = int(value_text)
            if maximum is None or value <= maximum:
                return value
        limit_message = f" and no more than {maximum}" if maximum else ""
        print(f"Please enter a positive whole number{limit_message}.")


# Collect all run settings before making API requests.
if not youtube:
    raise RuntimeError("YOUTUBE_API_KEY is required to fetch YouTube data.")

print("\n🔎 Select video source:")
print("   [1] Search by niche (default)")
print("   [2] Search within one creator")
print("   [3] Analyze specific YouTube links")
source_mode = input("👉 Choose 1-3 [1]: ").strip() or "1"
if source_mode not in {"1", "2", "3"}:
    source_mode = "1"

creator_query = ""
provided_links = []

if source_mode == "2":
    creator_query = input("👉 Creator name, handle, or channel URL: ").strip()
    if not creator_query:
        raise RuntimeError("A creator name, handle, or channel URL is required.")
elif source_mode == "3":
    links_text = input("👉 YouTube links (comma-separated): ").strip()
    provided_links = re.findall(
        r"(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})",
        links_text
    )
    provided_links = list(dict.fromkeys(provided_links))
    if not provided_links:
        raise RuntimeError("No valid YouTube video links were provided.")

niche = input("👉 Niche or topic [Albion Online]: ").strip() or "Albion Online"
        
print("\n📊 Select video ranking:")
print("   [1] Most comments (default)")
print("   [2] Most views")
print("   [3] Highest Views Per Hour (VPH)")
print("   [4] Mixed selection: VPH + views + comments")
ranking_mode = input("👉 Choose 1-4 [1]: ").strip() or "1"
if ranking_mode not in {"1", "2", "3", "4"}:
    ranking_mode = "1"

max_videos = read_positive_integer("👉 Maximum videos to analyze", 1, 50)
max_comments_per_video = read_positive_integer("👉 Maximum comments per video", 100)
max_video_age_days = read_positive_integer("👉 Maximum video age in days", 30)
max_channel_subscribers_text = input(
    "👉 Maximum channel subscribers [No limit]: "
).strip()
max_channel_subscribers = (
    int(max_channel_subscribers_text)
    if max_channel_subscribers_text.isdigit() and int(max_channel_subscribers_text) > 0
    else None
)

published_after = (
    datetime.now(timezone.utc) - timedelta(days=max_video_age_days)
).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"\n🔎 Preparing videos for '{niche}' from the last {max_video_age_days} days...")


# Fetch top-level comments and included replies.
def fetch_comments(video_id, max_comments=50):
    """Fetch a larger candidate pool, then keep the most engaging comments."""
    comments = []
    page_token = None
    candidate_limit = max(max_comments * 3, max_comments)
    try:
        while len(comments) < candidate_limit:
            res = execute_with_retry(
                lambda: youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, candidate_limit - len(comments)),
                    textFormat="plainText",
                    pageToken=page_token
                ),
                "YouTube comments"
            )
            
            for item in res.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]
                top_snippet = top_comment["snippet"]
                reply_count = int(item["snippet"].get("totalReplyCount", 0))
                comments.append({
                    "comment_id": top_comment["id"],
                    "video_id": video_id,
                    "text": top_snippet["textDisplay"],
                    "author": top_snippet.get("authorDisplayName", "Anonymous"),
                    "published_at": top_snippet.get("publishedAt"),
                    "like_count": int(top_snippet.get("likeCount", 0)),
                    "reply_count": reply_count,
                    "is_reply": False
                })
            
                if "replies" in item:
                    for reply in item["replies"].get("comments", []):
                        reply_snippet = reply["snippet"]
                        if len(comments) >= candidate_limit:
                            break
                        comments.append({
                            "comment_id": reply["id"],
                            "video_id": video_id,
                            "text": reply_snippet["textDisplay"],
                            "author": reply_snippet.get("authorDisplayName", "Anonymous"),
                            "published_at": reply_snippet.get("publishedAt"),
                            "like_count": int(reply_snippet.get("likeCount", 0)),
                            "reply_count": 0,
                            "is_reply": True
                        })

            page_token = res.get("nextPageToken")
            if not page_token or len(comments) >= candidate_limit:
                break
    except Exception as e:
        print(f"   ⚠️ Comment retrieval warning: {e}")

    comments.sort(
        key=lambda comment: (
            comment["like_count"] + comment["reply_count"],
            comment["like_count"],
            comment["reply_count"]
        ),
        reverse=True
    )
    return comments[:max_comments]


csv_rows = []
analyzed_comments = []
selected_videos = []

# === 3. YOUTUBE FETCHING ===
if source_mode == "3":
    video_ids = provided_links[:max_videos]
else:
    channel_id = None
    if source_mode == "2":
        creator_search = execute_with_retry(
            lambda: youtube.search().list(
                q=creator_query,
                part="id,snippet",
                type="channel",
                maxResults=1
            ),
            "YouTube creator search"
        )
        creator_items = creator_search.get("items", [])
        if not creator_items:
            raise RuntimeError("No YouTube creator matched the provided input.")
        channel_id = creator_items[0]["id"].get("channelId")

    search_result_limit = 50
    search_res = execute_with_retry(
        lambda: youtube.search().list(
            q=niche if source_mode == "1" else None,
            channelId=channel_id,
            part="id",
            type="video",
            publishedAfter=published_after,
            maxResults=search_result_limit,
            order="viewCount"
        ),
        "YouTube search"
    )

    video_ids = [
        item["id"]["videoId"]
        for item in search_res.get("items", [])
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", item.get("id", {}).get("videoId", ""))
    ]

if not video_ids:
    raise RuntimeError("No videos matched the selected search criteria.")

if video_ids:
    # 4. Fetch video statistics.
    videos = []
    for start in range(0, len(video_ids), 50):
        video_batch = video_ids[start:start + 50]
        videos_res = execute_with_retry(
            lambda ids=video_batch: youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(ids)
            ),
            "YouTube video details"
        )

        for item in videos_res.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})

            description = snippet.get("description", "").replace("\n", " ")
            tags = ", ".join(snippet.get("tags", []))
            duration = content.get("duration", "")
            like_count = int(stats.get("likeCount", 0))

            videos.append({
                "id": item["id"],
                "title": snippet.get("title"),
                "description": description,
                "tags": tags,
                "duration": duration,
                "like_count": like_count,
                "channel": snippet.get("channelTitle"),
                "channel_id": snippet.get("channelId"),
                "views": int(stats.get("viewCount", 0)),
                "comments_count": int(stats.get("commentCount", 0)),
                "published_at": snippet.get("publishedAt", "")
            })

    if videos:
        channel_ids = list({video["channel_id"] for video in videos if video["channel_id"]})
        channel_response = execute_with_retry(
            lambda: youtube.channels().list(
                part="snippet,statistics",
                id=",".join(channel_ids)
            ),
            "YouTube channel details"
        )
        channel_info_map = {}
        for item in channel_response.get("items", []):
            ch_id = item["id"]
            ch_stats = item.get("statistics", {})
            ch_snippet = item.get("snippet", {})
            channel_info_map[ch_id] = {
                "subscriber_count": int(ch_stats.get("subscriberCount", 0)),
                "video_count": int(ch_stats.get("videoCount", 0)),
                "view_count": int(ch_stats.get("viewCount", 0)),
                "custom_url": ch_snippet.get("customUrl", ""),
                "country": ch_snippet.get("country", "")
            }

        subscriber_counts = {ch_id: info["subscriber_count"] for ch_id, info in channel_info_map.items()}

        if max_channel_subscribers is not None:
            videos = [
                video for video in videos
                if subscriber_counts.get(video["channel_id"], 0) <= max_channel_subscribers
            ]
            
        now_dt = datetime.now(timezone.utc)
        for video in videos:
            ch_id = video["channel_id"]
            ch_data = channel_info_map.get(ch_id, {})
            subscribers = ch_data.get("subscriber_count", 0)
            video["subscribers"] = subscribers
            video["channel_video_count"] = ch_data.get("video_count", 0)
            video["channel_view_count"] = ch_data.get("view_count", 0)
            video["channel_custom_url"] = ch_data.get("custom_url", "")
            video["channel_country"] = ch_data.get("country", "")
                
            pub_raw = video.get("published_at", "")
            try:
                published_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except Exception:
                published_dt = now_dt - timedelta(days=1)
                
            age_hours = max((now_dt - published_dt).total_seconds() / 3600.0, 1.0)
            age_days = max(age_hours / 24.0, 0.1)
                
            video["age_in_hours"] = round(age_hours, 1)
            video["age_in_days"] = round(age_days, 1)
            video["views_per_hour"] = round(video["views"] / age_hours, 2)
            video["views_per_subscriber"] = (
                video["views"] / subscribers if subscribers > 0 else 0.0
            )

    videos = [video for video in videos if video["comments_count"] > 0]
    ranking_keys = {
        "1": lambda video: video["comments_count"],
        "2": lambda video: video["views"],
        "3": lambda video: video["views_per_hour"],
    }
    if ranking_mode != "4":
        videos.sort(key=ranking_keys[ranking_mode], reverse=True)

    if ranking_mode == "4":
        by_vph = sorted(videos, key=lambda video: video["views_per_hour"], reverse=True)
        by_views = sorted(videos, key=lambda video: video["views"], reverse=True)
        by_comments = sorted(videos, key=lambda video: video["comments_count"], reverse=True)

        vph_slots = max_videos // 3
        view_slots = max_videos // 3
        comment_slots = max_videos - vph_slots - view_slots

        for ranked_videos, slot_count in (
            (by_vph, vph_slots),
            (by_views, view_slots),
            (by_comments, comment_slots),
        ):
            if slot_count == 0:
                continue
            selected_for_criterion = 0
            for video in ranked_videos:
                if video["id"] not in {item["id"] for item in selected_videos}:
                    selected_videos.append(video)
                    selected_for_criterion += 1
                    if selected_for_criterion >= slot_count:
                        break
                if len(selected_videos) >= max_videos:
                    break

        if len(selected_videos) < max_videos:
            for video in videos:
                if video["id"] not in {item["id"] for item in selected_videos}:
                    selected_videos.append(video)
                if len(selected_videos) >= max_videos:
                    break
    else:
        selected_videos = videos[:max_videos]

    ranking_labels = {
        "1": "the most comments",
        "2": "the most views",
        "3": "the highest Views Per Hour (VPH)",
        "4": "the mixed selection",
    }
    print(f"\n🎯 Selected {len(selected_videos)} YouTube video(s) ranked by {ranking_labels[ranking_mode]}:")

    for video in selected_videos:
        print(f"\n🎬 Video: {video['title']}")
        print(f"👤 Channel: {video['channel']} | 📅 Date: {video['published_at'][:10]} | 👁️ Views: {video['views']:,} (VPH: {video['views_per_hour']})")
        print(f"💬 YouTube reports {video['comments_count']} comments...")
        
        comments = fetch_comments(video["id"], max_comments=max_comments_per_video)
        print(f"📥 Fetched {len(comments)} comments. Analyzing with AI...")
        
        analysis = analyze_discourse_with_ai(comments, video["title"], niche)
        camp_counts = pd.Series(result["stance"] for result in analysis).value_counts()
        camp_counts = camp_counts[camp_counts.index.str.match(r"^CAMP_[A-E]$")]
        total_camp_comments = camp_counts.sum()
        camp_shares = (
            "; ".join(
                f"{camp}: {count / total_camp_comments * 100:.1f}%"
                for camp, count in camp_counts.items()
            )
            if total_camp_comments
            else "No identified camps"
        )
        total_comments = video["comments_count"]
        analyzed_comments_count = len(analysis)
        coverage_percentage = (
            min(analyzed_comments_count / total_comments * 100, 100)
            if total_comments > 0
            else 0.0
        )
        coverage_label = (
            f"{coverage_percentage:.1f}% ({analyzed_comments_count}/{total_comments})"
        )
        
        for result in analysis:
            likes = result["like_count"]
            replies = result["reply_count"]
            is_rep = 1 if replies > 0 or result.get("is_reply", False) else 0
            eng_score = likes + (replies * 2)
            txt_len = len(result["text"])

            csv_rows.append([
                video["id"], video["title"], video["channel_id"], video["channel"], video["published_at"][:10],
                result["comment_id"], result["text"].replace("\n", " "), result["author"], 
                (result.get("published_at") or "")[:10], likes, replies, is_rep, eng_score,
                result["stance"], result["arousal_score"],
                result["emotion_tag"], result["core_argument"],
                f"https://www.youtube.com/watch?v={video['id']}",
                coverage_label,
                camp_shares
            ])
            
            result["video_id"] = video["id"]
            result["is_reply"] = bool(is_rep)
            result["engagement_score"] = eng_score
            analyzed_comments.append(result)

if not csv_rows:
    raise RuntimeError("No comments were collected or analyzed.")

# 6. Write the analysis to CSV files inside an 'exports' folder.
os.makedirs("exports", exist_ok=True)
safe_niche_name = "".join(char for char in niche if char.isalnum() or char in (" ", "_")).rstrip().replace(" ", "_").lower()
base_output_name = f"exports/discourse_{safe_niche_name}"

output_comments_csv = f"{base_output_name}_comments.csv"
output_videos_csv = f"{base_output_name}_videos.csv"
output_channels_csv = f"{base_output_name}_channels.csv"
output_camps_csv = f"{base_output_name}_camps.csv"

suffix = 1
while os.path.exists(output_comments_csv):
    output_comments_csv = f"{base_output_name}_comments_{suffix}.csv"
    output_videos_csv = f"{base_output_name}_videos_{suffix}.csv"
    output_channels_csv = f"{base_output_name}_channels_{suffix}.csv"
    output_camps_csv = f"{base_output_name}_camps_{suffix}.csv"
    suffix += 1

# Map channels to channel_key (1, 2, 3...)
unique_channels = {}
for video in selected_videos:
    ch_id = video["channel_id"]
    if ch_id not in unique_channels:
        unique_channels[ch_id] = {
            "channel_key": len(unique_channels) + 1,
            "channel_id": ch_id,
            "channel_title": video["channel"],
            "custom_url": video.get("channel_custom_url", ""),
            "country": video.get("channel_country", ""),
            "subscriber_count": video.get("subscribers", 0),
            "video_count": video.get("channel_video_count", 0),
            "view_count": video.get("channel_view_count", 0)
        }

# Attach channel_key to video dictionaries
for video in selected_videos:
    ch_id = video["channel_id"]
    video["channel_key"] = unique_channels[ch_id]["channel_key"]

# Map videos to video_key (1, 2, 3...)
video_key_map = {}
for idx, video in enumerate(selected_videos, start=1):
    video_key_map[video["id"]] = idx

# 3. Zapis tabeli kanałów (zaczynając od channel_key)
channel_export_rows = [
    [
        ch["channel_key"],
        ch["channel_id"],
        ch["channel_title"],
        ch["custom_url"],
        ch["country"],
        ch["subscriber_count"],
        ch["video_count"],
        ch["view_count"]
    ]
    for ch in unique_channels.values()
]

with open(output_channels_csv, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.writer(output_file, delimiter=";")
    writer.writerow([
        "Channel Key", "Channel ID", "Channel Title", "Custom URL", "Country", 
        "Subscriber Count", "Video Count", "Total View Count"
    ])
    writer.writerows(channel_export_rows)

# 2. Zapis tabeli wymiarów (filmy z video_key, channel_key, bez channel_id i bez channel title)
video_export_rows = []
for video in selected_videos:
    v_key = video_key_map[video["id"]]
    video_export_rows.append([
        v_key,
        video["id"],
        video["title"],
        video["channel_key"],
        video["published_at"][:10],
        video["views"],
        video.get("like_count", 0),
        video["subscribers"],
        video["views_per_hour"],
        video["comments_count"],
        video.get("duration", ""),
        video.get("tags", ""),
        video.get("description", "")
    ])

with open(output_videos_csv, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.writer(output_file, delimiter=";")
    writer.writerow([
        "Video Key", "Video ID", "Title", "Channel Key", "Published Date", 
        "Views", "Likes", "Subscribers", "Views Per Hour (VPH)", "Total Comments",
        "Duration", "Tags", "Description"
    ])
    writer.writerows(video_export_rows)

# 1. Zapis tabeli faktów (komentarze z comment_key na początku, video_key zaraz za comment_id, odchudzona)
comments_export_rows = []
for idx, row in enumerate(csv_rows, start=1):
    v_id = row[0]
    v_key = video_key_map.get(v_id, 1)
    comments_export_rows.append([
        idx,     # Comment Key
        row[5],  # Comment ID
        v_key,   # Video Key
        row[6],  # Comment
        row[7],  # Author
        row[8],  # Comment Date
        row[9],  # Likes
        row[10], # Reply Count
        row[11], # Is Reply
        row[12], # Engagement Score
        row[13], # Stance
        row[14], # Arousal Score
        row[15], # Emotion
        row[16]  # Core Argument
    ])

with open(output_comments_csv, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.writer(output_file, delimiter=";")
    writer.writerow([
        "Comment Key", "Comment ID", "Video Key", "Comment", "Author", "Comment Date", 
        "Likes", "Reply Count", "Is Reply", "Engagement Score",
        "Stance", "Arousal Score", "Emotion", "Core Argument"
    ])
    writer.writerows(comments_export_rows)

# 4. Zapis nowego pliku wymiaru _camps.csv (szeroki układ)
camps_export_rows = []
for video in selected_videos:
    v_key = video_key_map[video["id"]]
    vid_comms = [c for c in analyzed_comments if c.get("video_id") == video["id"]]
    
    raw_def = ""
    for c in vid_comms:
        if c.get("camp_definition"):
            raw_def = c["camp_definition"]
            break
            
    camp_counts = pd.Series([c["stance"] for c in vid_comms]).value_counts(normalize=True) * 100 if vid_comms else pd.Series(dtype=float)
    
    row_data = [v_key]
    for camp_id in ["CAMP_A", "CAMP_B", "CAMP_C", "CAMP_D", "CAMP_E"]:
        def extract_clean_definition(full_text, c_id):
            if not full_text:
                return ""
            pattern = rf"{c_id}:?\s*(.*?)(?=CAMP_[A-E]|$)"
            match = re.search(pattern, str(full_text), re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().rstrip(';')
            return ""

        c_def = extract_clean_definition(raw_def, camp_id)
        c_share = float(camp_counts.get(camp_id, 0.0))
        
        row_data.append(c_def)
        row_data.append(c_share)
        
    camps_export_rows.append(row_data)

with open(output_camps_csv, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.writer(output_file, delimiter=";")
    writer.writerow([
        "Video Key",
        "Camp A", "Percent Share of Camp A",
        "Camp B", "Percent Share of Camp B",
        "Camp C", "Percent Share of Camp C",
        "Camp D", "Percent Share of Camp D",
        "Camp E", "Percent Share of Camp E"
    ])
    writer.writerows(camps_export_rows)

# Keep the dashboard database synchronized with the CSV export.
init_db()
database = get_connection()
database.execute("DELETE FROM Fact_Comments")
database.execute("DELETE FROM Dim_Clusters")
database.execute("DELETE FROM Dim_Videos")

video_frame = pd.DataFrame([
    {
        "video_id": video["id"],
        "title": video["title"],
        "channel_title": video["channel"],
        "channel_id": video["channel_id"],
        "published_at": video["published_at"][:10],
        "views": video["views"],
        "likes": 0,
        "comments_count": video["comments_count"],
        "channel_median_views": 0.0,
        "outlier_factor": video["views_per_hour"],
        "channel_subs": video["subscribers"],
    }
    for video in selected_videos
])
comment_frame = pd.DataFrame(analyzed_comments)

if not video_frame.empty:
    database.register("video_frame", video_frame)
    database.execute("INSERT INTO Dim_Videos SELECT * FROM video_frame")
if not comment_frame.empty:
    database.register("comment_frame", comment_frame)
    database.execute("""
        INSERT INTO Fact_Comments (
            comment_id, video_id, text, author, published_at, like_count,
            is_reply, stance, arousal_score, emotion_tag, core_argument,
            unresolved_question
        )
        SELECT comment_id, video_id, text, author, published_at, like_count,
               is_reply, stance, arousal_score, emotion_tag, core_argument,
               unresolved_question
        FROM comment_frame
    """)

if not video_frame.empty and not comment_frame.empty:
    cluster_frame = compute_toi_cluster_metrics(video_frame, comment_frame)
    if not cluster_frame.empty:
        database.register("cluster_frame", cluster_frame)
        database.execute("""
            INSERT INTO Dim_Clusters (
                cluster_id, video_id, topic_name, camp_a_thesis, camp_b_thesis,
                shannon_polarization, weighted_arousal, discussion_depth,
                outlier_factor, toi_score, opportunity_quadrant
            )
            SELECT cluster_id, video_id, topic_name, camp_a_thesis, camp_b_thesis,
                   shannon_polarization, weighted_arousal, discussion_depth,
                   outlier_factor, toi_score, opportunity_quadrant
            FROM cluster_frame
        """)
database.close()

print("\n" + "=" * 60)
print(f"🎉 DONE! Analyzed {len(csv_rows)} comments.")
print(f"📁 Results saved to folder 'exports/':")
print(f"   - Comments Fact Table: {output_comments_csv}")
print(f"   - Videos Dimension Table: {output_videos_csv}")
print(f"   - Channels Dimension Table: {output_channels_csv}")
print("=" * 60)
