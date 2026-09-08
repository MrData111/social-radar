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

if not youtube_api_key or not gemini_key:
    raise RuntimeError("Set YOUTUBE_API_KEY and GEMINI_API_KEY in .env before running the pipeline.")

# 2. Initialize the YouTube client.
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
print("🎯 SOCIAL RADAR: YOUTUBE DISCOURSE ANALYSIS")
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
print("\n🔎 Select video source:")
print("   [1] Search by niche (default)")
print("   [2] Search within one creator")
print("   [3] Analyze specific YouTube links")
source_mode = input("👉 Choose 1-3 [1]: ").strip() or "1"
if source_mode not in {"1", "2", "3"}:
    source_mode = "1"

niche = input("👉 Niche or topic [Albion Online]: ").strip() or "Albion Online"
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
print("\n📊 Select video ranking:")
print("   [1] Most comments (default)")
print("   [2] Most views")
print("   [3] Best views-to-subscribers ratio")
print("   [4] Mixed selection: subscribers + views + views/subscribers")
ranking_mode = input("👉 Choose 1-4 [1]: ").strip() or "1"
if ranking_mode not in {"1", "2", "3", "4"}:
    ranking_mode = "1"
max_videos = read_positive_integer("👉 Maximum videos to analyze", 3, 50)
max_comments_per_video = read_positive_integer("👉 Maximum comments per video", 50)
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


# 3. Find the video IDs according to the selected source mode.
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

# 4. Fetch video statistics.
videos = []
for start in range(0, len(video_ids), 50):
    video_batch = video_ids[start:start + 50]
    videos_res = execute_with_retry(
        lambda ids=video_batch: youtube.videos().list(
            part="snippet,statistics",
            id=",".join(ids)
        ),
        "YouTube video details"
    )

    for item in videos_res.get("items", []):
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})

        videos.append({
            "id": item["id"],
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "channel_id": snippet.get("channelId"),
            "views": int(stats.get("viewCount", 0)),
            "comments_count": int(stats.get("commentCount", 0)),
            "published_at": snippet.get("publishedAt", "")[:10]
        })

if videos:
    channel_ids = list({video["channel_id"] for video in videos if video["channel_id"]})
    channel_response = execute_with_retry(
        lambda: youtube.channels().list(
            part="statistics",
            id=",".join(channel_ids)
        ),
        "YouTube channel details"
    )
    subscriber_counts = {
        item["id"]: int(item.get("statistics", {}).get("subscriberCount", 0))
        for item in channel_response.get("items", [])
    }
    if max_channel_subscribers is not None:
        videos = [
            video for video in videos
            if subscriber_counts.get(video["channel_id"], 0) <= max_channel_subscribers
        ]
    for video in videos:
        subscribers = subscriber_counts.get(video["channel_id"], 0)
        video["subscribers"] = subscribers
        video["views_per_subscriber"] = (
            video["views"] / subscribers if subscribers > 0 else 0.0
        )

if not videos:
    raise RuntimeError("No videos remained after applying the search and channel filters.")

# Skip videos with no comments and rank the remaining videos.
videos = [video for video in videos if video["comments_count"] > 0]
ranking_keys = {
    "1": lambda video: video["comments_count"],
    "2": lambda video: video["views"],
    "3": lambda video: video["views_per_subscriber"],
}
if ranking_mode != "4":
    videos.sort(key=ranking_keys[ranking_mode], reverse=True)

if not videos:
    raise RuntimeError("No videos with comments remained after applying the filters.")

# 5. Fetch and analyze comments for the selected videos.
if ranking_mode == "4":
    by_subscribers = sorted(videos, key=lambda video: video["subscribers"], reverse=True)
    by_views = sorted(videos, key=lambda video: video["views"], reverse=True)
    by_ratio = sorted(videos, key=lambda video: video["views_per_subscriber"], reverse=True)

    subscriber_slots = max_videos // 2
    view_slots = max_videos // 2
    ratio_slots = max_videos - subscriber_slots - view_slots
    selected_videos = []

    for ranked_videos, slot_count in (
        (by_subscribers, subscriber_slots),
        (by_views, view_slots),
        (by_ratio, ratio_slots),
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
csv_rows = []
analyzed_comments = []

ranking_labels = {
    "1": "the most comments",
    "2": "the most views",
    "3": "the best views-to-subscribers ratio",
    "4": "the mixed selection",
}
print(f"\n🎯 Selected {len(selected_videos)} video(s) ranked by {ranking_labels[ranking_mode]}:")

for video in selected_videos:
    print(f"\n🎬 Video: {video['title']}")
    print(f"👤 Channel: {video['channel']} | 📅 Date: {video['published_at']} | 👁️ Views: {video['views']:,}")
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
        csv_rows.append([
            video["title"], video["channel"], video["published_at"],
            result["text"].replace("\n", " "), result["author"], result["like_count"],
            result["reply_count"],
            result["stance"], result["camp_definition"], result["arousal_score"],
            result["emotion_tag"], result["core_argument"],
            result["unresolved_question"] or "",
            f"https://www.youtube.com/watch?v={video['id']}",
            coverage_label,
            camp_shares
        ])
        analyzed_comments.append(result)

# 6. Write the analysis to a niche-specific CSV file.
safe_niche_name = "".join(char for char in niche if char.isalnum() or char in (" ", "_")).rstrip().replace(" ", "_").lower()
base_output_csv = f"discourse_{safe_niche_name}"
output_csv = f"{base_output_csv}.csv"
suffix = 1
while os.path.exists(output_csv):
    output_csv = f"{base_output_csv}_{suffix}.csv"
    suffix += 1

with open(output_csv, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.writer(output_file, delimiter=";")
    writer.writerow([
        "Video", "Channel", "Published Date", "Comment", "Author", "Likes", "Reply Count",
        "Stance", "Camp Definition", "Arousal Score", "Emotion",
        "Core Argument", "Unresolved Question", "Link", "Comment Analysis Coverage", "Camp Shares"
    ])
    writer.writerows(csv_rows)

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
        "published_at": video["published_at"],
        "views": video["views"],
        "likes": 0,
        "comments_count": video["comments_count"],
        "channel_median_views": 0.0,
        "outlier_factor": 1.0,
        "channel_subs": 0,
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
print(f"📁 Results saved to: {output_csv}")
print("=" * 60)