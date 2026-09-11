import os
import json
import re
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-3.5-flash-lite"

VALID_PERSPECTIVE_GROUPS = [
    "Youtuber vs Audience",
    "Creator vs Audience",
    "Experts vs Beginners",
    "Hardcore vs Casuals",
    "Faction Rivalry",
    "Community"
]


def is_temporary_rate_limit(error):
    """Return whether an AI error is likely to clear after waiting."""
    error_text = str(error).lower()
    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return status_code == 429 or any(
        phrase in error_text
        for phrase in ("429", "resource exhausted", "rate limit", "too many requests")
    )

def analyze_discourse_with_ai(comments_batch, video_title, niche_context="Albion Online"):
    """
    Classify comments by stance and affective arousal from 0.0 to 1.0, and extract relational camps.
    """
    if not comments_batch:
        return [], []
        
    payload = [{"id": c["comment_id"], "text": c["text"]} for c in comments_batch]
    
    prompt = f"""
You are an affective computing and discourse analyst studying the "{niche_context}" community.
Video under review: "{video_title}".

1. Identify between two and five meaningful factions/camps in the discussion. Use only single uppercase letters from "A" to "E" as camp identifiers (e.g., "A", "B", "C"), and use fewer labels when fewer factions exist.
For EACH identified camp, provide:
- "camp_letter": exactly one single letter from "A" to "E" (e.g., "A", "B", "C", "D", "E")
- "camp_title": a short 2-4 word headline for the camp
- "perspective_group": EXACTLY ONE value from this closed list of universal dispute archetypes: {json.dumps(VALID_PERSPECTIVE_GROUPS)}
- "camp_description": a full 1-2 sentence description of this group's stance

2. Classify each comment by community polarization, emotional intensity, and stance.
For EACH comment, return a JSON object with:
- "id": exactly the input comment ID
- "stance": exactly one single letter from "A" to "E" (i.e. "A", "B", "C", "D", or "E") representing the camp, or "NEUTRAL" if it doesn't belong to any camp
- "arousal_score": a float from 0.0 to 1.0
- "emotion_tag": exactly one of "Anger", "Frustration", "Disappointment", "Excitement", "Sarcasm", or "Neutral"
- "core_argument": a one-sentence summary of the player's substantive argument
- "unresolved_question": an important question from the comment, or null

Comments to analyze:
{json.dumps(payload, ensure_ascii=False)}

Return a valid JSON object with two keys:
- "camps": an array of camp objects as defined in step 1.
- "comments": an array of comment analysis objects as defined in step 2, containing exactly {len(comments_batch)} objects.
"""
    try:
        response = None
        for attempt in range(4):
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                break
            except Exception as error:
                if not is_temporary_rate_limit(error) or attempt == 3:
                    raise
                delay = 5 * (2 ** attempt)
                print(f"   ⏳ Gemini rate limit reached. Retrying in {delay}s...")
                time.sleep(delay)

        response_data = json.loads(response.text)
        
        if isinstance(response_data, list):
            analyzed_comments_data = response_data
            extracted_camps = []
        else:
            analyzed_comments_data = response_data.get("comments", [])
            extracted_camps = response_data.get("camps", [])

        validated_camps = []
        for camp in extracted_camps:
            c_letter = str(camp.get("camp_letter", "")).upper().replace("CAMP_", "")
            if not re.fullmatch(r"[A-E]", c_letter):
                continue
            p_group = camp.get("perspective_group", "Community")
            if p_group not in VALID_PERSPECTIVE_GROUPS:
                p_group = "Community"
            validated_camps.append({
                "camp_letter": c_letter,
                "camp_title": str(camp.get("camp_title", "General Discussion")),
                "perspective_group": p_group,
                "camp_description": str(camp.get("camp_description", ""))
            })

        # Merge the AI result with the original comment metadata.
        data_lookup = {item["id"]: item for item in analyzed_comments_data if "id" in item}
        
        results = []
        for c in comments_batch:
            ai_info = data_lookup.get(c["comment_id"], {})
            raw_stance = str(ai_info.get("stance", "")).upper().replace("CAMP_", "")
            valid_stance = raw_stance if re.fullmatch(r"[A-E]", raw_stance) else "NEUTRAL"
            results.append({
                "comment_id": c["comment_id"],
                "video_id": c.get("video_id", ""),
                "text": c["text"],
                "author": c.get("author", "Anonymous"),
                "published_at": c.get("published_at") or c.get("publishedAt"),
                "like_count": int(c.get("like_count", 0)),
                "reply_count": int(c.get("reply_count", 0)),
                "is_reply": bool(c.get("is_reply", False)),
                "stance": valid_stance,
                "arousal_score": min(max(float(ai_info.get("arousal_score", 0.0)), 0.0), 1.0),
                "emotion_tag": (
                    ai_info.get("emotion_tag")
                    if ai_info.get("emotion_tag") in {
                        "Anger", "Frustration", "Disappointment", "Excitement", "Sarcasm", "Neutral"
                    }
                    else "Neutral"
                ),
                "core_argument": ai_info.get("core_argument", "-"),
                "unresolved_question": ai_info.get("unresolved_question")
            })
        return results, validated_camps

    except Exception as e:
        print(f"   ⚠️ LLM analysis error: {e}")
        return [], []