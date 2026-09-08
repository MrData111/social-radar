import os
import json
import re
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-3.5-flash-lite"


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
    Classify comments by stance and affective arousal from 0.0 to 1.0.
    """
    if not comments_batch:
        return []
        
    payload = [{"id": c["comment_id"], "text": c["text"]} for c in comments_batch]
    
    prompt = f"""
You are an affective computing and discourse analyst studying the "{niche_context}" community.
Video under review: "{video_title}".

Classify each comment by community polarization and emotional intensity.

Identify between two and five meaningful factions in the discussion. Use only the
labels CAMP_A through CAMP_E, and use fewer labels when fewer factions exist.

For EACH comment, return a JSON object with:
- "id": exactly the input comment ID
- "stance": exactly one of "CAMP_A", "CAMP_B", "CAMP_C", "CAMP_D", "CAMP_E", or "NEUTRAL_SPAM"
- "camp_definition": concise definitions of all camps used in this video
- "arousal_score": a float from 0.0 to 1.0
- "emotion_tag": exactly one of "Anger", "Frustration", "Disappointment", "Excitement", "Sarcasm", or "Neutral"
- "core_argument": a one-sentence summary of the player's substantive argument
- "unresolved_question": an important question from the comment, or null

Comments to analyze:
{json.dumps(payload, ensure_ascii=False)}

Return a valid JSON array containing exactly {len(comments_batch)} objects.
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

        analyzed_data = json.loads(response.text)
        
        # Merge the AI result with the original comment metadata.
        data_lookup = {item["id"]: item for item in analyzed_data if "id" in item}
        
        results = []
        for c in comments_batch:
            ai_info = data_lookup.get(c["comment_id"], {})
            camp_definition = ai_info.get("camp_definition", "")
            if isinstance(camp_definition, dict):
                camp_definition = "; ".join(
                    f"{camp}: {definition}"
                    for camp, definition in camp_definition.items()
                )
            results.append({
                "comment_id": c["comment_id"],
                "video_id": c.get("video_id", ""),
                "text": c["text"],
                "author": c.get("author", "Anonymous"),
                "published_at": c.get("published_at") or c.get("publishedAt"),
                "like_count": int(c.get("like_count", 0)),
                "reply_count": int(c.get("reply_count", 0)),
                "is_reply": bool(c.get("is_reply", False)),
                "stance": (
                    ai_info.get("stance")
                    if re.fullmatch(r"CAMP_[A-E]", str(ai_info.get("stance", "")))
                    else "NEUTRAL_SPAM"
                ),
                "arousal_score": min(max(float(ai_info.get("arousal_score", 0.0)), 0.0), 1.0),
                "emotion_tag": (
                    ai_info.get("emotion_tag")
                    if ai_info.get("emotion_tag") in {
                        "Anger", "Frustration", "Disappointment", "Excitement", "Sarcasm", "Neutral"
                    }
                    else "Neutral"
                ),
                "core_argument": ai_info.get("core_argument", "-"),
                "unresolved_question": ai_info.get("unresolved_question"),
                "camp_definition": camp_definition
            })
        return results

    except Exception as e:
        print(f"   ⚠️ LLM analysis error: {e}")
        return []