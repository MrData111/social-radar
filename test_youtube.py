import os
import csv
import json
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build
import google.generativeai as genai

# 1. Wczytujemy klucze z pliku .env
load_dotenv()
yt_key = os.getenv("YOUTUBE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not yt_key:
    print("❌ BŁĄD: Brak YOUTUBE_API_KEY w pliku .env!")
    exit()

if not gemini_key:
    print("❌ BŁĄD: Brak GEMINI_API_KEY w pliku .env!")
    exit()

# 2. Inicjalizacja połączeń
youtube = build("youtube", "v3", developerKey=yt_key)
genai.configure(api_key=gemini_key)

# Wybieramy najszybszy model z największym darmowym limitem (500 zapytań/dzień)
NAZWA_MODELU = "gemini-3.5-flash-lite"

print(f"🤖 Inicjalizacja modelu AI: {NAZWA_MODELU}...")
model = genai.GenerativeModel(
    model_name=NAZWA_MODELU,
    generation_config={"response_mime_type": "application/json"}
)


# Funkcja pobierająca komentarze pod filmem
def pobierz_komentarze(video_id, max_do_pobrania=50):
    komentarze = []
    page_token = None
    try:
        while len(komentarze) < max_do_pobrania:
            res = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_do_pobrania - len(komentarze)),
                textFormat="plainText",
                pageToken=page_token
            ).execute()
            
            for item in res.get("items", []):
                tekst = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                komentarze.append(tekst)
            
            page_token = res.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        print(f"   ⚠️ Uwaga przy pobieraniu komentarzy: {e}")
    return komentarze


# Funkcja wysyłająca paczki komentarzy do analizy przez AI
def analizuj_komentarze_ai(lista_komentarzy):
    if not lista_komentarzy:
        return []
    
    wyniki = []
    batch_size = 20  # Analiza po 20 komentarzy na raz
    
    for i in range(0, len(lista_komentarzy), batch_size):
        paczka = lista_komentarzy[i:i + batch_size]
        print(f"   🤖 AI analizuje komentarze {i+1} do {min(i+batch_size, len(lista_komentarzy))}...")
        
        prompt = f"""
Jesteś analitykiem nastrojów społeczności gry Albion Online.
Oto lista komentarzy graczy:
{json.dumps(paczka, ensure_ascii=False)}

Dla KAŻDEGO komentarza stwórz analizę w formacie JSON z następującymi polami:
- "temat": krótka kategoria (np. "Balans broni", "PvP", "Ekonomia", "Serwery/Lagi", "Nowy Patch", "Inne")
- "sentyment": dokładnie jedno ze słów: "pozytywny", "negatywny", "neutralny"
- "emocjonalnosc": liczba float od 0.0 do 1.0 (gdzie 0.0 to spokój/fakt, a 1.0 to skrajne emocje)
- "typ": dokładnie jedno ze słów: "Krytyka/Problem", "Pochwała", "Sugestia zmiany", "Pytanie", "Humor/Spam"
- "tag": krótkie podsumowanie w 2-3 słowach (np. "nerf toporów", "drogie premium")

Zwróć tablicę JSON zawierającą dokładnie {len(paczka)} obiektów.
"""
        try:
            response = model.generate_content(prompt)
            dane_json = json.loads(response.text)
            
            for idx, item in enumerate(dane_json):
                wyniki.append({
                    "komentarz": paczka[idx],
                    "temat": item.get("temat", "Inne"),
                    "sentyment": item.get("sentyment", "neutralny"),
                    "emocjonalnosc": item.get("emocjonalnosc", 0.0),
                    "typ": item.get("typ", "Inne"),
                    "tag": item.get("tag", "-")
                })
        except Exception as e:
            print(f"   ❌ Błąd AI dla tej paczki: {e}")
            for k in paczka:
                wyniki.append({"komentarz": k, "temat": "Błąd", "sentyment": "-", "emocjonalnosc": "-", "typ": "-", "tag": "-"})
        
        time.sleep(0.5)  # Krótka pauza dla stabilności API
        
    return wyniki


# === GŁÓWNY PROGRAM ===
print("\n🔍 1. Szukam najpopularniejszych filmów o Albion Online...")

search_res = youtube.search().list(
    q="Albion Online",
    part="id",
    type="video",
    maxResults=10,
    order="viewCount"
).execute()

video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]

# Pobieramy szczegółowe statystyki (ilość komentarzy)
videos_res = youtube.videos().list(
    part="snippet,statistics",
    id=",".join(video_ids)
).execute()

filmy = []
for item in videos_res.get("items", []):
    stats = item.get("statistics", {})
    filmy.append({
        "id": item["id"],
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "comments_count": int(stats.get("commentCount", 0))
    })

# Sortujemy filmy od tych z największą dyskusją
filmy.sort(key=lambda x: x["comments_count"], reverse=True)

wszystkie_dane_do_csv = []
wybrane_filmy = filmy[:3]  # Analizujemy 3 filmy z największą liczbą opinii

print(f"\n🎯 Wybrano {len(wybrane_filmy)} filmy z największą dyskusją.")

for v in wybrane_filmy:
    print(f"\n🎬 Film: {v['title']}")
    print(f"👤 Kanał: {v['channel']} | 💬 Komentarzy na YT: {v['comments_count']}")
    
    # Pobieramy treść komentarzy (np. do 50 sztuk na film)
    komentarze = pobierz_komentarze(v["id"], max_do_pobrania=50)
    print(f"📥 Pobrano {len(komentarze)} komentarzy. Rozpoczynam analizę AI...")
    
    analiza = analizuj_komentarze_ai(komentarze)
    
    for r in analiza:
        wszystkie_dane_do_csv.append([
            v["title"],
            v["channel"],
            r["komentarz"].replace("\n", " "),
            r["temat"],
            r["sentyment"],
            r["emocjonalnosc"],
            r["typ"],
            r["tag"],
            f"https://www.youtube.com/watch?v={v['id']}"
        ])

# 3. Zapisujemy wyniki do pliku CSV
folder_projektu = os.path.dirname(os.path.abspath(__file__))
PLIK_CSV = os.path.join(folder_projektu, "nastroje_spolecznosci_albion.csv")

with open(PLIK_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")  # Średnik dla polskiego Excela
    writer.writerow([
        "Film", 
        "Kanał", 
        "Komentarz", 
        "Temat", 
        "Sentyment", 
        "Emocjonalność (0.0-1.0)", 
        "Typ Wypowiedzi", 
        "Główny Tag", 
        "Link"
    ])
    writer.writerows(wszystkie_dane_do_csv)

print(f"\n🎉 SUKCES! Przeanalizowano i zapisano {len(wszystkie_dane_do_csv)} komentarzy.")
print(f"📁 Plik CSV znajdziesz tutaj:\n   {PLIK_CSV}")