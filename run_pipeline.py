import os
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build
import google.generativeai as genai

# 1. Wczytujemy klucze z pliku .env
load_dotenv()
yt_key = os.getenv("YOUTUBE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not yt_key or not gemini_key:
    print("❌ BŁĄD: Upewnij się, że w pliku .env masz YOUTUBE_API_KEY oraz GEMINI_API_KEY!")
    exit()

# 2. Inicjalizacja połączeń
youtube = build("youtube", "v3", developerKey=yt_key)
genai.configure(api_key=gemini_key)

NAZWA_MODELU = "gemini-3.5-flash-lite"
model = genai.GenerativeModel(
    model_name=NAZWA_MODELU,
    generation_config={"response_mime_type": "application/json"}
)

# === INTERAKTYWNE PYTANIA DLA UŻYTKOWNIKA ===
print("=" * 60)
print("🎯 SOCIAL RADAR: INTERAKTYWNA ANALIZA NASTROJÓW YOUTUBE")
print("=" * 60)

# A) Temat / Nisza
nisza = input("👉 Jakiej niszy / tematu szukasz? (np. Albion Online): ").strip()
if not nisza:
    nisza = "Albion Online"

# B) Jednostka czasu
print("\n📅 Wybierz jednostkę czasu od publikacji filmu:")
print("   [1] Dni")
print("   [2] Tygodnie")
print("   [3] Miesiące (Domyślnie)")
print("   [4] Lata")
wybor_jednostki = input("👉 Twój wybór (1-4, Enter = Miesiące): ").strip()

# C) Wartość liczbowa
ile_jednostek_str = input("👉 Ile wstecz? (np. 1 dla ostatniego miesiąca, Enter = 1): ").strip()
ile_jednostek = int(ile_jednostek_str) if ile_jednostek_str.isdigit() and int(ile_jednostek_str) > 0 else 1

# Obliczamy czas wstecz
now = datetime.now(timezone.utc)
if wybor_jednostki == "1":
    delta = timedelta(days=ile_jednostek)
    opis_czasu = f"{ile_jednostek} dni"
elif wybor_jednostki == "2":
    delta = timedelta(weeks=ile_jednostek)
    opis_czasu = f"{ile_jednostek} tyg."
elif wybor_jednostki == "4":
    delta = timedelta(days=ile_jednostek * 365)
    opis_czasu = f"{ile_jednostek} lat"
else:
    # Domyślnie miesiące (ok. 30 dni na miesiąc)
    delta = timedelta(days=ile_jednostek * 30)
    opis_czasu = f"{ile_jednostek} mies."

data_graniczna = (now - delta).strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"\n🔎 Szukam najpopularniejszych filmów dla: '{nisza}' (z ostatnich: {opis_czasu})...")


# Funkcja pobierająca komentarze
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
                komentarze.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            
            page_token = res.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        print(f"   ⚠️ Uwaga przy pobieraniu komentarzy: {e}")
    return komentarze


# Funkcja analizująca komentarze przez Gemini AI
def analizuj_komentarze_ai(lista_komentarzy, temat_glowny):
    if not lista_komentarzy:
        return []
    
    wyniki = []
    batch_size = 20
    
    for i in range(0, len(lista_komentarzy), batch_size):
        paczka = lista_komentarzy[i:i + batch_size]
        print(f"   🤖 AI analizuje komentarze {i+1} - {min(i+batch_size, len(lista_komentarzy))}...")
        
        prompt = f"""
Jesteś ekspertem ds. badania nastrojów społeczności dotyczących tematu: "{temat_glowny}".
Oto lista komentarzy:
{json.dumps(paczka, ensure_ascii=False)}

Dla KAŻDEGO komentarza stwórz analizę w formacie JSON z następującymi polami:
- "temat": krótka kategoria (np. "Gameplay", "Cena/Ekonomia", "Balans/Mechanika", "Społeczność", "Problemy techniczne", "Inne")
- "sentyment": dokładnie jedno ze słów: "pozytywny", "negatywny", "neutralny"
- "emocjonalnosc": liczba float od 0.0 do 1.0
- "typ": dokładnie jedno ze słów: "Krytyka/Problem", "Pochwała", "Sugestia zmiany", "Pytanie", "Humor/Spam"
- "tag": krótkie podsumowanie w 2-3 słowach

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
            print(f"   ❌ Błąd AI: {e}")
            for k in paczka:
                wyniki.append({"komentarz": k, "temat": "Błąd", "sentyment": "-", "emocjonalnosc": "-", "typ": "-", "tag": "-"})
        
        time.sleep(0.4)
        
    return wyniki


# 3. Wyszukujemy filmy z wybranego okresu posortowane po wyświetleniach (najpopularniejsze)
search_res = youtube.search().list(
    q=nisza,
    part="id",
    type="video",
    publishedAfter=data_graniczna,
    maxResults=15,
    order="viewCount"
).execute()

video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]

if not video_ids:
    print("❌ Nie znaleziono żadnych filmów dla podanych kryteriów.")
    exit()

# 4. Pobieramy statystyki filmów
videos_res = youtube.videos().list(
    part="snippet,statistics",
    id=",".join(video_ids)
).execute()

filmy = []
for item in videos_res.get("items", []):
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})
    
    filmy.append({
        "id": item["id"],
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "views": int(stats.get("viewCount", 0)),
        "comments_count": int(stats.get("commentCount", 0)),
        "published_at": snippet.get("publishedAt", "")[:10]
    })

# Sortujemy po liczbie komentarzy, aby analizować filmy z największą dyskusją
filmy.sort(key=lambda x: x["comments_count"], reverse=True)

# 5. Pobieranie i analiza komentarzy dla 3 topowych filmów
wybrane_filmy = filmy[:3]
wszystkie_wiersze_csv = []

print(f"\n🎯 Wybrano {len(wybrane_filmy)} filmy z największą dyskusją do analizy:")

for v in wybrane_filmy:
    print(f"\n🎬 Film: {v['title']}")
    print(f"👤 Kanał: {v['channel']} | 📅 Data: {v['published_at']} | 👁️ Wyświetlenia: {v['views']:,}")
    print(f"💬 Licznik YT wskazuje: {v['comments_count']} komentarzy...")
    
    komentarze = pobierz_komentarze(v["id"], max_do_pobrania=50)
    print(f"📥 Pobrano {len(komentarze)} komentarzy. Analiza AI...")
    
    analiza = analizuj_komentarze_ai(komentarze, nisza)
    
    for r in analiza:
        wszystkie_wiersze_csv.append([
            v["title"],
            v["channel"],
            v["published_at"],
            r["komentarz"].replace("\n", " "),
            r["temat"],
            r["sentyment"],
            r["emocjonalnosc"],
            r["typ"],
            r["tag"],
            f"https://www.youtube.com/watch?v={v['id']}"
        ])

# 6. Zapis do dynamicznego pliku CSV
bezpieczna_nazwa = "".join(c for c in nisza if c.isalnum() or c in (" ", "_")).rstrip().replace(" ", "_").lower()
PLIK_CSV = f"nastroje_{bezpieczna_nazwa}.csv"

with open(PLIK_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "Film", 
        "Kanał", 
        "Data Publikacji",
        "Komentarz", 
        "Temat", 
        "Sentyment", 
        "Emocjonalność (0.0-1.0)", 
        "Typ Wypowiedzi", 
        "Główny Tag", 
        "Link"
    ])
    writer.writerows(wszystkie_wiersze_csv)

print("\n" + "=" * 60)
print(f"🎉 GOTOWE! Przeanalizowano {len(wszystkie_dane_do_csv := wszystkie_wiersze_csv)} komentarzy.")
print(f"📁 Wyniki zapisano do pliku: {PLIK_CSV}")
print("=" * 60)