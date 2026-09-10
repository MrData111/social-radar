---
description: A description of your rule
---

## 1. Cel i działanie programu
Program analizuje dyskurs społecznościowy (komentarze YouTube) w wybranej niszy przy użyciu AI (Gemini), wykrywając polaryzację, emocje i kluczowe stanowiska (frakcje/campy). Następnie przetwarza dane w indeks TOI (Topic Opportunity Index) i prezentuje wyniki w interaktywnym dashboardzie Streamlit do wykrywania viralowych tematów i okazji contentowych.

## 2. Tech Stack
- **Język:** Python 3.10+
- **AI / LLM:** `google-genai` (model Gemini Flash Lite)
- **API YouTube:** `google-api-python-client`
- **Przetwarzanie danych:** `pandas`, `numpy`, `duckdb`
- **Dashboard / Wizualizacja:** `streamlit`, `plotly`
- **Konfiguracja:** `python-dotenv`

## 3. Struktura projektu
- `run_pipeline.py` – Skrypt pobierający dane z YouTube, analizujący komentarze przez LLM i generujący eksporty CSV / DuckDB.
- `app.py` – Interaktywny dashboard w Streamlit (wykresy kwadrantów, statystyki, campy, wskaźniki KPI).
- `src/analytics.py` – Modulet analityczny (polaryzacja Shannona, arousal, TOI, głębokość dyskusji).
- `src/enrichment.py` – Integracja z Gemini API do klasyfikacji postaw i emocji komentarzy.
- `src/db.py` – Inicjalizacja schematu gwiazdy w bazie DuckDB (`social_radar.duckdb`).
- `exports/` – Katalog na wygenerowane raporty CSV.

## 4. Zasady i konwencje kodowania
- **Typowanie i czysty kod:** Używanie funkcji pomocniczych, adnotacji oraz dokumentacji docstring.
- **Obsługa błędów:** Odporność na limity API z mechanizmem ponawiania z wykładniczym opóźnieniem (exponential backoff).
- **Format danych:** Kodowanie UTF-8-sig dla plików CSV oraz mapowanie kolumn dla spójności raportów.
- **Architektura:** Podział na warstwę pobierania (pipeline), logiki analitycznej (`src/`) oraz prezentacji (Streamlit).