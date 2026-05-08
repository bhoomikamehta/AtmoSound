import streamlit as st
import base64
import os
import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime

try:
    from pipeline import AtmoSoundPipeline
    from google_maps_utils import parse_google_maps_response, extract_place_id_from_url
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

st.set_page_config(page_title="AtmoSound", layout="wide", initial_sidebar_state="expanded")

# ── Persistent storage helpers ──
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.json")

def load_user_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"favorites": [], "recent": []}

def save_user_data(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def add_recent(venue_name, genres, n_songs):
    data = load_user_data()
    entry = {"venue": venue_name, "genres": genres, "songs": n_songs,
             "timestamp": datetime.now().strftime("%b %d, %I:%M %p")}
    data["recent"] = [entry] + [r for r in data["recent"] if r["venue"] != venue_name][:4]
    save_user_data(data)

def toggle_favorite(venue_name, genres, n_songs):
    data = load_user_data()
    existing = [f for f in data["favorites"] if f["venue"] == venue_name]
    if existing:
        data["favorites"] = [f for f in data["favorites"] if f["venue"] != venue_name]
        saved = False
    else:
        entry = {"venue": venue_name, "genres": genres, "songs": n_songs,
                 "timestamp": datetime.now().strftime("%b %d")}
        data["favorites"] = [entry] + data["favorites"][:9]
        saved = True
    save_user_data(data)
    return saved

def is_favorite(venue_name):
    data = load_user_data()
    return any(f["venue"] == venue_name for f in data["favorites"])

# ── Session state ──
for key, default in {
    "page": "home", "result": None,
    "venue_name": "Your Venue", "review_count": 0, "url_input": "",
    "is_fav": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_pipeline():
    if not PIPELINE_AVAILABLE:
        return None
    artifacts_dir = os.path.join(os.path.dirname(__file__), "pipeline_artifacts")
    if not os.path.exists(artifacts_dir):
        return None
    try:
        return AtmoSoundPipeline(artifacts_dir)
    except Exception:
        return None

pipeline = load_pipeline()

DEMO_VENUE_DATA = {
    "rating": 4.4, "price_level": "PRICE_LEVEL_MODERATE", "primary_type": "cafe",
    "neighbourhood": "Roosevelt Island",
    "review_summary": "Great coffee and cozy atmosphere for studying and focused work",
    "generative_summary": "A popular campus cafe with a focused, modern vibe and friendly staff",
    "serves_coffee": True, "serves_breakfast": True, "serves_lunch": True,
    "outdoor_seating": False, "good_for_groups": True,
}

DEMO_RESULT = {
    "venue_info": {"rating": 4.4, "price_level": 2, "primary_type": "cafe",
        "neighbourhood": "Roosevelt Island", "has_text": True,
        "active_flags": ["serves_coffee", "serves_lunch", "good_for_groups"]},
    "audio_profile": {"danceability": 0.48, "energy": 0.62, "acousticness": 0.45,
        "valence": 0.58, "instrumentalness": 0.55, "liveness": 0.16, "speechiness": 0.06},
    "nearest_genres": [
        {"genre": "pop", "distance": 0.031}, {"genre": "indie", "distance": 0.048},
        {"genre": "acoustic", "distance": 0.062}],
    "playlist": pd.DataFrame([
        {"track_name": "Softcore", "artists": "The Neighbourhood", "album_name": "Hard to Imagine", "genre": "indie", "popularity": 85, "duration_ms": 206000, "energy": 0.52},
        {"track_name": "Greedy", "artists": "Tate McRae", "album_name": "Greedy", "genre": "pop", "popularity": 94, "duration_ms": 131000, "energy": 0.78},
        {"track_name": "As It Was", "artists": "Harry Styles", "album_name": "Harry's House", "genre": "pop", "popularity": 88, "duration_ms": 167000, "energy": 0.73},
        {"track_name": "Daylight", "artists": "David Kushner", "album_name": "Daylight", "genre": "indie", "popularity": 80, "duration_ms": 182000, "energy": 0.54},
        {"track_name": "Another Love", "artists": "Tom Odell", "album_name": "Long Way Down", "genre": "acoustic", "popularity": 72, "duration_ms": 246000, "energy": 0.42},
    ]),
}

def extract_venue_name_from_url(url):
    import re
    match = re.search(r'/maps/place/([^/@]+)', url)
    if match:
        name = match.group(1).replace('+', ' ').replace('%20', ' ')
        name = re.sub(r'%[0-9A-Fa-f]{2}', '', name)
        return name.strip()
    return None

def fetch_venue_data(url):
    try:
        api_key = st.secrets.get("GMAPS_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        return DEMO_VENUE_DATA, "Cornell Tech Cafe", 312
    try:
        venue_name = extract_venue_name_from_url(url)
        search_query = venue_name if venue_name else url
        search_resp = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={"Content-Type": "application/json", "X-Goog-Api-Key": api_key,
                     "X-Goog-FieldMask": "places.id,places.displayName"},
            json={"textQuery": search_query}, timeout=10)
        places = search_resp.json().get("places", [])
        if not places:
            st.error("Couldn't find that venue. Try pasting the full Google Maps URL.")
            return None, None, None
        place_id = places[0]["id"]
        fields = ",".join(["displayName", "rating", "userRatingCount", "priceLevel",
            "primaryType", "types", "addressComponents", "reviews", "generativeSummary",
            "editorialSummary", "goodForChildren", "goodForGroups", "goodForWatchingSports",
            "allowsDogs", "liveMusic", "outdoorSeating", "reservable", "servesBeer",
            "servesCocktails", "servesWine", "servesCoffee", "servesBreakfast", "servesBrunch",
            "servesDinner", "servesLunch", "servesVegetarianFood", "servesDessert", "menuForChildren"])
        resp = requests.get(f"https://places.googleapis.com/v1/places/{place_id}",
            headers={"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": fields}, timeout=10)
        if resp.status_code != 200:
            return DEMO_VENUE_DATA, "Your Venue", 0
        data = resp.json()
        venue_data = parse_google_maps_response(data)
        display_name = data.get("displayName", {}).get("text", "Your Venue")
        review_count = data.get("userRatingCount", 0)
        return venue_data, display_name, review_count
    except Exception as e:
        st.warning(f"API error: {e}. Showing demo data.")
        return DEMO_VENUE_DATA, "Cornell Tech Cafe", 312

def run_pipeline(venue_data):
    if pipeline is None:
        return DEMO_RESULT
    try:
        return pipeline.generate_playlist(venue_data, model="ridge", n_songs=20, seed=42)
    except Exception as e:
        st.warning(f"Pipeline error: {e}. Showing demo data.")
        return DEMO_RESULT

def format_duration(ms):
    total_sec = int(ms) // 1000
    return f"{total_sec // 60}:{total_sec % 60:02d}"

def total_duration_str(playlist_df):
    total_ms = playlist_df["duration_ms"].sum()
    total_min = int(total_ms) // 60000
    h, m = divmod(total_min, 60)
    return f"{h}h {m}m" if h > 0 else f"{m}m"

def get_top_artists(playlist_df, n=4):
    artists = playlist_df["artists"].tolist()
    seen, unique = set(), []
    for a in artists:
        name = a.split(",")[0].strip()
        if name.lower() not in seen:
            seen.add(name.lower()); unique.append(name)
        if len(unique) >= n: break
    return ", ".join(unique) + " and more..."

GENRE_EMOJI = {
    "jazz": "🎷", "rock": "🎸", "pop": "🎵", "classical": "🎻",
    "hip-hop": "🎤", "hip hop": "🎤", "electronic": "🎧", "r&b": "💜",
    "soul": "✨", "country": "🤠", "indie": "🌿", "metal": "⚡",
    "blues": "🎺", "latin": "🌺", "folk": "🪕", "dance": "🪩",
    "ambient": "🌊", "acoustic": "🎸", "afrobeat": "🥁", "reggae": "🌴",
}

def genre_emoji(genre):
    g = (genre or "").lower()
    for k, v in GENRE_EMOJI.items():
        if k in g: return v
    return "🎵"

FLAG_LABELS = {
    "live_music": "Live Music", "outdoor_seating": "Outdoor",
    "serves_cocktails": "Cocktails", "serves_wine": "Wine Bar",
    "serves_coffee": "Coffee", "serves_beer": "Beer Garden",
    "good_for_groups": "Groups", "allows_dogs": "Dog Friendly",
    "serves_breakfast": "Breakfast", "serves_brunch": "Brunch",
    "serves_dinner": "Dinner", "serves_lunch": "Lunch",
    "good_for_watching_sports": "Sports Bar", "reservable": "Reservations",
    "serves_vegetarian_food": "Veggie-Friendly",
}

PTYPE_LABELS = {
    "cafe": "Cafe Vibes", "coffee_shop": "Coffee Shop", "restaurant": "Dining",
    "bar": "Bar Scene", "italian_restaurant": "Italian", "gym": "High Energy",
    "night_club": "Club Scene", "bakery": "Bakery", "fast_food_restaurant": "Fast Casual",
}

def get_vibe_tags(venue_info, nearest_genres):
    tags = []
    for flag in venue_info.get("active_flags", []):
        if flag in FLAG_LABELS and len(tags) < 2: tags.append(FLAG_LABELS[flag])
    ptype = venue_info.get("primary_type", "")
    for k, v in PTYPE_LABELS.items():
        if k in ptype.lower() and v not in tags and len(tags) < 3:
            tags.append(v); break
    if nearest_genres and len(tags) < 4:
        tags.append(nearest_genres[0]["genre"].replace("-", " ").replace("_", " ").title())
    for d in ["Modern", "Curated", "Focused", "Upbeat"]:
        if len(tags) >= 4: break
        if d not in tags: tags.append(d)
    return tags[:4]

def get_sentiment_rows(audio_profile):
    return [
        ("Atmosphere",   int(audio_profile.get("valence", 0.5) * 100),      "#00c97a"),
        ("Energy Level", int(audio_profile.get("energy", 0.5) * 100),       "#ff2d78"),
        ("Danceability", int(audio_profile.get("danceability", 0.5) * 100), "#a78bfa"),
        ("Acousticness", int(audio_profile.get("acousticness", 0.5) * 100), "#38bdf8"),
    ]

def get_acoustic_rows(audio_profile):
    return [
        ("Energy",       audio_profile.get("energy", 0),           audio_profile.get("energy", 0)),
        ("Valence",      audio_profile.get("valence", 0),          audio_profile.get("valence", 0)),
        ("Danceability", audio_profile.get("danceability", 0),     audio_profile.get("danceability", 0)),
        ("Acousticness", audio_profile.get("acousticness", 0),     audio_profile.get("acousticness", 0)),
        ("Instrumental", audio_profile.get("instrumentalness", 0), audio_profile.get("instrumentalness", 0)),
        ("Liveness",     audio_profile.get("liveness", 0),         audio_profile.get("liveness", 0)),
        ("Speechiness",  audio_profile.get("speechiness", 0),      audio_profile.get("speechiness", 0)),
    ]

# ══════════════════════════════
# GLOBAL CSS
# ══════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Syne:wght@700;800&display=swap');
:root {
    --pink: #ff2d78; --purple: #a855f7; --cyan: #06b6d4;
    --bg: #07070f; --surface: #0e0e1c; --surface2: #14142a;
    --border: rgba(255,255,255,0.06); --text: #e2e8f0; --muted: #4a5568;
}
* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background-color: var(--bg); }

/* ── SIDEBAR ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    transform: none !important; min-width: 232px !important; max-width: 232px !important;
    width: 232px !important; visibility: visible !important; display: block !important;
    position: relative !important;
    background: #080812 !important;
    border-right: 1px solid var(--border) !important; padding-top: 0;
}
button[data-testid="baseButton-header"] { display: none !important; }
[data-testid="stSidebar"] button[kind="header"] { display: none !important; }
[data-testid="stSidebar"] .sidebar-section-label,
[data-testid="stSidebar"] .sidebar-item,
[data-testid="stSidebar"] span { color: var(--text) !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important; color: #64748b !important; border: none !important;
    border-radius: 8px !important; padding: 8px 14px !important; font-size: 13px !important;
    font-weight: 500 !important; width: 100% !important; text-align: left !important;
    height: auto !important; position: relative !important; transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,45,120,0.1) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stButton:first-child > button {
    font-family: 'Syne', sans-serif !important; font-size: 20px !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important; color: transparent !important;
    padding: 20px 16px 16px 16px !important; border-radius: 0 !important; letter-spacing: -0.02em !important;
    border-bottom: 1px solid var(--border) !important; margin-bottom: 8px !important;
}
[data-testid="stSidebar"] .stButton:first-child > button:hover {
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
.sidebar-label { color: #2d3748 !important; font-size: 9.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; padding: 14px 16px 5px 16px; display: block; }
.sidebar-item { display: flex; align-items: center; gap: 8px; padding: 8px 14px; color: #4a5568 !important; font-size: 13px; border-radius: 8px; margin: 1px 6px; }
.sidebar-item:hover { background: rgba(255,255,255,0.03); color: #94a3b8 !important; }
.sidebar-pill { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; margin: 2px 6px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); cursor: pointer; transition: all 0.15s; }
.sidebar-pill:hover { border-color: rgba(255,45,120,0.3); background: rgba(255,45,120,0.05); }
.sidebar-pill-name { color: white; font-size: 12.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px; }
.sidebar-pill-meta { color: #4a5568; font-size: 10.5px; margin-top: 1px; }
.sidebar-pill-badge { background: rgba(168,85,247,0.15); color: #a855f7; font-size: 9px; font-weight: 700; padding: 2px 7px; border-radius: 10px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.sidebar-empty { color: #2d3748; font-size: 12px; padding: 8px 14px; font-style: italic; }
.sidebar-divider { height: 1px; background: var(--border); margin: 10px 0; }

/* ── TOPNAV ── */
.topnav {
    background: rgba(7,7,15,0.95); backdrop-filter: blur(20px);
    padding: 13px 36px; display: flex; justify-content: flex-end; align-items: center;
    gap: 28px; border-bottom: 1px solid var(--border);
}
.topnav a { color: #4a5568; text-decoration: none; font-size: 13px; font-weight: 500; transition: color 0.15s; }
.topnav a:hover { color: white; }
.topnav .login-btn { border: 1px solid rgba(255,255,255,0.1); color: #94a3b8; padding: 6px 18px; border-radius: 20px; font-size: 13px; background: transparent; cursor: pointer; font-family: 'DM Sans', sans-serif; transition: all 0.15s; }
.topnav .login-btn:hover { border-color: rgba(255,255,255,0.25); color: white; }
.topnav .signup-btn { background: linear-gradient(135deg, #ff2d78, #c4004d); color: white; padding: 7px 20px; border-radius: 20px; font-weight: 600; font-size: 13px; border: none; cursor: pointer; font-family: 'DM Sans', sans-serif; box-shadow: 0 2px 12px rgba(255,45,120,0.25); }

/* ── HERO ── */
.hero-container { position: relative; width: 100%; height: 360px; overflow: hidden; }
.hero-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: linear-gradient(120deg, rgba(7,7,15,0.92) 0%, rgba(7,7,15,0.5) 55%, rgba(168,85,247,0.12) 100%); }
.hero-badge { position: absolute; top: 24px; left: 48px; background: rgba(255,45,120,0.12); border: 1px solid rgba(255,45,120,0.25); color: #ff6b9d; font-size: 10.5px; font-weight: 700; padding: 5px 13px; border-radius: 20px; letter-spacing: 1.5px; text-transform: uppercase; }
.hero-text { position: absolute; bottom: 44px; left: 48px; color: white; font-family: 'Syne', sans-serif; font-size: 40px; font-weight: 800; line-height: 1.15; letter-spacing: -0.025em; }
.hero-text .grad { background: linear-gradient(90deg, #ff2d78, #c084fc); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { position: absolute; bottom: 18px; left: 48px; color: #4a5568; font-size: 13px; font-weight: 500; letter-spacing: 0.02em; }

/* ── INPUT ── */
.input-section { padding: 32px 48px 16px 48px; }
.input-label { color: #94a3b8; font-size: 14px; font-weight: 500; margin-bottom: 14px; letter-spacing: 0.01em; }
.input-label .pink { color: var(--pink); font-weight: 600; }
div[data-testid="stTextInput"] input {
    background-color: var(--surface) !important; color: white !important;
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    font-size: 14px !important; padding: 13px 18px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(255,45,120,0.4) !important;
    box-shadow: 0 0 0 3px rgba(255,45,120,0.08) !important;
}

/* ── FEATURE CARDS ── */
.cards-section { padding: 8px 48px 48px 48px; display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }
.feat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px 22px; transition: transform 0.2s, border-color 0.2s; }
.feat-card:hover { transform: translateY(-2px); border-color: rgba(255,45,120,0.2); }
.feat-icon { font-size: 24px; margin-bottom: 14px; display: block; }
.feat-card h3 { color: white; font-size: 14px; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.01em; }
.feat-card p { color: #4a5568; font-size: 12.5px; line-height: 1.65; margin: 0; }

/* ── STATS PAGE ── */
.venue-header { text-align: center; padding: 28px 48px 20px 48px; }
.venue-header h1 { color: white; font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.025em; }
.venue-header .subtitle { color: #4a5568; font-size: 13px; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px 16px; text-align: center; position: relative; overflow: hidden; }
.metric-card::after { content: ''; position: absolute; bottom: 0; left: 20%; right: 20%; height: 1px; background: linear-gradient(90deg, transparent, var(--pink), transparent); }
.metric-value { background: linear-gradient(135deg, #ff2d78, #a855f7); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Syne', sans-serif; font-size: 38px; font-weight: 800; margin: 0; line-height: 1; }
.metric-label { color: #2d3748; font-size: 10px; font-weight: 700; margin: 8px 0 0 0; text-transform: uppercase; letter-spacing: 2px; }
.box { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; margin-bottom: 14px; }
.box-title { color: #4a5568; font-size: 10px; font-weight: 700; text-align: center; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 2.5px; }
.vibe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.vibe-pill { background: rgba(255,45,120,0.06); border: 1px solid rgba(255,45,120,0.14); color: #e2e8f0; font-size: 12.5px; font-weight: 600; border-radius: 8px; padding: 11px; text-align: center; }
.sentiment-row { display: flex; align-items: center; margin-bottom: 11px; gap: 11px; }
.sentiment-label { color: #4a5568; font-size: 11.5px; font-weight: 500; width: 92px; flex-shrink: 0; }
.sentiment-bar-bg { flex:1; background: rgba(255,255,255,0.04); border-radius: 4px; height: 6px; overflow: hidden; }
.sentiment-bar-fill { height: 100%; border-radius: 4px; }
.sentiment-pct { color: #94a3b8; font-size: 11.5px; font-weight: 700; width: 34px; text-align: right; flex-shrink: 0; }
.legend { display: flex; gap: 14px; margin-top: 12px; font-size: 10.5px; color: #2d3748; }
.legend-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 4px; }
.acoustic-row { display: flex; align-items: center; margin-bottom: 9px; gap: 10px; }
.acoustic-label { color: #4a5568; font-size: 11.5px; font-weight: 500; width: 90px; flex-shrink: 0; }
.acoustic-bar-bg { flex:1; background: rgba(255,255,255,0.04); border-radius: 4px; height: 6px; overflow: hidden; }
.acoustic-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #ff2d78, #a855f7); }
.acoustic-val { color: #64748b; font-size: 11.5px; font-weight: 600; width: 36px; text-align: right; flex-shrink: 0; }

/* ── SLIDERS ── */
div[data-testid="stSlider"] { padding-top:0 !important; padding-bottom:0 !important; margin-top:-5px !important; margin-bottom:-5px !important; }
div[data-testid="stSlider"] label, div[data-testid="stSlider"] p { color: #4a5568 !important; font-size: 11.5px !important; font-family: 'DM Sans',sans-serif !important; margin-bottom:0 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: var(--pink) !important; font-weight:700; font-size:10px !important; }
.stSlider div[role="slider"] { background-color: var(--pink) !important; border-color: var(--pink) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--pink) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 0 0 14px 14px !important; padding: 2px 12px 12px 12px !important; margin-bottom: 14px !important; }

/* ── PLAYLIST PAGE ── */
.playlist-banner { background: linear-gradient(135deg, #07071a 0%, #0a1035 60%, #070718 100%); padding: 24px 32px; display: flex; align-items: center; gap: 24px; border-bottom: 1px solid var(--border); }
.album-art { width: 110px; height: 110px; border-radius: 12px; background: linear-gradient(135deg, #ff2d78, #7c3aed); display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 6px 24px rgba(255,45,120,0.25); }
.album-art-label { color: white; font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 800; text-align: center; line-height: 1.2; }
.banner-title { color: white; font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; margin: 0 0 5px 0; letter-spacing: -0.02em; }
.banner-artists { color: #2d3748; font-size: 12.5px; margin: 0 0 12px 0; }
.banner-meta { display: flex; align-items: center; gap: 16px; }
.banner-meta span { color: #4a5568; font-size: 12.5px; }
.play-all { color: var(--pink); font-size: 12.5px; font-weight: 700; margin-left: auto; display: flex; align-items: center; gap: 7px; cursor: pointer; }
.play-btn { width: 30px; height: 30px; border-radius: 50%; background: var(--pink); display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; }
.fav-btn { background: transparent; border: 1px solid rgba(255,255,255,0.1); color: #4a5568; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 6px; font-family: 'DM Sans', sans-serif; }
.fav-btn.active { background: rgba(255,45,120,0.1); border-color: rgba(255,45,120,0.3); color: #ff2d78; }
.fav-btn:hover { border-color: rgba(255,45,120,0.3); color: #ff2d78; }
.track-item { padding: 6px 32px; border-bottom: 1px solid rgba(255,255,255,0.03); }
.track-item:hover { background: rgba(255,45,120,0.03); }
.track-header { display: flex; align-items: center; gap: 12px; padding: 8px 0 4px 0; }
.track-num-badge { color: #2d3748; font-size: 12px; font-weight: 500; width: 20px; text-align: right; flex-shrink: 0; }
.track-emoji { font-size: 16px; width: 28px; text-align: center; flex-shrink: 0; }
.track-info-inline { flex: 1; min-width: 0; }
.track-name-inline { color: white; font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.track-artist-inline { color: #4a5568; font-size: 11px; margin-top: 1px; }
.track-genre-badge { background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.2); color: #a855f7; font-size: 9.5px; font-weight: 700; padding: 3px 9px; border-radius: 10px; text-transform: uppercase; letter-spacing: 0.5px; flex-shrink: 0; }
.track-dur { color: #2d3748; font-size: 11.5px; flex-shrink: 0; width: 36px; text-align: right; }
.spotify-wrap { padding: 0 0 8px 60px; }
.bottom-fade { height: 60px; background: linear-gradient(to bottom, var(--bg), rgba(10,15,50,0.2)); }

/* ── ABOUT ── */
.about-hero { padding: 56px 48px 40px 48px; text-align: center; border-bottom: 1px solid var(--border); }
.about-hero h1 { font-family: 'Syne', sans-serif; font-size: 44px; font-weight: 800; background: linear-gradient(90deg, #ff2d78, #a855f7, #06b6d4); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 14px; letter-spacing: -0.03em; }
.about-hero p { color: #4a5568; font-size: 15px; line-height: 1.75; max-width: 580px; margin: 0 auto; }
.about-section { padding: 40px 48px; border-bottom: 1px solid var(--border); }
.about-section:last-child { border-bottom: none; }
.about-section h2 { color: white; font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 20px; letter-spacing: -0.02em; }
.about-section p { color: #4a5568; font-size: 13.5px; line-height: 1.8; margin-bottom: 14px; }
.how-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 20px; }
.how-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px 20px; }
.how-step { width: 32px; height: 32px; border-radius: 8px; background: rgba(255,45,120,0.1); border: 1px solid rgba(255,45,120,0.2); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 14px; color: var(--pink); margin-bottom: 14px; }
.how-card h3 { color: white; font-size: 14px; font-weight: 700; margin-bottom: 8px; }
.how-card p { color: #4a5568; font-size: 12.5px; line-height: 1.65; margin: 0; }
.tech-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.tech-pill { background: rgba(168,85,247,0.08); border: 1px solid rgba(168,85,247,0.18); color: #7c3aed; color: #9f67fa; font-size: 11.5px; font-weight: 600; padding: 5px 14px; border-radius: 20px; }

/* ── LIBRARY PAGES ── */
.lib-header { padding: 32px 48px 20px 48px; border-bottom: 1px solid var(--border); }
.lib-header h1 { color: white; font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.lib-header p { color: #4a5568; font-size: 13px; margin-top: 4px; }
.lib-grid { padding: 24px 48px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.lib-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; transition: transform 0.15s, border-color 0.15s; cursor: pointer; }
.lib-card:hover { transform: translateY(-2px); border-color: rgba(255,45,120,0.25); }
.lib-card-icon { width: 44px; height: 44px; border-radius: 10px; background: linear-gradient(135deg, #ff2d78, #7c3aed); display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 14px; }
.lib-card-name { color: white; font-size: 14px; font-weight: 700; margin-bottom: 4px; }
.lib-card-meta { color: #4a5568; font-size: 12px; }
.lib-card-genres { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.lib-card-genre { background: rgba(168,85,247,0.08); border: 1px solid rgba(168,85,247,0.15); color: #7c3aed; color: #9f67fa; font-size: 10px; font-weight: 700; padding: 2px 9px; border-radius: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.lib-empty { padding: 60px 48px; text-align: center; }
.lib-empty-icon { font-size: 48px; margin-bottom: 16px; }
.lib-empty-text { color: #2d3748; font-size: 15px; font-weight: 500; margin-bottom: 8px; }
.lib-empty-sub { color: #1e2433; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════
# SIDEBAR
# ══════════════════════════════
user_data = load_user_data()
recent = user_data.get("recent", [])
favorites = user_data.get("favorites", [])

with st.sidebar:
    if st.button("AtmoSound", key="logo_btn"):
        st.session_state.page = "home"; st.rerun()

    st.markdown('<span class="sidebar-label">Menu</span>', unsafe_allow_html=True)
    if st.button("🏠  Home", key="nav_home"):
        st.session_state.page = "home"; st.rerun()
    if st.button("✦  About", key="nav_about"):
        st.session_state.page = "about"; st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">Recently Generated</span>', unsafe_allow_html=True)

    if recent:
        for r in recent[:3]:
            genres_str = ", ".join(r.get("genres", [])[:2])
            st.markdown(f"""
            <div class="sidebar-pill">
                <div>
                    <div class="sidebar-pill-name">{r["venue"]}</div>
                    <div class="sidebar-pill-meta">{r["timestamp"]}</div>
                </div>
                <div class="sidebar-pill-badge">{genres_str[:12]}</div>
            </div>""", unsafe_allow_html=True)
        if st.button("See All Recent →", key="nav_recent"):
            st.session_state.page = "recent"; st.rerun()
    else:
        st.markdown('<div class="sidebar-empty">No playlists yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">Favorites</span>', unsafe_allow_html=True)

    if favorites:
        for f in favorites[:3]:
            genres_str = ", ".join(f.get("genres", [])[:2])
            st.markdown(f"""
            <div class="sidebar-pill">
                <div>
                    <div class="sidebar-pill-name">{f["venue"]}</div>
                    <div class="sidebar-pill-meta">{f["timestamp"]}</div>
                </div>
                <div class="sidebar-pill-badge">♡</div>
            </div>""", unsafe_allow_html=True)
        if st.button("See All Favorites →", key="nav_favs"):
            st.session_state.page = "favorites"; st.rerun()
    else:
        st.markdown('<div class="sidebar-empty">No favorites yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">General</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">⚙️  Settings</span>', unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="topnav">
        <a href="#">About</a><a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    img_path = "Gym pic.webp"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/webp;base64,{img_b64}" />'
    else:
        img_tag = '<div style="width:100%;height:360px;background:linear-gradient(135deg,#07070f,#14082e);display:flex;align-items:center;justify-content:center;font-size:64px;">🎵</div>'

    st.markdown(f"""
    <div class="hero-container">
        {img_tag}
        <div class="hero-overlay"></div>
        <div class="hero-badge">✦ ML-Powered</div>
        <div class="hero-text">The <span class="grad">Perfect Playlist</span><br>for any venue</div>
        <div class="hero-sub">Powered by Ridge Regression · 91,000+ Spotify tracks · Google Places API</div>
    </div>
    <div class="input-section">
        <div class="input-label">Paste your venue's <span class="pink">Google Maps link</span> to generate a playlist</div>
    </div>""", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input("url", placeholder="https://www.google.com/maps/place/...",
            label_visibility="collapsed", key="url_field")
    with col_btn:
        go = st.button("Generate  ✦", key="gen_btn", use_container_width=True)

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background: linear-gradient(135deg, #ff2d78, #c4004d) !important; color: white !important;
        border: none !important; border-radius: 12px !important; font-size: 13.5px !important;
        font-weight: 700 !important; padding: 13px !important; position: relative !important;
        height: auto !important; box-shadow: 0 3px 14px rgba(255,45,120,0.3) !important;
    }
    </style>""", unsafe_allow_html=True)

    if go and url.strip():
        with st.spinner("🔍 Looking up venue..."):
            venue_data, venue_name, review_count = fetch_venue_data(url.strip())
        if venue_data is not None:
            with st.spinner("🎵 Generating your playlist..."):
                result = run_pipeline(venue_data)
            st.session_state.result = result
            st.session_state.venue_name = venue_name or "Your Venue"
            st.session_state.review_count = review_count or 0
            st.session_state.is_fav = is_favorite(venue_name or "Your Venue")
            genres = [g["genre"] for g in result.get("nearest_genres", [])[:3]]
            n_songs = len(result.get("playlist", pd.DataFrame()))
            add_recent(venue_name or "Your Venue", genres, n_songs)
            st.session_state.page = "statistics"
            st.rerun()

    st.markdown("""
    <div class="cards-section">
        <div class="feat-card">
            <span class="feat-icon">🗺️</span>
            <h3>Zero Setup</h3>
            <p>Paste any Google Maps link. We pull venue data automatically — no manual input needed.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🧠</span>
            <h3>ML-Powered</h3>
            <p>Ridge Regression predicts your venue's audio profile from 279 real features.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🎧</span>
            <h3>Venue-Adaptive</h3>
            <p>A cafe, bar, and gym all get different playlists. The model learns from your venue's DNA.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🎵</span>
            <h3>Spotify-Ready</h3>
            <p>20 real tracks from 91,000+ songs. Preview and play directly in the app.</p>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 2 — STATISTICS
# ══════════════════════════════
elif st.session_state.page == "statistics":
    result = st.session_state.result or DEMO_RESULT
    venue_info = result["venue_info"]
    audio_profile = result["audio_profile"]
    nearest_genres = result["nearest_genres"]
    venue_name = st.session_state.venue_name
    review_count = st.session_state.review_count
    rating = venue_info.get("rating", 0)
    ptype = venue_info.get("primary_type", "").replace("_", " ").title()
    neighbourhood = venue_info.get("neighbourhood", "")
    subtitle_parts = [p for p in [ptype, neighbourhood] if p]
    subtitle = " · ".join(subtitle_parts)

    st.markdown("""
    <div class="topnav">
        <a href="#">About</a><a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="venue-header">
        <h1>{venue_name}</h1>
        <div class="subtitle">{subtitle}{f" · ⭐ {rating}" if rating else ""}</div>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><p class="metric-value">{rating}</p><p class="metric-label">Avg Rating</p></div>', unsafe_allow_html=True)
    with m2:
        rc_display = f"{review_count:,}" if review_count else "N/A"
        st.markdown(f'<div class="metric-card"><p class="metric-value">{rc_display}</p><p class="metric-label">Reviews</p></div>', unsafe_allow_html=True)
    with m3:
        peak = int((audio_profile.get("energy", 0.5) * 0.6 + audio_profile.get("danceability", 0.5) * 0.4) * 100)
        st.markdown(f'<div class="metric-card"><p class="metric-value">{peak}%</p><p class="metric-label">Energy Score</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        vibe_tags = get_vibe_tags(venue_info, nearest_genres)
        vibe_pills = "".join([f'<div class="vibe-pill">{t}</div>' for t in vibe_tags])
        sentiment_rows = get_sentiment_rows(audio_profile)
        sentiment_html = ""
        for label, pct, color in sentiment_rows:
            sentiment_html += f'<div class="sentiment-row"><span class="sentiment-label">{label}</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:{pct}%;background:{color};"></div></div><span class="sentiment-pct">{pct}%</span></div>'
        genre_pills = "".join([
            f'<span style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.2);color:#9f67fa;font-size:10.5px;font-weight:700;padding:3px 11px;border-radius:12px;margin-right:5px;text-transform:uppercase;letter-spacing:0.5px;">{g["genre"]}</span>'
            for g in nearest_genres[:3]])

        st.markdown(f"""
        <div class="box">
            <div class="box-title">Vibe Tags</div>
            <div class="vibe-grid">{vibe_pills}</div>
        </div>
        <div class="box">
            <div class="box-title">Audio Profile</div>
            {sentiment_html}
            <div class="legend">
                <span><span class="legend-dot" style="background:#00c97a;"></span>Mood</span>
                <span><span class="legend-dot" style="background:#ff2d78;"></span>Intensity</span>
                <span><span class="legend-dot" style="background:#38bdf8;"></span>Texture</span>
            </div>
        </div>
        <div class="box">
            <div class="box-title">Predicted Genres</div>
            <div style="text-align:center;padding:2px 0;">{genre_pills}</div>
        </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div style="background:var(--surface);border:1px solid var(--border);border-top-left-radius:14px;border-top-right-radius:14px;padding:16px 18px 8px 18px;"><div class="box-title" style="margin-bottom:0;">Busyness by Hour</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            for label, val in {"7am": 20, "9am": 75, "11am": 60, "12pm": 89, "2pm": 65, "4pm": 42}.items():
                st.slider(label, 0, 100, val, format="%d%%", key=f"busy_{label}")
        acoustic_rows = get_acoustic_rows(audio_profile)
        html = '<div class="box"><div class="box-title">Acoustic Targets</div>'
        for lbl, dv, pct in acoustic_rows:
            html += f'<div class="acoustic-row"><span class="acoustic-label">{lbl}</span><div class="acoustic-bar-bg"><div class="acoustic-bar-fill" style="width:{int(pct*100)}%;"></div></div><span class="acoustic-val">{dv:.2f}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("GO TO PLAYLIST  →", key="go_playlist", use_container_width=True):
        st.session_state.page = "playlist"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background: linear-gradient(135deg, #0e7490, #0c5f75) !important; color: white !important;
        font-size: 15px !important; font-weight: 700 !important; border-radius: 12px !important;
        padding: 16px !important; letter-spacing: 2px !important; border: none !important;
        position: relative !important; height: auto !important;
    }
    </style>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 3 — PLAYLIST
# ══════════════════════════════
elif st.session_state.page == "playlist":
    result = st.session_state.result or DEMO_RESULT
    playlist_df = result["playlist"]
    venue_name = st.session_state.venue_name
    nearest_genres = result.get("nearest_genres", [])
    genres = [g["genre"] for g in nearest_genres[:3]]

    col_back, col_mid, col_home = st.columns([1, 10, 1])
    with col_back:
        if st.button("← Back", key="back_btn"):
            st.session_state.page = "statistics"; st.rerun()
    with col_home:
        if st.button("🏠", key="playlist_home"):
            st.session_state.page = "home"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: #4a5568 !important;
        border: 1px solid rgba(255,255,255,0.06) !important; font-size: 13px !important;
        font-weight: 600 !important; padding: 7px 14px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    div[data-testid="stMainBlockContainer"] .stButton > button:hover {
        background-color: rgba(255,45,120,0.08) !important; color: var(--pink) !important;
        border-color: rgba(255,45,120,0.2) !important;
    }
    </style>""", unsafe_allow_html=True)

    n_songs = len(playlist_df)
    total_dur = total_duration_str(playlist_df) if not playlist_df.empty else "—"
    top_artists = get_top_artists(playlist_df) if not playlist_df.empty else "—"
    is_fav = is_favorite(venue_name)
    fav_class = "active" if is_fav else ""
    fav_label = "♥ Saved" if is_fav else "♡ Save Playlist"

    st.markdown(f"""
    <div class="playlist-banner">
        <div class="album-art">
            <div class="album-art-label">ATMO<br>SOUND</div>
            <div style="font-size:26px;margin-top:4px;">🎵</div>
        </div>
        <div style="flex:1;">
            <p class="banner-title">The perfect mix for {venue_name}</p>
            <p class="banner-artists">{top_artists}</p>
            <div class="banner-meta">
                <span>🎵 {n_songs} songs</span>
                <span>⏱ {total_dur}</span>
                <div class="play-all">Play All &nbsp;<div class="play-btn">&#9654;</div></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Favorite button
    st.markdown("<div style='padding:12px 32px 4px 32px;'>", unsafe_allow_html=True)
    if st.button(fav_label, key="fav_btn"):
        saved = toggle_favorite(venue_name, genres, n_songs)
        st.session_state.is_fav = saved
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if playlist_df.empty:
        st.markdown('<p style="color:#2d3748;text-align:center;padding:60px;">No tracks found.</p>', unsafe_allow_html=True)
    else:
        for i, row in enumerate(playlist_df.itertuples(), start=1):
            title = getattr(row, "track_name", "Unknown")
            artist = getattr(row, "artists", "Unknown")
            genre = getattr(row, "genre", "—")
            dur_ms = getattr(row, "duration_ms", 0)
            dur_str = format_duration(dur_ms)
            emoji = genre_emoji(genre)
            genre_display = genre.replace("-", " ").replace("_", " ").title() if genre else "—"
            track_id = getattr(row, "track_id", None)

            # Show track header
            st.markdown(f"""
            <div class="track-item">
                <div class="track-header">
                    <span class="track-num-badge">{i}</span>
                    <span class="track-emoji">{emoji}</span>
                    <div class="track-info-inline">
                        <div class="track-name-inline">{title}</div>
                        <div class="track-artist-inline">{artist}</div>
                    </div>
                    <span class="track-genre-badge">{genre_display}</span>
                    <span class="track-dur">{dur_str}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Show Spotify embed ONLY (no duplicate)
            if track_id and str(track_id) != "nan":
                st.markdown(
                    f'<div class="spotify-wrap"><iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" '
                    f'width="100%" height="80" frameborder="0" allowtransparency="true" '
                    f'allow="encrypted-media" style="border-radius:8px;"></iframe></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="bottom-fade"></div>', unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════
elif st.session_state.page == "about":
    st.markdown("""
    <div class="topnav">
        <a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    col_b, _ = st.columns([1, 11])
    with col_b:
        if st.button("← Home", key="about_back"):
            st.session_state.page = "home"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: #4a5568 !important;
        border: 1px solid rgba(255,255,255,0.06) !important; font-size: 13px !important;
        font-weight: 600 !important; padding: 7px 14px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
        <h1>About AtmoSound</h1>
        <p>An ML-powered playlist generation system that creates venue-adaptive music recommendations using real Google Maps data, Ridge Regression, and Neural Networks — trained on 4,484 Manhattan venues and 91,000+ Spotify tracks.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section">
        <h2>⚙️ How It Works</h2>
        <div class="how-grid">
            <div class="how-card">
                <div class="how-step">1</div>
                <h3>Venue Lookup</h3>
                <p>Paste any Google Maps URL. We fetch live data — rating, price level, venue type, neighbourhood, reviews, and 15+ boolean attributes like outdoor seating and live music.</p>
            </div>
            <div class="how-card">
                <div class="how-step">2</div>
                <h3>Audio Profile Prediction</h3>
                <p>Venue data becomes a 279-dimensional feature vector via TF-IDF, SVD, and one-hot encoding. Ridge Regression predicts a 7-dimensional audio profile: energy, valence, danceability, acousticness, instrumentalness, liveness, and speechiness.</p>
            </div>
            <div class="how-card">
                <div class="how-step">3</div>
                <h3>Playlist Generation</h3>
                <p>We find the 5 nearest Spotify genre clusters using cosine distance, then sample 20 tracks weighted by popularity and audio proximity. Songs are arranged in an energy arc — building to peak, then cooling down.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section">
        <h2>🧪 ML Models</h2>
        <p>All models are implemented from scratch using NumPy — no scikit-learn, TensorFlow, or PyTorch.</p>
        <div class="how-grid">
            <div class="how-card">
                <div class="how-step">λ</div>
                <h3>Ridge Regression</h3>
                <p>Closed-form W* = (XᵀX + λI)⁻¹Xᵀy with L2 regularization. Optimal for high-dimensional sparse feature spaces. MSE = 0.14, CosSim = 0.65.</p>
            </div>
            <div class="how-card">
                <div class="how-step">∿</div>
                <h3>Neural Network</h3>
                <p>Two hidden layers (256, 128) with ReLU activations, dropout = 0.2, mini-batch SGD. Grid search over 54 configurations. MSE = 0.015, CosSim = 0.96.</p>
            </div>
            <div class="how-card">
                <div class="how-step">K</div>
                <h3>K-Means Clustering</h3>
                <p>Groups 112 Spotify genre profiles into venue archetypes. Predicted audio vectors are matched to nearest centroids using Euclidean distance, weighted for playlist sampling.</p>
            </div>
        </div>
        <div class="tech-pills">
            <span class="tech-pill">NumPy</span><span class="tech-pill">Pandas</span>
            <span class="tech-pill">Streamlit</span><span class="tech-pill">Google Places API</span>
            <span class="tech-pill">Spotify Dataset</span><span class="tech-pill">TF-IDF</span>
            <span class="tech-pill">SVD</span><span class="tech-pill">Ridge Regression</span>
            <span class="tech-pill">Neural Networks</span><span class="tech-pill">K-Means</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section">
        <h2>📊 The Data</h2>
        <p><strong style="color:white;">4,484 Manhattan venue records</strong> collected via the Google Places API. Missing boolean attributes use tri-state encoding (1 = True, 0 = False, -1 = Unknown). Review text is processed with TF-IDF + SVD compressed to 50 features.</p>
        <p>The Spotify dataset contains <strong style="color:white;">91,271 tracks</strong> across 112 genres. Pseudo-labels were generated by mapping venue types to genre groups, modulated by price level and boolean attributes.</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 5 — FAVORITES
# ══════════════════════════════
elif st.session_state.page == "favorites":
    col_b, _ = st.columns([1, 11])
    with col_b:
        if st.button("← Back", key="fav_back"):
            st.session_state.page = "home"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: #4a5568 !important;
        border: 1px solid rgba(255,255,255,0.06) !important; font-size: 13px !important;
        font-weight: 600 !important; padding: 7px 14px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    </style>""", unsafe_allow_html=True)

    data = load_user_data()
    favs = data.get("favorites", [])

    st.markdown(f"""
    <div class="lib-header">
        <h1>♥ Your Favorites</h1>
        <p>{len(favs)} saved playlist{"s" if len(favs) != 1 else ""}</p>
    </div>""", unsafe_allow_html=True)

    if not favs:
        st.markdown("""
        <div class="lib-empty">
            <div class="lib-empty-icon">♡</div>
            <div class="lib-empty-text">No favorites yet</div>
            <div class="lib-empty-sub">Generate a playlist and tap "Save Playlist" to save it here</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="lib-grid">', unsafe_allow_html=True)
        for f in favs:
            genres_html = "".join([f'<span class="lib-card-genre">{g}</span>' for g in f.get("genres", [])[:3]])
            st.markdown(f"""
            <div class="lib-card">
                <div class="lib-card-icon">🎵</div>
                <div class="lib-card-name">{f["venue"]}</div>
                <div class="lib-card-meta">Saved {f["timestamp"]} · {f.get("songs", "—")} songs</div>
                <div class="lib-card-genres">{genres_html}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 6 — RECENTLY GENERATED
# ══════════════════════════════
elif st.session_state.page == "recent":
    col_b, _ = st.columns([1, 11])
    with col_b:
        if st.button("← Back", key="rec_back"):
            st.session_state.page = "home"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: #4a5568 !important;
        border: 1px solid rgba(255,255,255,0.06) !important; font-size: 13px !important;
        font-weight: 600 !important; padding: 7px 14px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    </style>""", unsafe_allow_html=True)

    data = load_user_data()
    recent_all = data.get("recent", [])

    st.markdown(f"""
    <div class="lib-header">
        <h1>🕐 Recently Generated</h1>
        <p>{len(recent_all)} playlist{"s" if len(recent_all) != 1 else ""} generated</p>
    </div>""", unsafe_allow_html=True)

    if not recent_all:
        st.markdown("""
        <div class="lib-empty">
            <div class="lib-empty-icon">🕐</div>
            <div class="lib-empty-text">No playlists generated yet</div>
            <div class="lib-empty-sub">Paste a Google Maps link on the home page to get started</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="lib-grid">', unsafe_allow_html=True)
        for r in recent_all:
            genres_html = "".join([f'<span class="lib-card-genre">{g}</span>' for g in r.get("genres", [])[:3]])
            st.markdown(f"""
            <div class="lib-card">
                <div class="lib-card-icon">🎧</div>
                <div class="lib-card-name">{r["venue"]}</div>
                <div class="lib-card-meta">{r["timestamp"]} · {r.get("songs", "—")} songs</div>
                <div class="lib-card-genres">{genres_html}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
