import streamlit as st
import base64
import os
import sys
import requests
import numpy as np
import pandas as pd

# ── Pipeline imports (files must be in same folder as app.py) ──
try:
    from pipeline import AtmoSoundPipeline
    from google_maps_utils import parse_google_maps_response, extract_place_id_from_url
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ══════════════════════════════
# CONFIG & SESSION STATE
# ══════════════════════════════
st.set_page_config(page_title="AtmoSound", layout="wide", initial_sidebar_state="expanded")

for key, default in {
    "page": "home",
    "result": None,
    "venue_name": "Your Venue",
    "review_count": 0,
    "url_input": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════
# LOAD PIPELINE (cached)
# ══════════════════════════════
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

# ══════════════════════════════
# DEMO FALLBACK DATA
# ══════════════════════════════
DEMO_VENUE_DATA = {
    "rating": 4.4,
    "price_level": "PRICE_LEVEL_MODERATE",
    "primary_type": "cafe",
    "neighbourhood": "Roosevelt Island",
    "review_summary": "Great coffee and cozy atmosphere for studying and focused work",
    "generative_summary": "A popular campus cafe with a focused, modern vibe and friendly staff",
    "serves_coffee": True,
    "serves_breakfast": True,
    "serves_lunch": True,
    "outdoor_seating": False,
    "good_for_groups": True,
}

DEMO_RESULT = {
    "venue_info": {
        "rating": 4.4, "price_level": 2, "primary_type": "cafe",
        "neighbourhood": "Roosevelt Island", "has_text": True,
        "active_flags": ["serves_coffee", "serves_lunch", "good_for_groups"],
    },
    "audio_profile": {
        "danceability": 0.48, "energy": 0.62, "acousticness": 0.45,
        "valence": 0.58, "instrumentalness": 0.55, "liveness": 0.16, "speechiness": 0.06,
    },
    "nearest_genres": [
        {"genre": "pop", "distance": 0.031},
        {"genre": "indie", "distance": 0.048},
        {"genre": "acoustic", "distance": 0.062},
    ],
    "playlist": pd.DataFrame([
        {"track_name": "Softcore", "artists": "The Neighbourhood", "album_name": "Hard to Imagine", "genre": "indie", "popularity": 85, "duration_ms": 206000, "energy": 0.52},
        {"track_name": "Greedy", "artists": "Tate McRae", "album_name": "Greedy", "genre": "pop", "popularity": 94, "duration_ms": 131000, "energy": 0.78},
        {"track_name": "Lovin On Me", "artists": "Jack Harlow", "album_name": "Lovin On Me", "genre": "hip-hop", "popularity": 88, "duration_ms": 138000, "energy": 0.71},
        {"track_name": "Water", "artists": "Tyla", "album_name": "Water", "genre": "afrobeat", "popularity": 91, "duration_ms": 200000, "energy": 0.80},
        {"track_name": "As It Was", "artists": "Harry Styles", "album_name": "Harry's House", "genre": "pop", "popularity": 88, "duration_ms": 167000, "energy": 0.73},
        {"track_name": "Daylight", "artists": "David Kushner", "album_name": "Daylight", "genre": "indie", "popularity": 80, "duration_ms": 182000, "energy": 0.54},
        {"track_name": "Another Love", "artists": "Tom Odell", "album_name": "Long Way Down", "genre": "acoustic", "popularity": 72, "duration_ms": 246000, "energy": 0.42},
        {"track_name": "What Was I Made For", "artists": "Billie Eilish", "album_name": "Barbie OST", "genre": "pop", "popularity": 83, "duration_ms": 222000, "energy": 0.38},
        {"track_name": "Daddy Issues", "artists": "The Neighbourhood", "album_name": "Wiped Out!", "genre": "indie", "popularity": 77, "duration_ms": 268000, "energy": 0.56},
        {"track_name": "Rolling In The Deep", "artists": "Adele", "album_name": "21", "genre": "pop", "popularity": 100, "duration_ms": 228000, "energy": 0.86},
        {"track_name": "Houdini", "artists": "Dua Lipa", "album_name": "Radical Optimism", "genre": "dance", "popularity": 95, "duration_ms": 185000, "energy": 0.84},
        {"track_name": "Beggin", "artists": "Maneskin", "album_name": "Chosen", "genre": "rock", "popularity": 91, "duration_ms": 201000, "energy": 0.89},
        {"track_name": "I Wanna Be Yours", "artists": "Arctic Monkeys", "album_name": "AM", "genre": "indie", "popularity": 82, "duration_ms": 183000, "energy": 0.60},
        {"track_name": "Lala", "artists": "Myke Towers", "album_name": "La Vida Es Una", "genre": "latin", "popularity": 80, "duration_ms": 197000, "energy": 0.74},
        {"track_name": "Paint The Town Red", "artists": "Doja Cat", "album_name": "Scarlet", "genre": "pop", "popularity": 84, "duration_ms": 231000, "energy": 0.69},
        {"track_name": "Dance The Night", "artists": "Dua Lipa", "album_name": "Barbie OST", "genre": "dance", "popularity": 88, "duration_ms": 176000, "energy": 0.91},
        {"track_name": "Paradise", "artists": "Bazzi", "album_name": "Paradise", "genre": "pop", "popularity": 78, "duration_ms": 158000, "energy": 0.65},
        {"track_name": "Skyfall", "artists": "Adele", "album_name": "Skyfall", "genre": "pop", "popularity": 86, "duration_ms": 286000, "energy": 0.48},
        {"track_name": "Midnight Rain", "artists": "Taylor Swift", "album_name": "Midnights", "genre": "pop", "popularity": 81, "duration_ms": 174000, "energy": 0.50},
        {"track_name": "Golden Hour", "artists": "JVKE", "album_name": "this is what ____ feels like", "genre": "acoustic", "popularity": 79, "duration_ms": 209000, "energy": 0.55},
    ]),
}

# ══════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════

def fetch_venue_data(url):
    """
    Fetch venue from Google Maps API.
    Returns (venue_data_dict, display_name, review_count).
    Falls back to demo data if no API key is configured.
    """
    try:
        api_key = st.secrets.get("GMAPS_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return DEMO_VENUE_DATA, "Cornell Tech Cafe", 312

    try:
        place_id = extract_place_id_from_url(url)

        if place_id is None:
            search_resp = requests.post(
                "https://places.googleapis.com/v1/places:searchText",
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": "places.id,places.displayName",
                },
                json={"textQuery": url},
                timeout=10,
            )
            places = search_resp.json().get("places", [])
            if not places:
                st.error("Couldn't find that venue. Try pasting the full Google Maps URL.")
                return None, None, None
            place_id = places[0]["id"]

        fields = ",".join([
            "displayName", "rating", "userRatingCount", "priceLevel",
            "primaryType", "types", "addressComponents",
            "reviews", "generativeSummary", "editorialSummary",
            "goodForChildren", "goodForGroups", "goodForWatchingSports",
            "allowsDogs", "liveMusic", "outdoorSeating", "reservable",
            "servesBeer", "servesCocktails", "servesWine", "servesCoffee",
            "servesBreakfast", "servesBrunch", "servesDinner",
            "servesLunch", "servesVegetarianFood", "servesDessert",
            "menuForChildren",
        ])
        resp = requests.get(
            f"https://places.googleapis.com/v1/places/{place_id}",
            headers={"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": fields},
            timeout=10,
        )
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
    """Run the ML pipeline on venue data, return result dict."""
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
            seen.add(name.lower())
            unique.append(name)
        if len(unique) >= n:
            break
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
        if k in g:
            return v
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
    "cafe": "Cafe Vibes", "coffee_shop": "Coffee Shop",
    "restaurant": "Dining", "bar": "Bar Scene",
    "italian_restaurant": "Italian", "gym": "High Energy",
    "night_club": "Club Scene", "bakery": "Bakery",
    "fast_food_restaurant": "Fast Casual",
}

def get_vibe_tags(venue_info, nearest_genres):
    tags = []
    for flag in venue_info.get("active_flags", []):
        if flag in FLAG_LABELS and len(tags) < 2:
            tags.append(FLAG_LABELS[flag])
    ptype = venue_info.get("primary_type", "")
    for k, v in PTYPE_LABELS.items():
        if k in ptype.lower() and v not in tags and len(tags) < 3:
            tags.append(v)
            break
    if nearest_genres and len(tags) < 4:
        tags.append(nearest_genres[0]["genre"].replace("-", " ").replace("_", " ").title())
    defaults = ["Modern", "Curated", "Focused", "Upbeat"]
    for d in defaults:
        if len(tags) >= 4:
            break
        if d not in tags:
            tags.append(d)
    return tags[:4]


def get_sentiment_rows(audio_profile):
    """Map audio profile values to sentiment-like display rows."""
    return [
        ("Atmosphere",   int(audio_profile.get("valence", 0.5) * 100),      "#00c97a"),
        ("Energy Level", int(audio_profile.get("energy", 0.5) * 100),       "#ff2d78"),
        ("Danceability", int(audio_profile.get("danceability", 0.5) * 100), "#888"),
        ("Acousticness", int(audio_profile.get("acousticness", 0.5) * 100), "#a855f7"),
    ]


def get_acoustic_rows(audio_profile):
    """Return rows for the Acoustic Targets panel."""
    return [
        ("Energy",           audio_profile.get("energy", 0),           audio_profile.get("energy", 0)),
        ("Valence",          audio_profile.get("valence", 0),          audio_profile.get("valence", 0)),
        ("Danceability",     audio_profile.get("danceability", 0),     audio_profile.get("danceability", 0)),
        ("Acousticness",     audio_profile.get("acousticness", 0),     audio_profile.get("acousticness", 0)),
        ("Instrumentalness", audio_profile.get("instrumentalness", 0), audio_profile.get("instrumentalness", 0)),
        ("Liveness",         audio_profile.get("liveness", 0),         audio_profile.get("liveness", 0)),
        ("Speechiness",      audio_profile.get("speechiness", 0),      audio_profile.get("speechiness", 0)),
    ]

# ══════════════════════════════
# CSS
# ══════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif; box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background-color: #0d0d0d; }

[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    transform: none !important; min-width: 244px !important; max-width: 244px !important;
    width: 244px !important; visibility: visible !important; display: block !important;
    position: relative !important; background-color: #1a1a2e !important; padding-top: 20px;
}
button[data-testid="baseButton-header"] { display: none !important; }
[data-testid="stSidebar"] button[kind="header"] { display: none !important; }
[data-testid="stSidebar"] .sidebar-section-label,
[data-testid="stSidebar"] .sidebar-item,
[data-testid="stSidebar"] span { color: white !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important; color: white !important; border: none !important;
    border-radius: 0 !important; padding: 7px 20px !important; font-size: 14px !important;
    font-weight: 400 !important; width: 100% !important; text-align: left !important;
    height: auto !important; position: relative !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #ff2d78 !important; border-radius: 20px !important;
}
[data-testid="stSidebar"] .stButton:first-child > button {
    font-size: 24px !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important; color: transparent !important;
    padding: 0 20px 20px 20px !important; border-radius: 0 !important;
    width: 100% !important; text-align: left !important;
}
[data-testid="stSidebar"] .stButton:first-child > button:hover {
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important; border-radius: 0 !important;
}
.sidebar-section-label { color: #ff2d78 !important; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; padding: 14px 20px 4px 20px; display: block; }
.sidebar-item { display: block; padding: 7px 20px; color: white !important; font-size: 14px; }

.topnav { background-color: #1a1a2e; padding: 12px 40px; display: flex; justify-content: flex-end;
    align-items: center; gap: 30px; margin-bottom: 20px; border-radius: 30px; }
.topnav a { color: white; text-decoration: none; font-size: 14px; }
.topnav .login-btn { border: 1px solid #555; color: white; padding: 7px 22px; border-radius: 20px;
    font-size: 14px; background: transparent; cursor: pointer; }
.topnav .signup-btn { background-color: #ff2d78; color: white; padding: 8px 22px; border-radius: 20px;
    font-weight: 600; font-size: 14px; border: none; cursor: pointer; }

.hero-container { position: relative; width: 100%; height: 340px; overflow: hidden; }
.hero-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.55); }
.hero-text { position: absolute; bottom: 40px; left: 50px; color: white; font-size: 36px;
    font-weight: 800; line-height: 1.2; }
.hero-text .pink { color: #ff2d78; }
.input-section { padding: 30px 50px 10px 50px; }
.input-label { color: white; font-size: 22px; font-weight: 700; margin-bottom: 16px; }
.input-label .pink { color: #ff2d78; }
.cards-section { padding: 10px 50px 40px 50px; display: flex; gap: 24px; }
.card { flex: 1; border-radius: 16px; padding: 28px 26px; }
.card-green { background-color: #2d5a3d; }
.card-teal { background-color: #2d4a6b; }
.card h3 { color: white; font-size: 20px; font-weight: 700; margin: 0 0 12px 0; }
.card p { color: white; font-size: 14px; line-height: 1.6; margin: 0; }

div[data-testid="stTextInput"] input { background-color: #1e1e1e !important; color: white !important;
    border: 1px solid #333 !important; border-radius: 10px !important; font-size: 14px !important;
    padding: 14px 18px !important; }

.venue-header { text-align: center; margin-bottom: 24px; padding-top: 20px; }
.venue-header h1 { color: white; font-size: 32px; font-weight: 800; margin: 0; }
.venue-header p { color: #aaaaaa; font-size: 14px; margin: 4px 0 0 0; }
.metric-card { background-color: #2a2a2a; border-radius: 14px; padding: 24px; text-align: center; }
.metric-card .metric-value { color: #ff2d78; font-size: 40px; font-weight: 800; margin: 0; }
.metric-card .metric-label { color: #ff2d78; font-size: 14px; font-weight: 600; margin: 4px 0 0 0; }
.box { background-color: #2a2a2a; border-radius: 14px; padding: 20px; margin-bottom: 16px; }
.box-title { color: #ff2d78; font-size: 18px; font-weight: 700; text-align: center; margin-bottom: 16px; }
.vibe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.vibe-pill { background-color: #3a3a4a; color: white; font-size: 14px; font-weight: 600;
    border-radius: 10px; padding: 12px; text-align: center; }
.sentiment-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
.sentiment-label { color: white; font-size: 13px; width: 90px; flex-shrink: 0; }
.sentiment-bar-bg { flex:1; background-color: #444; border-radius: 6px; height: 10px; overflow: hidden; }
.sentiment-bar-fill { height: 100%; border-radius: 6px; }
.sentiment-pct { color: #ff2d78; font-size: 13px; font-weight: 600; width: 38px; text-align: right; flex-shrink: 0; }
.legend { display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: #aaa; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
.acoustic-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
.acoustic-label { color: white; font-size: 13px; width: 110px; flex-shrink: 0; }
.acoustic-bar-bg { flex:1; background-color: #444; border-radius: 6px; height: 10px; overflow: hidden; }
.acoustic-bar-fill { height: 100%; border-radius: 6px; background-color: #ff2d78; }
.acoustic-val { color: #dddddd; font-size: 13px; width: 38px; text-align: right; flex-shrink: 0; }

div[data-testid="stSlider"] { padding-top:0 !important; padding-bottom:0 !important;
    margin-top:-8px !important; margin-bottom:-8px !important; }
div[data-testid="stSlider"] label, div[data-testid="stSlider"] p { color: white !important;
    font-size: 12px !important; font-family: 'Poppins',sans-serif !important; margin-bottom:0 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #ff2d78 !important;
    font-weight:700; font-size:11px !important; }
.stSlider div[role="slider"] { background-color: #ff2d78 !important; border-color: #ff2d78 !important; }
[data-testid="stSlider"] > div > div > div > div { background: #ff2d78 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #2a2a2a !important;
    border: none !important; border-radius: 0 0 14px 14px !important;
    padding: 4px 14px 14px 14px !important; margin-bottom: 16px !important; }

.playlist-banner { background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
    padding: 24px 32px; display: flex; align-items: center; gap: 24px; }
.album-art { width: 110px; height: 110px; border-radius: 10px;
    background: linear-gradient(135deg, #00bcd4, #1565c0); display: flex; flex-direction: column;
    align-items: center; justify-content: center; flex-shrink: 0; }
.album-art-label { color: white; font-size: 11px; font-weight: 800; text-align: center;
    line-height: 1.2; padding: 8px; }
.banner-info { flex: 1; }
.banner-title { color: white; font-size: 22px; font-weight: 800; margin: 0 0 6px 0; }
.banner-artists { color: #cdd8e8; font-size: 13px; margin: 0 0 12px 0; }
.banner-meta { display: flex; align-items: center; gap: 20px; }
.banner-meta span { color: white; font-size: 12px; }
.play-all { color: #ff2d78; font-size: 13px; font-weight: 700; margin-left: auto;
    display: flex; align-items: center; gap: 8px; }
.play-btn { width: 30px; height: 30px; border-radius: 50%; border: 2px solid #ff2d78;
    display: flex; align-items: center; justify-content: center; color: #ff2d78; font-size: 12px; }
.col-headers { display: grid; grid-template-columns: 40px 60px 1fr 120px 200px 100px;
    padding: 10px 32px; background-color: #0d0d0d; border-bottom: 1px solid #222;
    color: #888; font-size: 12px; font-weight: 600; text-transform: uppercase; }
.track-row { display: grid; grid-template-columns: 40px 60px 1fr 120px 200px 100px;
    padding: 10px 32px; align-items: center; border-bottom: 1px solid #161616; }
.track-row:hover { background-color: #1a1a2e; }
.track-num { color: #888; font-size: 13px; text-align: center; }
.track-thumb { width: 42px; height: 42px; border-radius: 6px; background-color: #333;
    display: flex; align-items: center; justify-content: center; font-size: 18px; }
.track-info { padding-left: 12px; }
.track-title { color: white; font-size: 13px; font-weight: 600; margin: 0; }
.track-artist { color: #888; font-size: 11px; margin: 2px 0 0 0; }
.track-genre { color: #a855f7; font-size: 12px; text-align: center; font-weight: 600; }
.track-album { color: #888; font-size: 12px; text-align: center; }
.track-actions { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.track-actions .heart { color: #888; font-size: 14px; }
.track-actions .duration { color: #888; font-size: 12px; }
.track-actions .dots { color: #888; font-size: 14px; }
.bottom-fade { height: 80px; background: linear-gradient(to bottom, #0d0d0d, #0d2a4a); }
.pipeline-badge { background: #1a1a2e; border: 1px solid #ff2d78; color: #ff2d78;
    font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px;
    display: inline-block; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════
# SIDEBAR
# ══════════════════════════════
with st.sidebar:
    if st.button("AtmoSound", key="logo_btn"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown('<span class="sidebar-section-label">Menu</span>', unsafe_allow_html=True)
    if st.button("🏠 Home", key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    if st.button("ℹ️ About Us", key="nav_about"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown('<span class="sidebar-section-label">Library</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🕐 Recently Generated</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">▶️ Most Played</span>', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section-label">Playlist and Favorite</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">♡ Your Favorites</span>', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section-label">General</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">⚙️ Settings</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🚪 Logout</span>', unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="topnav">
        <a href="#">About Us</a>
        <a href="#">Contact</a>
        <a href="#">How It Works</a>
        <button class="login-btn">Login</button>
        <button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    img_path = "Gym pic.webp"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/webp;base64,{img_b64}" />'
    else:
        img_tag = '<div style="width:100%;height:340px;background:#1a1a2e;display:flex;align-items:center;justify-content:center;color:#555;">[ Hero Image ]</div>'

    st.markdown(f"""
    <div class="hero-container">
        {img_tag}
        <div class="hero-overlay"></div>
        <div class="hero-text">The <span class="pink">Perfect Playlist</span><br>with one click</div>
    </div>
    <div class="input-section">
        <div class="input-label">Just paste your venue's <span class="pink">Google Maps link</span> here!</div>
    </div>""", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input(
            "url",
            placeholder="https://www.google.com/maps/search/cornell+tech+cafe",
            label_visibility="collapsed",
            key="url_field",
        )
    with col_btn:
        go = st.button("Generate Playlist", key="gen_btn", use_container_width=True)

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: #ff2d78 !important; color: white !important;
        border: none !important; border-radius: 10px !important;
        font-size: 15px !important; font-weight: 600 !important;
        padding: 14px !important; position: relative !important; height: auto !important;
    }
    </style>""", unsafe_allow_html=True)

    if go and url.strip():
        with st.spinner("🔍 Looking up venue..."):
            venue_data, venue_name, review_count = fetch_venue_data(url.strip())

        if venue_data is not None:
            with st.spinner("🎵 Generating your playlist..."):
                result = run_pipeline(venue_data)

            # Store everything in session state
            st.session_state.result = result
            st.session_state.venue_name = venue_name or "Your Venue"
            st.session_state.review_count = review_count or 0
            st.session_state.page = "statistics"
            st.rerun()

    st.markdown("""
    <div class="cards-section">
        <div class="card card-green">
            <h3>Zero Setup</h3>
            <p>Just paste your Google Maps link. The model reads your venue data and does the rest automatically.</p>
        </div>
        <div class="card card-teal">
            <h3>Curated for You!</h3>
            <p>Music adapts to your venue's atmosphere, using ML to predict the perfect audio profile from real venue data.</p>
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
    subtitle = " — ".join(subtitle_parts) + (f" — {rating} stars" if rating else "")

    st.markdown(f"""
    <div class="topnav">
        <a href="#">About Us</a>
        <a href="#">Contact</a>
        <a href="#">How It Works</a>
        <button class="login-btn">Login</button>
        <button class="signup-btn">Sign Up</button>
    </div>
    <div class="venue-header">
        <h1>{venue_name}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

    # ── Metric Cards ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><p class="metric-value">{rating}</p><p class="metric-label">Avg Rating</p></div>', unsafe_allow_html=True)
    with m2:
        rc_display = f"{review_count:,}" if review_count else "N/A"
        st.markdown(f'<div class="metric-card"><p class="metric-value">{rc_display}</p><p class="metric-label">Reviews</p></div>', unsafe_allow_html=True)
    with m3:
        # Derive peak busyness proxy from energy + danceability
        peak = int((audio_profile.get("energy", 0.5) * 0.6 + audio_profile.get("danceability", 0.5) * 0.4) * 100)
        st.markdown(f'<div class="metric-card"><p class="metric-value">{peak}%</p><p class="metric-label">Energy Score</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    # ── LEFT COLUMN: Vibe Tags + Sentiment ──
    with left_col:
        vibe_tags = get_vibe_tags(venue_info, nearest_genres)
        vibe_pills = "".join([f'<div class="vibe-pill">{t}</div>' for t in vibe_tags])

        sentiment_rows = get_sentiment_rows(audio_profile)
        sentiment_html = ""
        for label, pct, color in sentiment_rows:
            sentiment_html += f'<div class="sentiment-row"><span class="sentiment-label">{label}</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:{pct}%;background:{color};"></div></div><span class="sentiment-pct">{pct}%</span></div>'

        st.markdown(f"""
        <div class="box">
            <div class="box-title">Vibe Tags</div>
            <div class="vibe-grid">{vibe_pills}</div>
        </div>
        <div class="box">
            <div class="box-title">Audio Profile Breakdown</div>
            {sentiment_html}
            <div class="legend">
                <span><span class="legend-dot" style="background:#00c97a;"></span>Mood</span>
                <span><span class="legend-dot" style="background:#ff2d78;"></span>Intensity</span>
                <span><span class="legend-dot" style="background:#a855f7;"></span>Texture</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── RIGHT COLUMN: Busyness Sliders + Acoustic Targets ──
    with right_col:
        st.markdown('<div style="background:#2a2a2a;border-radius:14px 14px 0 0;padding:16px 20px 10px 20px;"><div class="box-title" style="margin-bottom:0;">Busyness by Hour</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            for label, val in {"7am": 20, "9am": 75, "11am": 60, "12pm": 89, "2pm": 65, "4pm": 42}.items():
                st.slider(label, 0, 100, val, format="%d%%", key=f"busy_{label}")

        acoustic_rows = get_acoustic_rows(audio_profile)
        html = '<div class="box"><div class="box-title">Acoustic Targets</div>'
        for lbl, dv, pct in acoustic_rows:
            display_val = f"{dv:.2f}"
            html += f'<div class="acoustic-row"><span class="acoustic-label">{lbl}</span><div class="acoustic-bar-bg"><div class="acoustic-bar-fill" style="width:{int(pct*100)}%;"></div></div><span class="acoustic-val">{display_val}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("GO TO PLAYLIST", key="go_playlist", use_container_width=True):
        st.session_state.page = "playlist"
        st.rerun()

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: #00bcd4 !important; color: white !important;
        font-size: 18px !important; font-weight: 700 !important;
        border-radius: 14px !important; padding: 18px !important;
        letter-spacing: 1px !important; border: none !important;
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

    col_back, col_mid, col_home = st.columns([1, 10, 1])
    with col_back:
        if st.button("← Back", key="back_btn"):
            st.session_state.page = "statistics"
            st.rerun()
    with col_home:
        if st.button("🏠", key="playlist_home"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: white !important;
        border: none !important; font-size: 16px !important;
        font-weight: 600 !important; padding: 8px 12px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    div[data-testid="stMainBlockContainer"] .stButton > button:hover {
        background-color: #ff2d78 !important;
    }
    </style>
    <div style="background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%); height: 4px;"></div>
    """, unsafe_allow_html=True)

    # ── Banner ──
    n_songs = len(playlist_df)
    total_dur = total_duration_str(playlist_df) if not playlist_df.empty else "—"
    top_artists = get_top_artists(playlist_df) if not playlist_df.empty else "—"

    st.markdown(f"""
    <div class="playlist-banner">
        <div class="album-art">
            <div class="album-art-label">ATMO<br>SOUND</div>
            <div style="font-size:28px;">🎵</div>
        </div>
        <div class="banner-info">
            <p class="banner-title">The perfect mix for {venue_name}</p>
            <p class="banner-artists">{top_artists}</p>
            <div class="banner-meta">
                <span>{n_songs} songs</span>
                <span>{total_dur}</span>
                <div class="play-all">Play All <div class="play-btn">&#9654;</div></div>
            </div>
        </div>
    </div>
    <div class="col-headers">
        <div></div><div></div>
        <div style="padding-left:12px;">Title</div>
        <div style="text-align:center;">Genre</div>
        <div style="text-align:center;">Album</div>
        <div style="text-align:right;">Time</div>
    </div>""", unsafe_allow_html=True)

    # ── Track Rows ──
    if playlist_df.empty:
        st.markdown('<p style="color:#888;text-align:center;padding:40px;">No tracks found for this venue.</p>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for i, row in enumerate(playlist_df.itertuples(), start=1):
            title = getattr(row, "track_name", "Unknown")
            artist = getattr(row, "artists", "Unknown")
            album = getattr(row, "album_name", "—")
            genre = getattr(row, "genre", "—")
            dur_ms = getattr(row, "duration_ms", 0)
            dur_str = format_duration(dur_ms)
            emoji = genre_emoji(genre)
            genre_display = genre.replace("-", " ").replace("_", " ").title() if genre else "—"
            album_short = (album[:22] + "…") if len(str(album)) > 24 else album

            rows_html += f"""
            <div class="track-row">
                <div class="track-num">{i}</div>
                <div class="track-thumb">{emoji}</div>
                <div class="track-info">
                    <p class="track-title">{title}</p>
                    <p class="track-artist">{artist}</p>
                </div>
                <div class="track-genre">{genre_display}</div>
                <div class="track-album">{album_short}</div>
                <div class="track-actions">
                    <span class="heart">♡</span>
                    <span class="duration">{dur_str}</span>
                    <span class="dots">···</span>
                </div>
            </div>"""

        st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<div class="bottom-fade"></div>', unsafe_allow_html=True)
