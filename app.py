import streamlit as st
import base64
import os
import requests
import numpy as np
import pandas as pd

try:
    from pipeline import AtmoSoundPipeline
    from google_maps_utils import parse_google_maps_response, extract_place_id_from_url
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

st.set_page_config(page_title="AtmoSound", layout="wide", initial_sidebar_state="expanded")

for key, default in {
    "page": "home", "result": None,
    "venue_name": "Your Venue", "review_count": 0, "url_input": "",
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
        {"track_name": "What Was I Made For", "artists": "Billie Eilish", "album_name": "Barbie OST", "genre": "pop", "popularity": 83, "duration_ms": 222000, "energy": 0.38},
        {"track_name": "Rolling In The Deep", "artists": "Adele", "album_name": "21", "genre": "pop", "popularity": 100, "duration_ms": 228000, "energy": 0.86},
        {"track_name": "Houdini", "artists": "Dua Lipa", "album_name": "Radical Optimism", "genre": "dance", "popularity": 95, "duration_ms": 185000, "energy": 0.84},
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
:root {
    --pink: #ff2d78; --purple: #a855f7; --cyan: #06b6d4;
    --bg: #080810; --surface: #0f0f1a; --surface2: #161627;
    --border: rgba(255,255,255,0.07); --text: #e2e8f0; --muted: #64748b;
}
* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background-color: var(--bg); }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    transform: none !important; min-width: 240px !important; max-width: 240px !important;
    width: 240px !important; visibility: visible !important; display: block !important;
    position: relative !important;
    background: linear-gradient(180deg, #0a0a16 0%, #0d0d1f 100%) !important;
    border-right: 1px solid var(--border) !important; padding-top: 24px;
}
button[data-testid="baseButton-header"] { display: none !important; }
[data-testid="stSidebar"] button[kind="header"] { display: none !important; }
[data-testid="stSidebar"] .sidebar-section-label,
[data-testid="stSidebar"] .sidebar-item,
[data-testid="stSidebar"] span { color: var(--text) !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important; color: #94a3b8 !important; border: none !important;
    border-radius: 8px !important; padding: 8px 16px !important; font-size: 13.5px !important;
    font-weight: 500 !important; width: 100% !important; text-align: left !important;
    height: auto !important; position: relative !important; transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,45,120,0.12) !important;
    color: var(--pink) !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton:first-child > button {
    font-family: 'Syne', sans-serif !important; font-size: 22px !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important; color: transparent !important;
    padding: 0 16px 24px 16px !important; border-radius: 0 !important; letter-spacing: -0.02em !important;
}
[data-testid="stSidebar"] .stButton:first-child > button:hover {
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
    -webkit-text-fill-color: transparent !important; border-radius: 0 !important;
}
.sidebar-section-label { color: var(--muted) !important; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; padding: 16px 16px 6px 16px; display: block; }
.sidebar-item { display: block; padding: 8px 16px; color: #94a3b8 !important; font-size: 13.5px; }
.sidebar-divider { height: 1px; background: var(--border); margin: 8px 16px; }
.topnav { background: rgba(10,10,22,0.9); backdrop-filter: blur(12px); padding: 14px 40px; display: flex; justify-content: flex-end; align-items: center; gap: 32px; border-bottom: 1px solid var(--border); }
.topnav a { color: #94a3b8; text-decoration: none; font-size: 13.5px; font-weight: 500; transition: color 0.15s; }
.topnav a:hover { color: white; }
.topnav .login-btn { border: 1px solid rgba(255,255,255,0.15); color: white; padding: 7px 20px; border-radius: 20px; font-size: 13px; background: transparent; cursor: pointer; font-family: 'DM Sans', sans-serif; }
.topnav .signup-btn { background: linear-gradient(135deg, #ff2d78, #e8004d); color: white; padding: 8px 22px; border-radius: 20px; font-weight: 600; font-size: 13px; border: none; cursor: pointer; font-family: 'DM Sans', sans-serif; box-shadow: 0 4px 15px rgba(255,45,120,0.3); }
.hero-container { position: relative; width: 100%; height: 380px; overflow: hidden; }
.hero-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: linear-gradient(135deg, rgba(8,8,16,0.88) 0%, rgba(8,8,16,0.45) 60%, rgba(168,85,247,0.18) 100%); }
.hero-badge { position: absolute; top: 28px; left: 52px; background: rgba(255,45,120,0.15); border: 1px solid rgba(255,45,120,0.3); color: #ff2d78; font-size: 11px; font-weight: 600; padding: 5px 14px; border-radius: 20px; letter-spacing: 1px; text-transform: uppercase; }
.hero-text { position: absolute; bottom: 48px; left: 52px; color: white; font-family: 'Syne', sans-serif; font-size: 42px; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em; }
.hero-text .pink { background: linear-gradient(90deg, #ff2d78, #a855f7); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.input-section { padding: 36px 52px 20px 52px; }
.input-label { color: white; font-size: 20px; font-weight: 600; margin-bottom: 16px; letter-spacing: -0.01em; }
.input-label .pink { color: var(--pink); }
div[data-testid="stTextInput"] input { background-color: var(--surface) !important; color: white !important; border: 1px solid var(--border) !important; border-radius: 12px !important; font-size: 14px !important; padding: 14px 18px !important; }
div[data-testid="stTextInput"] input:focus { border-color: rgba(255,45,120,0.5) !important; box-shadow: 0 0 0 3px rgba(255,45,120,0.1) !important; }
.cards-section { padding: 8px 52px 52px 52px; display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 20px; }
.feat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px 24px; transition: transform 0.2s, border-color 0.2s; }
.feat-card:hover { transform: translateY(-3px); border-color: rgba(255,45,120,0.3); }
.feat-icon { font-size: 28px; margin-bottom: 16px; display: block; }
.feat-card h3 { color: white; font-size: 16px; font-weight: 700; margin: 0 0 10px 0; }
.feat-card p { color: var(--muted); font-size: 13px; line-height: 1.6; margin: 0; }
.venue-header { text-align: center; padding: 32px 52px 24px 52px; }
.venue-header h1 { color: white; font-family: 'Syne', sans-serif; font-size: 36px; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.02em; }
.venue-header .subtitle { color: var(--muted); font-size: 14px; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px 20px; text-align: center; position: relative; overflow: hidden; }
.metric-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--pink), var(--purple)); }
.metric-value { background: linear-gradient(135deg, #ff2d78, #a855f7); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Syne', sans-serif; font-size: 42px; font-weight: 800; margin: 0; line-height: 1; }
.metric-label { color: var(--muted); font-size: 11px; font-weight: 600; margin: 8px 0 0 0; text-transform: uppercase; letter-spacing: 1.5px; }
.box { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 22px; margin-bottom: 16px; }
.box-title { color: white; font-size: 12px; font-weight: 700; text-align: center; margin-bottom: 18px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.7; }
.vibe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.vibe-pill { background: linear-gradient(135deg, rgba(255,45,120,0.08), rgba(168,85,247,0.08)); border: 1px solid rgba(255,45,120,0.18); color: white; font-size: 13px; font-weight: 600; border-radius: 10px; padding: 12px; text-align: center; }
.sentiment-row { display: flex; align-items: center; margin-bottom: 12px; gap: 12px; }
.sentiment-label { color: #94a3b8; font-size: 12px; font-weight: 500; width: 95px; flex-shrink: 0; }
.sentiment-bar-bg { flex:1; background: rgba(255,255,255,0.06); border-radius: 6px; height: 8px; overflow: hidden; }
.sentiment-bar-fill { height: 100%; border-radius: 6px; }
.sentiment-pct { color: white; font-size: 12px; font-weight: 700; width: 36px; text-align: right; flex-shrink: 0; }
.legend { display: flex; gap: 16px; margin-top: 14px; font-size: 11px; color: var(--muted); }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
.acoustic-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
.acoustic-label { color: #94a3b8; font-size: 12px; font-weight: 500; width: 95px; flex-shrink: 0; }
.acoustic-bar-bg { flex:1; background: rgba(255,255,255,0.06); border-radius: 6px; height: 8px; overflow: hidden; }
.acoustic-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #ff2d78, #a855f7); }
.acoustic-val { color: #e2e8f0; font-size: 12px; font-weight: 600; width: 38px; text-align: right; flex-shrink: 0; }
div[data-testid="stSlider"] { padding-top:0 !important; padding-bottom:0 !important; margin-top:-6px !important; margin-bottom:-6px !important; }
div[data-testid="stSlider"] label, div[data-testid="stSlider"] p { color: #94a3b8 !important; font-size: 12px !important; font-family: 'DM Sans',sans-serif !important; margin-bottom:0 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: var(--pink) !important; font-weight:700; font-size:11px !important; }
.stSlider div[role="slider"] { background-color: var(--pink) !important; border-color: var(--pink) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--pink) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 0 0 16px 16px !important; padding: 4px 14px 14px 14px !important; margin-bottom: 16px !important; }
.playlist-banner { background: linear-gradient(135deg, #0a0a2e 0%, #0d1a4a 50%, #0a1a3a 100%); padding: 28px 36px; display: flex; align-items: center; gap: 28px; border-bottom: 1px solid var(--border); }
.album-art { width: 120px; height: 120px; border-radius: 12px; background: linear-gradient(135deg, #ff2d78, #a855f7); display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 8px 32px rgba(255,45,120,0.3); }
.album-art-label { color: white; font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 800; text-align: center; line-height: 1.2; }
.banner-info { flex: 1; }
.banner-title { color: white; font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; }
.banner-artists { color: #64748b; font-size: 13px; margin: 0 0 14px 0; }
.banner-meta { display: flex; align-items: center; gap: 20px; }
.banner-meta span { color: #94a3b8; font-size: 13px; }
.play-all { color: var(--pink); font-size: 13px; font-weight: 700; margin-left: auto; display: flex; align-items: center; gap: 8px; }
.play-btn { width: 32px; height: 32px; border-radius: 50%; background: var(--pink); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; box-shadow: 0 4px 12px rgba(255,45,120,0.4); }
.col-headers { display: grid; grid-template-columns: 44px 64px 1fr 130px 200px 90px; padding: 10px 36px; background: var(--bg); border-bottom: 1px solid var(--border); color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.track-row { display: grid; grid-template-columns: 44px 64px 1fr 130px 200px 90px; padding: 12px 36px; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.15s; }
.track-row:hover { background: rgba(255,45,120,0.05); }
.track-num { color: var(--muted); font-size: 13px; text-align: center; }
.track-thumb { width: 44px; height: 44px; border-radius: 8px; background: var(--surface2); display: flex; align-items: center; justify-content: center; font-size: 20px; border: 1px solid var(--border); }
.track-info { padding-left: 14px; }
.track-title { color: white; font-size: 13px; font-weight: 600; margin: 0; }
.track-artist { color: var(--muted); font-size: 11px; margin: 3px 0 0 0; }
.track-genre { color: var(--purple); font-size: 11px; text-align: center; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.track-album { color: var(--muted); font-size: 12px; text-align: center; }
.track-actions { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.track-actions .heart { color: var(--muted); font-size: 14px; }
.track-actions .duration { color: var(--muted); font-size: 12px; }
.bottom-fade { height: 80px; background: linear-gradient(to bottom, var(--bg), rgba(13,26,74,0.3)); }
.spotify-wrap { padding: 0 36px 4px 36px; }
.about-hero { padding: 64px 52px 48px 52px; background: linear-gradient(135deg, rgba(255,45,120,0.04) 0%, rgba(168,85,247,0.04) 100%); border-bottom: 1px solid var(--border); text-align: center; }
.about-hero h1 { font-family: 'Syne', sans-serif; font-size: 48px; font-weight: 800; background: linear-gradient(90deg, #ff2d78, #a855f7, #06b6d4); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 16px; letter-spacing: -0.03em; }
.about-hero p { color: #94a3b8; font-size: 16px; line-height: 1.7; max-width: 640px; margin: 0 auto; }
.about-section { padding: 48px 52px; }
.about-section h2 { color: white; font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800; margin-bottom: 24px; letter-spacing: -0.02em; }
.about-section p { color: #94a3b8; font-size: 14px; line-height: 1.8; margin-bottom: 16px; }
.team-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; margin-top: 24px; }
.team-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px 20px; text-align: center; transition: transform 0.2s, border-color 0.2s; }
.team-card:hover { transform: translateY(-4px); border-color: rgba(168,85,247,0.4); }
.team-avatar { width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 14px auto; background: linear-gradient(135deg, #ff2d78, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 800; color: white; font-family: 'Syne', sans-serif; }
.team-name { color: white; font-size: 14px; font-weight: 700; margin-bottom: 4px; }
.team-role { color: var(--muted); font-size: 12px; }
.how-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 24px; }
.how-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px 24px; }
.how-step { width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(255,45,120,0.2), rgba(168,85,247,0.2)); border: 1px solid rgba(255,45,120,0.3); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 16px; color: var(--pink); margin-bottom: 16px; }
.how-card h3 { color: white; font-size: 15px; font-weight: 700; margin-bottom: 10px; }
.how-card p { color: var(--muted); font-size: 13px; line-height: 1.6; }
.tech-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.tech-pill { background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.25); color: #c084fc; font-size: 12px; font-weight: 600; padding: 6px 16px; border-radius: 20px; }
.about-divider { height: 1px; background: var(--border); margin: 0 52px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.button("AtmoSound", key="logo_btn"):
        st.session_state.page = "home"; st.rerun()
    st.markdown('<span class="sidebar-section-label">Menu</span>', unsafe_allow_html=True)
    if st.button("🏠  Home", key="nav_home"):
        st.session_state.page = "home"; st.rerun()
    if st.button("✦  About Us", key="nav_about"):
        st.session_state.page = "about"; st.rerun()
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-section-label">Library</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🕐  Recently Generated</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">▶️  Most Played</span>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-section-label">Playlist & Favorites</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">♡  Your Favorites</span>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-section-label">General</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">⚙️  Settings</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🚪  Logout</span>', unsafe_allow_html=True)

if st.session_state.page == "home":
    st.markdown("""
    <div class="topnav">
        <a href="#">About Us</a><a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    img_path = "Gym pic.webp"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/webp;base64,{img_b64}" />'
    else:
        img_tag = '<div style="width:100%;height:380px;background:linear-gradient(135deg,#0d0d1f,#1a0a2e);display:flex;align-items:center;justify-content:center;font-size:64px;">🎵</div>'

    st.markdown(f"""
    <div class="hero-container">
        {img_tag}
        <div class="hero-overlay"></div>
        <div class="hero-badge">✦ AI-Powered Music</div>
        <div class="hero-text">The <span class="pink">Perfect Playlist</span><br>with one click</div>
    </div>
    <div class="input-section">
        <div class="input-label">Paste your venue's <span class="pink">Google Maps link</span> to get started</div>
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
        background: linear-gradient(135deg, #ff2d78, #e8004d) !important; color: white !important;
        border: none !important; border-radius: 12px !important; font-size: 14px !important;
        font-weight: 700 !important; padding: 14px !important; position: relative !important;
        height: auto !important; box-shadow: 0 4px 20px rgba(255,45,120,0.35) !important;
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
            st.session_state.page = "statistics"
            st.rerun()

    st.markdown("""
    <div class="cards-section">
        <div class="feat-card">
            <span class="feat-icon">🗺️</span>
            <h3>Zero Setup</h3>
            <p>Just paste a Google Maps link. Our model reads venue data automatically — no manual input needed.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🧠</span>
            <h3>ML-Powered</h3>
            <p>Ridge Regression and Neural Networks predict the ideal audio profile from 279 venue features.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🎧</span>
            <h3>Venue-Adaptive</h3>
            <p>A cafe, a bar, and a gym each get completely different playlists based on their unique atmosphere.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🎵</span>
            <h3>Real Spotify Tracks</h3>
            <p>20 songs sampled from 91,000+ Spotify tracks matched to your predicted genre and audio profile.</p>
        </div>
    </div>""", unsafe_allow_html=True)

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
        <a href="#">About Us</a><a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="venue-header">
        <h1>{venue_name}</h1>
        <div class="subtitle">{subtitle}{f" &nbsp;·&nbsp; ⭐ {rating}" if rating else ""}</div>
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

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        vibe_tags = get_vibe_tags(venue_info, nearest_genres)
        vibe_pills = "".join([f'<div class="vibe-pill">{t}</div>' for t in vibe_tags])
        sentiment_rows = get_sentiment_rows(audio_profile)
        sentiment_html = ""
        for label, pct, color in sentiment_rows:
            sentiment_html += f'<div class="sentiment-row"><span class="sentiment-label">{label}</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:{pct}%;background:{color};"></div></div><span class="sentiment-pct">{pct}%</span></div>'
        genre_pills = "".join([
            f'<span style="background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.25);color:#c084fc;font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;margin-right:6px;letter-spacing:0.5px;text-transform:uppercase;">{g["genre"]}</span>'
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
            <div style="text-align:center;padding:4px 0;">{genre_pills}</div>
        </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div style="background:var(--surface);border:1px solid var(--border);border-radius:16px 16px 0 0;padding:18px 20px 10px 20px;"><div class="box-title" style="margin-bottom:0;">Busyness by Hour</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            for label, val in {"7am": 20, "9am": 75, "11am": 60, "12pm": 89, "2pm": 65, "4pm": 42}.items():
                st.slider(label, 0, 100, val, format="%d%%", key=f"busy_{label}")
        acoustic_rows = get_acoustic_rows(audio_profile)
        html = '<div class="box"><div class="box-title">Acoustic Targets</div>'
        for lbl, dv, pct in acoustic_rows:
            html += f'<div class="acoustic-row"><span class="acoustic-label">{lbl}</span><div class="acoustic-bar-bg"><div class="acoustic-bar-fill" style="width:{int(pct*100)}%;"></div></div><span class="acoustic-val">{dv:.2f}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("GO TO PLAYLIST  →", key="go_playlist", use_container_width=True):
        st.session_state.page = "playlist"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background: linear-gradient(135deg, #0891b2, #0e7490) !important; color: white !important;
        font-size: 16px !important; font-weight: 700 !important; border-radius: 14px !important;
        padding: 18px !important; letter-spacing: 2px !important; border: none !important;
        position: relative !important; height: auto !important; box-shadow: 0 4px 20px rgba(6,182,212,0.25) !important;
    }
    </style>""", unsafe_allow_html=True)

elif st.session_state.page == "playlist":
    result = st.session_state.result or DEMO_RESULT
    playlist_df = result["playlist"]
    venue_name = st.session_state.venue_name

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
        background-color: transparent !important; color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.08) !important; font-size: 14px !important;
        font-weight: 600 !important; padding: 8px 16px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    div[data-testid="stMainBlockContainer"] .stButton > button:hover {
        background-color: rgba(255,45,120,0.1) !important; color: var(--pink) !important;
    }
    </style>""", unsafe_allow_html=True)

    n_songs = len(playlist_df)
    total_dur = total_duration_str(playlist_df) if not playlist_df.empty else "—"
    top_artists = get_top_artists(playlist_df) if not playlist_df.empty else "—"

    st.markdown(f"""
    <div class="playlist-banner">
        <div class="album-art">
            <div class="album-art-label">ATMO<br>SOUND</div>
            <div style="font-size:30px;margin-top:4px;">🎵</div>
        </div>
        <div class="banner-info">
            <p class="banner-title">The perfect mix for {venue_name}</p>
            <p class="banner-artists">{top_artists}</p>
            <div class="banner-meta">
                <span>🎵 {n_songs} songs</span><span>⏱ {total_dur}</span>
                <div class="play-all">Play All &nbsp;<div class="play-btn">&#9654;</div></div>
            </div>
        </div>
    </div>
    <div class="col-headers">
        <div>#</div><div></div>
        <div style="padding-left:14px;">Title</div>
        <div style="text-align:center;">Genre</div>
        <div style="text-align:center;">Album</div>
        <div style="text-align:right;">Time</div>
    </div>""", unsafe_allow_html=True)

    if playlist_df.empty:
        st.markdown('<p style="color:#64748b;text-align:center;padding:60px;">No tracks found.</p>', unsafe_allow_html=True)
    else:
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
            track_id = getattr(row, "track_id", None)

            st.markdown(f"""
            <div class="track-row">
                <div class="track-num">{i}</div>
                <div class="track-thumb">{emoji}</div>
                <div class="track-info"><p class="track-title">{title}</p><p class="track-artist">{artist}</p></div>
                <div class="track-genre">{genre_display}</div>
                <div class="track-album">{album_short}</div>
                <div class="track-actions"><span class="heart">♡</span><span class="duration">{dur_str}</span></div>
            </div>""", unsafe_allow_html=True)

            if track_id and str(track_id) != "nan":
                st.markdown(
                    f'<div class="spotify-wrap"><iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" '
                    f'width="100%" height="80" frameborder="0" allowtransparency="true" '
                    f'allow="encrypted-media" style="border-radius:10px;margin-bottom:2px;"></iframe></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="bottom-fade"></div>', unsafe_allow_html=True)

elif st.session_state.page == "about":
    st.markdown("""
    <div class="topnav">
        <a href="#">Contact</a><a href="#">How It Works</a>
        <button class="login-btn">Login</button><button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    col_back2, _ = st.columns([1, 11])
    with col_back2:
        if st.button("← Home", key="about_back"):
            st.session_state.page = "home"; st.rerun()
    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.08) !important; font-size: 14px !important;
        font-weight: 600 !important; padding: 8px 16px !important;
        position: relative !important; height: auto !important; border-radius: 8px !important;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
        <h1>About AtmoSound</h1>
        <p>AtmoSound is an ML-powered playlist generation system that creates venue-adaptive music recommendations using real Google Maps data, Ridge Regression, and Neural Networks trained on 4,484 Manhattan venues and 91,000+ Spotify tracks.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-section">
        <h2>🎓 The Team</h2>
        <p>Built by a team of 5 graduate students at Cornell Tech as a final project for the Applied Machine Learning course.</p>
        <div class="team-grid">
            <div class="team-card"><div class="team-avatar">B</div><div class="team-name">Bhoomika Mehta</div><div class="team-role">ML Pipeline & Data Engineering</div></div>
            <div class="team-card"><div class="team-avatar">D</div><div class="team-name">Devki Veerareddy</div><div class="team-role">Frontend & Integration</div></div>
            <div class="team-card"><div class="team-avatar">J</div><div class="team-name">Jason Sebastian</div><div class="team-role">Model Training & Evaluation</div></div>
            <div class="team-card"><div class="team-avatar">A</div><div class="team-name">Alaka Balaji Vembar</div><div class="team-role">Data Preprocessing</div></div>
            <div class="team-card"><div class="team-avatar">L</div><div class="team-name">Lily Ling</div><div class="team-role">UI/UX Design</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-section">
        <h2>⚙️ How It Works</h2>
        <div class="how-grid">
            <div class="how-card">
                <div class="how-step">1</div>
                <h3>Venue Lookup</h3>
                <p>Paste any Google Maps URL. We fetch live data from the Google Places API — rating, price level, type, neighbourhood, reviews, and 15+ boolean attributes like outdoor seating and live music.</p>
            </div>
            <div class="how-card">
                <div class="how-step">2</div>
                <h3>Audio Profile Prediction</h3>
                <p>Venue data is transformed into a 279-dimensional feature vector using TF-IDF, SVD, and one-hot encoding. Our Ridge Regression model predicts a 7-dimensional audio profile: energy, valence, danceability, acousticness, instrumentalness, liveness, and speechiness.</p>
            </div>
            <div class="how-card">
                <div class="how-step">3</div>
                <h3>Playlist Generation</h3>
                <p>We find the 5 nearest Spotify genre clusters using cosine distance, then sample 20 tracks weighted by popularity and proximity to the predicted audio profile. Songs are arranged in an energy arc — building to peak, then cooling down.</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="about-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-section">
        <h2>🧪 The ML Models</h2>
        <p>All models are implemented from scratch using NumPy — no scikit-learn, TensorFlow, or PyTorch.</p>
        <div class="how-grid">
            <div class="how-card">
                <div class="how-step">λ</div>
                <h3>Ridge Regression</h3>
                <p>Closed-form solution W* = (XᵀX + λI)⁻¹Xᵀy with L2 regularization. Optimal for our high-dimensional sparse feature space. Achieves MSE = 0.14, CosSim = 0.65.</p>
            </div>
            <div class="how-card">
                <div class="how-step">∿</div>
                <h3>Neural Network</h3>
                <p>Two hidden layers (256, 128) with ReLU activations, dropout = 0.2, trained via mini-batch SGD. Grid search over 54 configs. Achieves MSE = 0.015, CosSim = 0.96.</p>
            </div>
            <div class="how-card">
                <div class="how-step">K</div>
                <h3>K-Means Clustering</h3>
                <p>Groups 112 Spotify genre profiles into archetypes. Predicted audio vectors are matched to nearest genre centroids using Euclidean distance, weighted by proximity for playlist sampling.</p>
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

    st.markdown('<div class="about-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-section">
        <h2>📊 The Data</h2>
        <p><strong style="color:white;">4,484 Manhattan venue records</strong> collected via the Google Places API covering restaurants, cafes, bars, gyms, and more. Missing boolean attributes use tri-state encoding (1 = True, 0 = False, -1 = Unknown). Review text processed with TF-IDF + SVD compressed to 50 features.</p>
        <p>The Spotify dataset contains <strong style="color:white;">91,271 tracks</strong> across 112 genres. Pseudo-labels for audio targets were generated by mapping venue primary types to Spotify genre groups, modulated by price level and boolean attributes.</p>
    </div>""", unsafe_allow_html=True)
