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

for key, default in {
    "page": "home", "result": None,
    "venue_name": "Your Venue", "review_count": 0,
    "is_fav": False, "original_audio_profile": None,
    "busyness_adjusted": False,
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

def adjust_audio_for_busyness(base_profile, busyness_values):
    avg = sum(busyness_values) / len(busyness_values) / 100.0
    boost = (avg - 0.5) * 0.35
    adjusted = dict(base_profile)
    adjusted["energy"] = float(np.clip(base_profile.get("energy", 0.5) + boost, 0.05, 0.95))
    adjusted["danceability"] = float(np.clip(base_profile.get("danceability", 0.5) + boost * 0.8, 0.05, 0.95))
    adjusted["acousticness"] = float(np.clip(base_profile.get("acousticness", 0.5) - boost * 0.7, 0.05, 0.95))
    adjusted["valence"] = float(np.clip(base_profile.get("valence", 0.5) + boost * 0.5, 0.05, 0.95))
    return adjusted, avg

def regenerate_for_busyness(adjusted_profile):
    if pipeline is None:
        return DEMO_RESULT["playlist"], DEMO_RESULT["nearest_genres"]
    try:
        profile_array = np.array([
            adjusted_profile.get("danceability", 0.5),
            adjusted_profile.get("energy", 0.5),
            adjusted_profile.get("acousticness", 0.5),
            adjusted_profile.get("valence", 0.5),
            adjusted_profile.get("instrumentalness", 0.1),
            adjusted_profile.get("liveness", 0.15),
            adjusted_profile.get("speechiness", 0.05),
        ])
        genres = pipeline.get_nearest_genres(profile_array, k=5)
        pl = pipeline.sample_songs(genres, n_songs=20, seed=None)
        return pl, genres
    except Exception as e:
        st.warning(f"Could not adjust playlist: {e}")
        return None, None

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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');
:root {
    --pink:#ff2d78; --pink-dim:rgba(255,45,120,0.12); --pink-glow:rgba(255,45,120,0.25);
    --purple:#a855f7; --purple-dim:rgba(168,85,247,0.12);
    --bg:#06060e; --bg2:#09091a; --surface:#0d0d20; --surface2:#121228;
    --border:rgba(255,255,255,0.055); --border2:rgba(255,255,255,0.09);
    --text:#e8eaf0; --text2:#8892a4; --text3:#3d4558;
}
*{font-family:'Inter',sans-serif;box-sizing:border-box;margin:0;padding:0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
.stApp{background-color:var(--bg);}
[data-testid="collapsedControl"]{display:none!important;}
[data-testid="stSidebar"]{
    transform:none!important;min-width:260px!important;max-width:260px!important;
    width:260px!important;visibility:visible!important;display:block!important;
    position:relative!important;background:var(--bg2)!important;
    border-right:1px solid var(--border)!important;padding-top:0!important;
}
button[data-testid="baseButton-header"]{display:none!important;}
[data-testid="stSidebar"] button[kind="header"]{display:none!important;}
[data-testid="stSidebar"] span{color:var(--text)!important;}
[data-testid="stSidebar"] .stButton>button{
    background-color:transparent!important;color:var(--text2)!important;
    border:none!important;border-radius:10px!important;padding:10px 16px!important;
    font-size:14px!important;font-weight:500!important;width:100%!important;
    text-align:left!important;height:auto!important;position:relative!important;
    transition:all .18s ease!important;letter-spacing:.01em!important;
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,45,120,0.08)!important;color:white!important;
}
[data-testid="stSidebar"] .stButton:first-child>button{
    font-family:'Syne',sans-serif!important;font-size:22px!important;font-weight:800!important;
    background:linear-gradient(135deg,#ff2d78 0%,#c084fc 100%)!important;
    -webkit-background-clip:text!important;background-clip:text!important;
    -webkit-text-fill-color:transparent!important;
    padding:24px 20px 20px 20px!important;border-radius:0!important;
    letter-spacing:-0.03em!important;border-bottom:1px solid var(--border)!important;
    margin-bottom:4px!important;
}
[data-testid="stSidebar"] .stButton:first-child>button:hover{
    background:linear-gradient(135deg,#ff2d78,#c084fc)!important;
    -webkit-background-clip:text!important;background-clip:text!important;
    -webkit-text-fill-color:transparent!important;
}
.sb-section{padding:16px 16px 5px 16px;}
.sb-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2.5px;
    color:var(--text3)!important;display:block;margin-bottom:5px;}
.sb-divider{height:1px;background:var(--border);margin:6px 0;}
.sb-card{display:flex;align-items:center;gap:11px;padding:10px 12px;margin:3px 6px;
    border-radius:10px;background:var(--surface);border:1px solid var(--border);transition:all .15s;}
.sb-card:hover{border-color:rgba(255,45,120,0.25);background:rgba(255,45,120,0.06);}
.sb-card-icon{width:36px;height:36px;border-radius:8px;
    background:linear-gradient(135deg,var(--pink),var(--purple));
    display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;}
.sb-card-body{flex:1;min-width:0;}
.sb-card-name{color:var(--text)!important;font-size:13px;font-weight:600;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sb-card-meta{color:var(--text3)!important;font-size:11px;margin-top:2px;}
.sb-card-badge{background:var(--purple-dim);color:var(--purple)!important;
    font-size:9px;font-weight:700;padding:2px 7px;border-radius:8px;
    text-transform:uppercase;letter-spacing:.5px;flex-shrink:0;}
.sb-empty{color:var(--text3)!important;font-size:13px;padding:8px 12px;font-style:italic;}
.topnav{background:rgba(6,6,14,.96);backdrop-filter:blur(24px);padding:14px 40px;
    display:flex;justify-content:flex-end;align-items:center;gap:32px;
    border-bottom:1px solid var(--border);}
.topnav a{color:var(--text2);text-decoration:none;font-size:14px;font-weight:500;transition:color .15s;}
.topnav a:hover{color:white;}
.topnav .login-btn{border:1px solid var(--border2);color:var(--text2);padding:7px 20px;
    border-radius:20px;font-size:13.5px;background:transparent;cursor:pointer;
    font-family:'Inter',sans-serif;transition:all .15s;}
.topnav .login-btn:hover{border-color:rgba(255,255,255,.2);color:white;}
.topnav .signup-btn{background:linear-gradient(135deg,#ff2d78,#c4004d);color:white;
    padding:8px 22px;border-radius:20px;font-weight:600;font-size:13.5px;border:none;
    cursor:pointer;font-family:'Inter',sans-serif;box-shadow:0 2px 14px rgba(255,45,120,.3);}
.page-wrap{max-width:1160px;margin:0 auto;padding:0 40px;}
.hero-container{position:relative;width:100%;height:380px;overflow:hidden;}
.hero-container img{width:100%;height:100%;object-fit:cover;display:block;}
.hero-overlay{position:absolute;top:0;left:0;width:100%;height:100%;
    background:linear-gradient(115deg,rgba(6,6,14,.93) 0%,rgba(6,6,14,.52) 55%,rgba(160,80,240,.14) 100%);}
.hero-badge{position:absolute;top:28px;left:48px;background:var(--pink-dim);
    border:1px solid rgba(255,45,120,.28);color:#ff6b9d;font-size:11px;font-weight:700;
    padding:5px 14px;border-radius:20px;letter-spacing:1.8px;text-transform:uppercase;}
.hero-text{position:absolute;bottom:50px;left:48px;color:white;
    font-family:'Syne',sans-serif;font-size:46px;font-weight:800;
    line-height:1.12;letter-spacing:-.03em;}
.hero-text .grad{background:linear-gradient(90deg,#ff2d78,#c084fc);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.hero-sub{position:absolute;bottom:22px;left:48px;color:var(--text3);
    font-size:13.5px;font-weight:500;letter-spacing:.02em;}
.input-section{padding:36px 0 18px 0;}
.input-label{color:var(--text2);font-size:15px;font-weight:500;margin-bottom:14px;letter-spacing:.01em;}
.input-label .pink{color:var(--pink);font-weight:600;}
div[data-testid="stTextInput"] input{background-color:var(--surface)!important;color:white!important;
    border:1px solid var(--border2)!important;border-radius:12px!important;
    font-size:15px!important;padding:14px 20px!important;}
div[data-testid="stTextInput"] input:focus{border-color:rgba(255,45,120,.45)!important;
    box-shadow:0 0 0 3px rgba(255,45,120,.09)!important;}
.cards-section{padding:4px 0 52px 0;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;}
.feat-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
    padding:26px 22px;transition:transform .2s,border-color .2s;position:relative;overflow:hidden;}
.feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,45,120,.4),transparent);
    opacity:0;transition:opacity .2s;}
.feat-card:hover{transform:translateY(-3px);border-color:rgba(255,45,120,.18);}
.feat-card:hover::before{opacity:1;}
.feat-icon{font-size:26px;margin-bottom:16px;display:block;}
.feat-card h3{color:white;font-size:15px;font-weight:700;margin:0 0 9px 0;letter-spacing:-.01em;}
.feat-card p{color:var(--text3);font-size:13.5px;line-height:1.65;margin:0;}
.venue-header{text-align:center;padding:32px 0 24px 0;}
.venue-header h1{color:white;font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
    margin:0 0 8px 0;letter-spacing:-.03em;}
.venue-header .subtitle{color:var(--text2);font-size:14px;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
    padding:28px 20px;text-align:center;position:relative;overflow:hidden;}
.metric-card::after{content:'';position:absolute;bottom:0;left:15%;right:15%;height:1px;
    background:linear-gradient(90deg,transparent,var(--pink),transparent);}
.metric-value{background:linear-gradient(135deg,#ff2d78,#a855f7);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
    font-family:'Syne',sans-serif;font-size:42px;font-weight:800;margin:0;line-height:1;}
.metric-label{color:var(--text3);font-size:11px;font-weight:700;margin:10px 0 0 0;
    text-transform:uppercase;letter-spacing:2.5px;}
.box{background:var(--surface);border:1px solid var(--border);border-radius:16px;
    padding:22px;margin-bottom:16px;}
.box-title{color:var(--text2);font-size:11px;font-weight:700;text-align:center;
    margin-bottom:18px;text-transform:uppercase;letter-spacing:2.5px;}
.vibe-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.vibe-pill{background:rgba(255,45,120,.07);border:1px solid rgba(255,45,120,.16);
    color:var(--text)!important;font-size:13.5px;font-weight:600;border-radius:10px;
    padding:12px;text-align:center;}
.sentiment-row{display:flex;align-items:center;margin-bottom:13px;gap:12px;}
.sentiment-label{color:var(--text2);font-size:13px;font-weight:500;width:100px;flex-shrink:0;}
.sentiment-bar-bg{flex:1;background:rgba(255,255,255,.05);border-radius:4px;height:7px;overflow:hidden;}
.sentiment-bar-fill{height:100%;border-radius:4px;}
.sentiment-pct{color:var(--text);font-size:13px;font-weight:700;width:38px;text-align:right;flex-shrink:0;}
.legend{display:flex;gap:16px;margin-top:14px;font-size:12px;color:var(--text3);}
.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;}
.acoustic-row{display:flex;align-items:center;margin-bottom:11px;gap:12px;}
.acoustic-label{color:var(--text2);font-size:13px;font-weight:500;width:100px;flex-shrink:0;}
.acoustic-bar-bg{flex:1;background:rgba(255,255,255,.05);border-radius:4px;height:7px;overflow:hidden;}
.acoustic-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#ff2d78,#a855f7);}
.acoustic-val{color:var(--text2);font-size:13px;font-weight:600;width:38px;text-align:right;flex-shrink:0;}
div[data-testid="stSlider"]{padding-top:0!important;padding-bottom:0!important;
    margin-top:-5px!important;margin-bottom:-5px!important;}
div[data-testid="stSlider"] label,div[data-testid="stSlider"] p{color:var(--text2)!important;
    font-size:13px!important;font-family:'Inter',sans-serif!important;margin-bottom:0!important;}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"]{color:var(--pink)!important;
    font-weight:700;font-size:11px!important;}
.stSlider div[role="slider"]{background-color:var(--pink)!important;border-color:var(--pink)!important;}
[data-testid="stSlider"]>div>div>div>div{background:var(--pink)!important;}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface)!important;
    border:1px solid var(--border)!important;border-radius:0 0 16px 16px!important;
    padding:4px 14px 14px 14px!important;margin-bottom:16px!important;}
.playlist-banner{background:linear-gradient(135deg,#06061a 0%,#0b1040 55%,#060615 100%);
    padding:28px 36px;display:flex;align-items:center;gap:26px;
    border-bottom:1px solid var(--border);}
.album-art{width:116px;height:116px;border-radius:14px;
    background:linear-gradient(135deg,#ff2d78,#7c3aed);
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    flex-shrink:0;box-shadow:0 8px 32px rgba(255,45,120,.3),0 0 0 1px rgba(255,45,120,.2);}
.album-art-label{color:white;font-family:'Syne',sans-serif;font-size:12px;font-weight:800;
    text-align:center;line-height:1.2;}
.banner-title{color:white;font-family:'Syne',sans-serif;font-size:23px;font-weight:800;
    margin:0 0 5px 0;letter-spacing:-.025em;}
.banner-artists{color:var(--text3);font-size:13px;margin:0 0 13px 0;}
.banner-meta{display:flex;align-items:center;gap:18px;}
.banner-meta span{color:var(--text2);font-size:13px;}
.play-all{color:var(--pink);font-size:13px;font-weight:700;margin-left:auto;
    display:flex;align-items:center;gap:8px;cursor:pointer;}
.play-btn{width:32px;height:32px;border-radius:50%;background:var(--pink);
    display:flex;align-items:center;justify-content:center;color:white;font-size:12px;
    box-shadow:0 3px 12px rgba(255,45,120,.4);}
.track-wrap{padding:6px 36px 4px 36px;}
.track-label{display:flex;align-items:center;gap:10px;padding:8px 0 4px 0;}
.track-num{color:var(--text3);font-size:13px;font-weight:600;width:22px;text-align:right;flex-shrink:0;}
.track-genre-tag{background:var(--purple-dim);border:1px solid rgba(168,85,247,.22);
    color:var(--purple)!important;font-size:10px;font-weight:700;padding:3px 10px;
    border-radius:10px;text-transform:uppercase;letter-spacing:.6px;flex-shrink:0;}
.bottom-fade{height:60px;background:linear-gradient(to bottom,var(--bg),rgba(10,15,50,.15));}
div[data-testid="stMainBlockContainer"] .stButton>button{
    background:linear-gradient(135deg,#ff2d78,#c4004d)!important;color:white!important;
    border:none!important;border-radius:12px!important;font-size:14px!important;
    font-weight:600!important;padding:13px!important;position:relative!important;
    height:auto!important;box-shadow:0 3px 16px rgba(255,45,120,.3)!important;
    transition:box-shadow .15s!important;letter-spacing:.01em!important;}
div[data-testid="stMainBlockContainer"] .stButton>button:hover{
    box-shadow:0 5px 22px rgba(255,45,120,.5)!important;}
.about-hero{padding:60px 0 44px 0;text-align:center;border-bottom:1px solid var(--border);}
.about-hero h1{font-family:'Syne',sans-serif;font-size:48px;font-weight:800;
    background:linear-gradient(90deg,#ff2d78,#a855f7,#06b6d4);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:16px;letter-spacing:-.035em;}
.about-hero p{color:var(--text2);font-size:16px;line-height:1.75;max-width:580px;margin:0 auto;}
.about-section{padding:44px 0;border-bottom:1px solid var(--border);}
.about-section:last-child{border-bottom:none;}
.about-section h2{color:white;font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
    margin-bottom:20px;letter-spacing:-.025em;}
.about-section p{color:var(--text2);font-size:14.5px;line-height:1.8;margin-bottom:14px;}
.how-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:20px;}
.how-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:26px 22px;}
.how-step{width:34px;height:34px;border-radius:8px;background:var(--pink-dim);
    border:1px solid rgba(255,45,120,.25);display:flex;align-items:center;justify-content:center;
    font-family:'Syne',sans-serif;font-weight:800;font-size:15px;color:var(--pink);margin-bottom:15px;}
.how-card h3{color:white;font-size:15px;font-weight:700;margin-bottom:10px;}
.how-card p{color:var(--text3);font-size:13.5px;line-height:1.65;margin:0;}
.tech-pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;}
.tech-pill{background:var(--purple-dim);border:1px solid rgba(168,85,247,.2);
    color:#b57bff!important;font-size:12px;font-weight:600;padding:5px 15px;border-radius:20px;}
.lib-header{padding:36px 0 22px 0;border-bottom:1px solid var(--border);}
.lib-header h1{color:white;font-family:'Syne',sans-serif;font-size:30px;font-weight:800;
    margin:0;letter-spacing:-.025em;}
.lib-header p{color:var(--text2);font-size:14px;margin-top:5px;}
.lib-grid{padding:26px 0;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}
.lib-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
    padding:22px;transition:transform .15s,border-color .15s;}
.lib-card:hover{transform:translateY(-2px);border-color:rgba(255,45,120,.25);}
.lib-card-icon{width:46px;height:46px;border-radius:11px;
    background:linear-gradient(135deg,#ff2d78,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:14px;}
.lib-card-name{color:white;font-size:15px;font-weight:700;margin-bottom:4px;}
.lib-card-meta{color:var(--text2);font-size:13px;}
.lib-card-genres{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;}
.lib-card-genre{background:var(--purple-dim);border:1px solid rgba(168,85,247,.18);
    color:#b57bff!important;font-size:10.5px;font-weight:700;padding:3px 10px;
    border-radius:10px;text-transform:uppercase;letter-spacing:.5px;}
.lib-empty{padding:70px 0;text-align:center;}
.lib-empty-icon{font-size:52px;margin-bottom:18px;}
.lib-empty-text{color:var(--text3);font-size:16px;font-weight:600;margin-bottom:8px;}
.lib-empty-sub{color:var(--text3);font-size:13.5px;opacity:.6;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

user_data = load_user_data()
recent = user_data.get("recent", [])
favorites = user_data.get("favorites", [])

with st.sidebar:
    if st.button("⬡ AtmoSound", key="logo_btn"):
        st.session_state.page = "home"; st.rerun()
    st.markdown('<div class="sb-section"><span class="sb-label">Menu</span></div>', unsafe_allow_html=True)
    if st.button("🏠  Home", key="nav_home"):
        st.session_state.page = "home"; st.rerun()
    if st.button("✦  About", key="nav_about"):
        st.session_state.page = "about"; st.rerun()
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section"><span class="sb-label">Recently Generated</span></div>', unsafe_allow_html=True)
    if recent:
        for r in recent[:3]:
            g = ", ".join(r.get("genres", [])[:2]).upper()
            st.markdown(f"""<div class="sb-card"><div class="sb-card-icon">🎧</div><div class="sb-card-body"><div class="sb-card-name">{r["venue"]}</div><div class="sb-card-meta">{r["timestamp"]}</div></div><div class="sb-card-badge">{g[:10]}</div></div>""", unsafe_allow_html=True)
        if st.button("See all recent →", key="nav_recent"):
            st.session_state.page = "recent"; st.rerun()
    else:
        st.markdown('<div class="sb-empty">No playlists yet</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section"><span class="sb-label">Favorites</span></div>', unsafe_allow_html=True)
    if favorites:
        for f in favorites[:3]:
            g = ", ".join(f.get("genres", [])[:2]).upper()
            st.markdown(f"""<div class="sb-card"><div class="sb-card-icon">🎵</div><div class="sb-card-body"><div class="sb-card-name">{f["venue"]}</div><div class="sb-card-meta">Saved {f["timestamp"]}</div></div><div class="sb-card-badge">♥</div></div>""", unsafe_allow_html=True)
        if st.button("See all favorites →", key="nav_favs"):
            st.session_state.page = "favorites"; st.rerun()
    else:
        st.markdown('<div class="sb-empty">No favorites yet</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section"><span class="sb-label">General</span></div>', unsafe_allow_html=True)
    st.markdown('<span style="display:block;padding:9px 16px;color:#3d4558;font-size:14px;">⚙️  Settings</span>', unsafe_allow_html=True)

if st.session_state.page == "home":
    st.markdown("""<div class="topnav"><a href="#">About</a><a href="#">Contact</a><a href="#">How It Works</a><button class="login-btn">Login</button><button class="signup-btn">Sign Up</button></div>""", unsafe_allow_html=True)
    img_path = "Gym pic.webp"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/webp;base64,{img_b64}" />'
    else:
        img_tag = '<div style="width:100%;height:380px;background:linear-gradient(135deg,#06060e,#14082e);display:flex;align-items:center;justify-content:center;font-size:72px;">🎵</div>'
    st.markdown(f"""<div class="hero-container">{img_tag}<div class="hero-overlay"></div><div class="hero-badge">✦ ML-Powered</div><div class="hero-text">The <span class="grad">Perfect Playlist</span><br>for any venue</div><div class="hero-sub">Ridge Regression · 91,000+ Spotify tracks · Google Places API</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="input-section"><div class="input-label">Paste your venue\'s <span class="pink">Google Maps link</span> to generate a playlist</div></div>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input("url", placeholder="https://www.google.com/maps/place/...", label_visibility="collapsed", key="url_field")
    with col_btn:
        go = st.button("Generate  ✦", key="gen_btn", use_container_width=True)
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
            st.session_state.original_audio_profile = dict(result.get("audio_profile", {}))
            st.session_state.busyness_adjusted = False
            genres = [g["genre"] for g in result.get("nearest_genres", [])[:3]]
            n_songs = len(result.get("playlist", pd.DataFrame()))
            add_recent(venue_name or "Your Venue", genres, n_songs)
            st.session_state.page = "statistics"
            st.rerun()
    st.markdown("""<div class="cards-section"><div class="feat-card"><span class="feat-icon">🗺️</span><h3>Zero Setup</h3><p>Paste any Google Maps link. We pull live venue data automatically — no manual input needed.</p></div><div class="feat-card"><span class="feat-icon">🧠</span><h3>ML-Powered</h3><p>Ridge Regression predicts the ideal audio profile from 279 real venue features.</p></div><div class="feat-card"><span class="feat-icon">🎧</span><h3>Venue-Adaptive</h3><p>A cafe, bar, and gym each get different playlists based on their unique vibe and DNA.</p></div><div class="feat-card"><span class="feat-icon">🎵</span><h3>Spotify-Ready</h3><p>20 real tracks from 91,000+ songs. Preview and play directly in the app.</p></div></div></div>""", unsafe_allow_html=True)

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
    st.markdown("""<div class="topnav"><a href="#">About</a><a href="#">Contact</a><a href="#">How It Works</a><button class="login-btn">Login</button><button class="signup-btn">Sign Up</button></div>""", unsafe_allow_html=True)
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(f"""<div class="venue-header"><h1>{venue_name}</h1><div class="subtitle">{subtitle}{f" · ⭐ {rating}" if rating else ""}</div></div>""", unsafe_allow_html=True)
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
        genre_pills = "".join([f'<span style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.22);color:#b57bff;font-size:11px;font-weight:700;padding:4px 12px;border-radius:12px;margin-right:6px;text-transform:uppercase;letter-spacing:0.6px;">{g["genre"]}</span>' for g in nearest_genres[:3]])
        st.markdown(f"""<div class="box"><div class="box-title">Vibe Tags</div><div class="vibe-grid">{vibe_pills}</div></div><div class="box"><div class="box-title">Audio Profile</div>{sentiment_html}<div class="legend"><span><span class="legend-dot" style="background:#00c97a;"></span>Mood</span><span><span class="legend-dot" style="background:#ff2d78;"></span>Intensity</span><span><span class="legend-dot" style="background:#38bdf8;"></span>Texture</span></div></div><div class="box"><div class="box-title">Predicted Genres</div><div style="text-align:center;padding:3px 0;">{genre_pills}</div></div>""", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div style="background:var(--surface);border:1px solid var(--border);border-top-left-radius:16px;border-top-right-radius:16px;padding:18px 20px 10px 20px;"><div class="box-title" style="margin-bottom:0;">Busyness by Hour</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            busy_7am  = st.slider("7am",  0, 100, 20, format="%d%%", key="busy_7am")
            busy_9am  = st.slider("9am",  0, 100, 75, format="%d%%", key="busy_9am")
            busy_11am = st.slider("11am", 0, 100, 60, format="%d%%", key="busy_11am")
            busy_12pm = st.slider("12pm", 0, 100, 89, format="%d%%", key="busy_12pm")
            busy_2pm  = st.slider("2pm",  0, 100, 65, format="%d%%", key="busy_2pm")
            busy_4pm  = st.slider("4pm",  0, 100, 42, format="%d%%", key="busy_4pm")
        if st.button("⟳  Apply Busyness to Playlist", key="apply_busy", use_container_width=True):
            base = st.session_state.original_audio_profile or audio_profile
            busy_vals = [busy_7am, busy_9am, busy_11am, busy_12pm, busy_2pm, busy_4pm]
            adjusted, avg = adjust_audio_for_busyness(base, busy_vals)
            new_playlist, new_genres = regenerate_for_busyness(adjusted)
            if new_playlist is not None and not new_playlist.empty:
                st.session_state.result["playlist"] = new_playlist
                st.session_state.result["nearest_genres"] = new_genres
                st.session_state.result["audio_profile"] = adjusted
                st.session_state.busyness_adjusted = True
                st.success(f"✓ Playlist updated for {int(avg*100)}% avg busyness")
                st.rerun()
        acoustic_rows = get_acoustic_rows(audio_profile)
        html = '<div class="box"><div class="box-title">Acoustic Targets</div>'
        for lbl, dv, pct in acoustic_rows:
            html += f'<div class="acoustic-row"><span class="acoustic-label">{lbl}</span><div class="acoustic-bar-bg"><div class="acoustic-bar-fill" style="width:{int(pct*100)}%;"></div></div><span class="acoustic-val">{dv:.2f}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.session_state.busyness_adjusted:
        st.markdown('<div style="text-align:center;margin-bottom:10px;"><span style="background:rgba(0,201,122,0.1);border:1px solid rgba(0,201,122,0.25);color:#00c97a;font-size:12px;font-weight:600;padding:5px 16px;border-radius:20px;">✓ Playlist tuned to current busyness</span></div>', unsafe_allow_html=True)
    if st.button("GO TO PLAYLIST  →", key="go_playlist", use_container_width=True):
        st.session_state.page = "playlist"; st.rerun()
    st.markdown("""<style>div[data-testid="stMainBlockContainer"] .stButton>button{background:linear-gradient(135deg,#0e7490,#0c5f75)!important;box-shadow:0 3px 16px rgba(6,182,212,.2)!important;font-size:15px!important;font-weight:700!important;border-radius:13px!important;padding:17px!important;letter-spacing:1.5px!important;}</style>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "playlist":
    result = st.session_state.result or DEMO_RESULT
    playlist_df = result["playlist"]
    venue_name = st.session_state.venue_name
    nearest_genres = result.get("nearest_genres", [])
    genres = [g["genre"] for g in nearest_genres[:3]]
    st.markdown("""<style>div[data-testid="stMainBlockContainer"] .stButton>button{background:transparent!important;color:#8892a4!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:none!important;font-size:13.5px!important;font-weight:600!important;padding:8px 16px!important;border-radius:9px!important;}div[data-testid="stMainBlockContainer"] .stButton>button:hover{background:rgba(255,45,120,.08)!important;color:#ff2d78!important;border-color:rgba(255,45,120,.25)!important;box-shadow:none!important;}</style>""", unsafe_allow_html=True)
    col_back, _, col_home = st.columns([1, 10, 1])
    with col_back:
        if st.button("← Back", key="back_btn"):
            st.session_state.page = "statistics"; st.rerun()
    with col_home:
        if st.button("🏠", key="playlist_home"):
            st.session_state.page = "home"; st.rerun()
    n_songs = len(playlist_df)
    total_dur = total_duration_str(playlist_df) if not playlist_df.empty else "—"
    top_artists = get_top_artists(playlist_df) if not playlist_df.empty else "—"
    is_fav = is_favorite(venue_name)
    fav_label = "♥  Saved" if is_fav else "♡  Save Playlist"
    st.markdown(f"""<div class="playlist-banner"><div class="album-art"><div class="album-art-label">ATMO<br>SOUND</div><div style="font-size:28px;margin-top:5px;">🎵</div></div><div style="flex:1;"><p class="banner-title">The perfect mix for {venue_name}</p><p class="banner-artists">{top_artists}</p><div class="banner-meta"><span>🎵 {n_songs} songs</span><span>⏱ {total_dur}</span><div class="play-all">Play All &nbsp;<div class="play-btn">&#9654;</div></div></div></div></div>""", unsafe_allow_html=True)
    col_fav, _ = st.columns([1, 5])
    with col_fav:
        if st.button(fav_label, key="fav_btn"):
            saved = toggle_favorite(venue_name, genres, n_songs)
            st.session_state.is_fav = saved
            st.rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if playlist_df.empty:
        st.markdown('<p style="color:#3d4558;text-align:center;padding:60px;font-size:15px;">No tracks found.</p>', unsafe_allow_html=True)
    else:
        for i, row in enumerate(playlist_df.itertuples(), start=1):
            genre = getattr(row, "genre", "—")
            genre_display = genre.replace("-", " ").replace("_", " ").title() if genre else "—"
            track_id = getattr(row, "track_id", None)
            st.markdown(f"""<div class="track-wrap"><div class="track-label"><span class="track-num">{i}</span><span class="track-genre-tag">{genre_display}</span></div></div>""", unsafe_allow_html=True)
            if track_id and str(track_id) != "nan":
                st.markdown(f'<div style="padding:0 36px 12px 36px;"><iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media" style="border-radius:10px;display:block;"></iframe></div>', unsafe_allow_html=True)
            else:
                title = getattr(row, "track_name", "Unknown")
                artist = getattr(row, "artists", "Unknown")
                st.markdown(f'<div style="padding:4px 36px 12px 36px;color:#3d4558;font-size:13px;">{title} — {artist}</div>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-fade"></div>', unsafe_allow_html=True)

elif st.session_state.page == "about":
    st.markdown("""<div class="topnav"><a href="#">Contact</a><a href="#">How It Works</a><button class="login-btn">Login</button><button class="signup-btn">Sign Up</button></div>""", unsafe_allow_html=True)
    st.markdown("""<style>div[data-testid="stMainBlockContainer"] .stButton>button{background:transparent!important;color:#8892a4!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:none!important;font-size:13.5px!important;font-weight:600!important;padding:8px 16px!important;border-radius:9px!important;}div[data-testid="stMainBlockContainer"] .stButton>button:hover{background:rgba(255,45,120,.08)!important;color:#ff2d78!important;border-color:rgba(255,45,120,.25)!important;box-shadow:none!important;}</style>""", unsafe_allow_html=True)
    col_b, _ = st.columns([1, 11])
    with col_b:
        if st.button("← Home", key="about_back"):
            st.session_state.page = "home"; st.rerun()
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown("""<div class="about-hero"><h1>About AtmoSound</h1><p>An ML-powered playlist generation system that creates venue-adaptive music recommendations using real Google Maps data, Ridge Regression, and Neural Networks — trained on 4,484 Manhattan venues and 91,000+ Spotify tracks.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-section"><h2>⚙️ How It Works</h2><div class="how-grid"><div class="how-card"><div class="how-step">1</div><h3>Venue Lookup</h3><p>Paste any Google Maps URL. We fetch live data — rating, price, type, neighbourhood, reviews, and 15+ attributes like outdoor seating and live music.</p></div><div class="how-card"><div class="how-step">2</div><h3>Audio Profile Prediction</h3><p>Venue data becomes a 279-dimensional feature vector via TF-IDF, SVD, and one-hot encoding. Ridge Regression predicts 7 audio dimensions: energy, valence, danceability, acousticness, instrumentalness, liveness, speechiness.</p></div><div class="how-card"><div class="how-step">3</div><h3>Playlist Generation</h3><p>We find the 5 nearest Spotify genre clusters using cosine distance, then sample 20 tracks weighted by popularity and audio proximity. Songs are arranged in an energy arc.</p></div></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-section"><h2>🧪 The ML Models</h2><p>All models are implemented from scratch in NumPy — no scikit-learn, TensorFlow, or PyTorch.</p><div class="how-grid"><div class="how-card"><div class="how-step">λ</div><h3>Ridge Regression</h3><p>Closed-form W* = (XᵀX + λI)⁻¹Xᵀy with L2 regularization. Optimal for high-dimensional sparse feature spaces. MSE = 0.14, CosSim = 0.65.</p></div><div class="how-card"><div class="how-step">∿</div><h3>Neural Network</h3><p>Two hidden layers (256, 128) with ReLU activations, dropout = 0.2, mini-batch SGD. Grid search over 54 configurations. MSE = 0.015, CosSim = 0.96.</p></div><div class="how-card"><div class="how-step">K</div><h3>K-Means Clustering</h3><p>Groups 112 Spotify genre profiles into archetypes. Predicted audio vectors are matched to nearest centroids using Euclidean distance for playlist sampling.</p></div></div><div class="tech-pills"><span class="tech-pill">NumPy</span><span class="tech-pill">Pandas</span><span class="tech-pill">Streamlit</span><span class="tech-pill">Google Places API</span><span class="tech-pill">Spotify Dataset</span><span class="tech-pill">TF-IDF</span><span class="tech-pill">SVD</span><span class="tech-pill">Ridge Regression</span><span class="tech-pill">Neural Networks</span><span class="tech-pill">K-Means</span></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="about-section"><h2>📊 The Data</h2><p><strong style="color:white;">4,484 Manhattan venue records</strong> collected via the Google Places API. Missing boolean attributes use tri-state encoding. Review text processed with TF-IDF + SVD to 50 features.</p><p>The Spotify dataset contains <strong style="color:white;">91,271 tracks</strong> across 112 genres. Pseudo-labels were generated by mapping venue types to genre groups, modulated by price level and boolean attributes.</p></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "favorites":
    st.markdown("""<style>div[data-testid="stMainBlockContainer"] .stButton>button{background:transparent!important;color:#8892a4!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:none!important;font-size:13.5px!important;font-weight:600!important;padding:8px 16px!important;border-radius:9px!important;}div[data-testid="stMainBlockContainer"] .stButton>button:hover{background:rgba(255,45,120,.08)!important;color:#ff2d78!important;border-color:rgba(255,45,120,.25)!important;box-shadow:none!important;}</style>""", unsafe_allow_html=True)
    col_b, _ = st.columns([1, 11])
    with col_b:
        if st.button("← Back", key="fav_back"):
            st.session_state.page = "home"; st.rerun()
    data = load_user_data()
    favs = data.get("favorites", [])
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
st.markdown(f"""<div class="lib-header"><h1>♥ Your Favorites</h1><p>{len(favs)} saved playlist{"s" if len(favs) != 1 else ""}</p></div>""", unsafe_allow_html=True)
if not favs:
st.markdown("""<div class="lib-empty"><div class="lib-empty-icon">♡</div><div class="lib-empty-text">No favorites yet</div><div class="lib-empty-sub">Generate a playlist and tap "Save Playlist" to save it here</div></div>""", unsafe_allow_html=True)
else:
st.markdown('<div class="lib-grid">', unsafe_allow_html=True)
for f in favs:
genres_html = "".join([f'<span class="lib-card-genre">{g}</span>' for g in f.get("genres", [])[:3]])
st.markdown(f"""<div class="lib-card"><div class="lib-card-icon">🎵</div><div class="lib-card-name">{f["venue"]}</div><div class="lib-card-meta">Saved {f["timestamp"]} · {f.get("songs","—")} songs</div><div class="lib-card-genres">{genres_html}</div></div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.page == "recent":
st.markdown("""<style>div[data-testid="stMainBlockContainer"] .stButton>button{background:transparent!important;color:#8892a4!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:none!important;font-size:13.5px!important;font-weight:600!important;padding:8px 16px!important;border-radius:9px!important;}div[data-testid="stMainBlockContainer"] .stButton>button:hover{background:rgba(255,45,120,.08)!important;color:#ff2d78!important;border-color:rgba(255,45,120,.25)!important;box-shadow:none!important;}</style>""", unsafe_allow_html=True)
col_b, _ = st.columns([1, 11])
with col_b:
if st.button("← Back", key="rec_back"):
st.session_state.page = "home"; st.rerun()
data = load_user_data()
recent_all = data.get("recent", [])
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
st.markdown(f"""<div class="lib-header"><h1>🕐 Recently Generated</h1><p>{len(recent_all)} playlist{"s" if len(recent_all) != 1 else ""} generated</p></div>""", unsafe_allow_html=True)
if not recent_all:
st.markdown("""<div class="lib-empty"><div class="lib-empty-icon">🕐</div><div class="lib-empty-text">No playlists generated yet</div><div class="lib-empty-sub">Paste a Google Maps link on the home page to get started</div></div>""", unsafe_allow_html=True)
else:
st.markdown('<div class="lib-grid">', unsafe_allow_html=True)
for r in recent_all:
genres_html = "".join([f'<span class="lib-card-genre">{g}</span>' for g in r.get("genres", [])[:3]])
st.markdown(f"""<div class="lib-card"><div class="lib-card-icon">🎧</div><div class="lib-card-name">{r["venue"]}</div><div class="lib-card-meta">{r["timestamp"]} · {r.get("songs","—")} songs</div><div class="lib-card-genres">{genres_html}</div></div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
