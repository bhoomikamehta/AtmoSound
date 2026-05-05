# Streamlit Integration Guide

This document explains how to connect the AtmoSound ML pipeline to the Streamlit front end. Everything you need lives in the `atmosound/` package and `pipeline_artifacts/` folder.

---

## Folder Structure

Your Streamlit app directory should look like this:

```
your_app/
├── app.py                    ← your Streamlit entry point
├── atmosound/                ← the ML pipeline package (copy this folder in)
│   ├── __init__.py
│   ├── pipeline.py
│   ├── transformers.py
│   └── google_maps_utils.py
├── pipeline_artifacts/       ← model weights and data (copy this folder in)
│   ├── deployment_transformers.pkl
│   ├── ridge_model.pkl
│   ├── nn_model.pkl
│   ├── genre_profiles.csv
│   └── spotify_filtered.csv
└── requirements.txt
```

`requirements.txt` only needs:

```
numpy
pandas
streamlit
requests
```

No scikit-learn, no torch, nothing else. The ML code is all NumPy.

---

## Quick Start — Minimal Working Example

This is the simplest possible Streamlit app that takes venue data and returns a playlist. You can build out the full 3-page layout around this core.

```python
import streamlit as st
from atmosound import AtmoSoundPipeline

# Load once, cache so it doesn't reload on every interaction
@st.cache_resource
def load_pipeline():
    return AtmoSoundPipeline("pipeline_artifacts")

pipeline = load_pipeline()

# ── User Input ──
url = st.text_input("Paste a Google Maps link")

if st.button("Generate Playlist"):
    # For now, use a test venue dict — replace with real API call later
    venue_data = {
        "rating": 4.5,
        "price_level": "PRICE_LEVEL_MODERATE",
        "primary_type": "italian_restaurant",
        "neighbourhood": "West Village",
        "review_summary": "Great pasta and cozy atmosphere",
        "generative_summary": "A popular Italian spot in the Village",
        "serves_wine": True,
        "serves_dinner": True,
        "outdoor_seating": True,
    }

    result = pipeline.generate_playlist(venue_data, model="ridge", n_songs=25)

    st.write("Audio Profile:", result["audio_profile"])
    st.dataframe(result["playlist"])
```

---

## What `generate_playlist()` Returns

When you call `pipeline.generate_playlist(venue_data)`, you get back a dictionary with five keys. Here's what each one contains and which page it maps to.

### `result["venue_info"]` → Statistics Page

A dict with the parsed venue attributes:

```python
{
    "rating": 4.5,
    "price_level": 2,                    # ordinal 0-4 (0 = missing)
    "primary_type": "italian_restaurant",
    "neighbourhood": "West Village",
    "has_text": True,                     # whether reviews/summary existed
    "active_flags": ["serves_wine", "serves_dinner", "outdoor_seating"],
}
```

Use `active_flags` for the "Vibe Tags" section on the Statistics page. Display `rating` and `price_level` as metric cards.

### `result["audio_profile"]` → Statistics Page (Acoustic Targets)

A dict mapping each audio dimension to its predicted value (float 0–1):

```python
{
    "danceability": 0.5842,
    "energy": 0.4991,
    "acousticness": 0.4823,
    "valence": 0.4571,
    "instrumentalness": 0.0672,
    "liveness": 0.1649,
    "speechiness": 0.0621,
}
```

Use this for the Acoustic Targets panel and the radar chart. Every value is between 0 and 1.

### `result["radar_data"]` → Statistics Page (Radar Chart)

Pre-formatted for a radar chart:

```python
{
    "dimensions": ["danceability", "energy", ...],    # 7 labels
    "predicted": [0.584, 0.499, ...],                  # 7 floats
    "genre_means": {
        "jazz": [0.512, 0.394, ...],                   # comparison lines
        "soul": [0.571, 0.522, ...],
        ...
    },
}
```

You can plot the `predicted` line and overlay the genre means as reference. Here's a simple Streamlit radar chart using `st.plotly_chart`:

```python
import plotly.graph_objects as go

rd = result["radar_data"]
dims = rd["dimensions"]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=rd["predicted"], theta=dims, fill="toself", name="Predicted"))
for genre, profile in rd["genre_means"].items():
    fig.add_trace(go.Scatterpolar(r=profile, theta=dims, name=genre, opacity=0.4))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
st.plotly_chart(fig)
```

### `result["nearest_genres"]` → Statistics Page or Playlist Page

A list of the 5 closest genre matches:

```python
[
    {"genre": "jazz", "distance": 0.0312, "profile": [0.512, ...]},
    {"genre": "soul", "distance": 0.0487, "profile": [0.571, ...]},
    ...
]
```

Display as genre tags or badges. Closer distance = stronger match.

### `result["playlist"]` → Playlist Page

A pandas DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `track_name` | str | Song title |
| `artists` | str | Artist name(s) |
| `album_name` | str | Album |
| `genre` | str | Genre the song was sampled from |
| `popularity` | int | Spotify popularity score (0–100) |
| `duration_ms` | int | Duration in milliseconds |
| `danceability` | float | 0–1 |
| `energy` | float | 0–1 |
| `acousticness` | float | 0–1 |
| `valence` | float | 0–1 |
| `instrumentalness` | float | 0–1 |
| `liveness` | float | 0–1 |
| `speechiness` | float | 0–1 |

Songs are pre-ordered in an energy arc (builds up, peaks, cools down). Display them in the order returned — don't re-sort.

To show duration in a readable format:

```python
result["playlist"]["duration"] = result["playlist"]["duration_ms"].apply(
    lambda ms: f"{ms // 60000}:{(ms % 60000) // 1000:02d}"
)
```

---

## Connecting the Google Maps API

When the user pastes a Google Maps URL and clicks Generate, you need to:

1. Extract the Place ID from the URL
2. Call the Google Maps Places API
3. Parse the response into a `venue_data` dict
4. Pass it to the pipeline

```python
import requests
from atmosound import (
    AtmoSoundPipeline,
    parse_google_maps_response,
    extract_place_id_from_url,
)

GMAPS_API_KEY = st.secrets["GMAPS_API_KEY"]  # store in .streamlit/secrets.toml

def fetch_venue_data(url):
    """Fetch venue data from Google Maps given a URL."""
    place_id = extract_place_id_from_url(url)

    if place_id is None:
        # URL format doesn't contain a direct Place ID
        # Use the Text Search endpoint to resolve it
        search_resp = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GMAPS_API_KEY,
                "X-Goog-FieldMask": "places.id",
            },
            json={"textQuery": url},
        )
        places = search_resp.json().get("places", [])
        if not places:
            return None
        place_id = places[0]["id"]

    # Fetch full place details
    fields = ",".join([
        "rating", "priceLevel", "primaryType", "types",
        "addressComponents", "reviews", "generativeSummary",
        "editorialSummary", "goodForChildren", "goodForGroups",
        "goodForWatchingSports", "allowsDogs", "liveMusic",
        "outdoorSeating", "reservable", "servesBeer",
        "servesCocktails", "servesWine", "servesCoffee",
        "servesBreakfast", "servesBrunch", "servesDinner",
        "servesLunch", "servesVegetarianFood", "servesDessert",
        "menuForChildren",
    ])

    resp = requests.get(
        f"https://places.googleapis.com/v1/places/{place_id}",
        headers={
            "X-Goog-Api-Key": GMAPS_API_KEY,
            "X-Goog-FieldMask": fields,
        },
    )

    if resp.status_code != 200:
        return None

    return parse_google_maps_response(resp.json())
```

Then in your main app logic:

```python
if st.button("Generate Playlist"):
    with st.spinner("Analyzing venue..."):
        venue_data = fetch_venue_data(url)

    if venue_data is None:
        st.error("Couldn't find that venue. Check the URL and try again.")
    else:
        result = pipeline.generate_playlist(venue_data)
        st.session_state["result"] = result
        st.session_state["page"] = "statistics"
```

---

## Page Navigation with `st.session_state`

Since the app has three pages (Home → Statistics → Playlist), use session state to track which page to show:

```python
if "page" not in st.session_state:
    st.session_state["page"] = "home"

page = st.session_state["page"]

if page == "home":
    show_home_page()
elif page == "statistics":
    show_statistics_page(st.session_state["result"])
elif page == "playlist":
    show_playlist_page(st.session_state["result"])
```

The "GO TO PLAYLIST" button on the Statistics page just sets `st.session_state["page"] = "playlist"` and calls `st.rerun()`.

---

## Feature Importance (Optional — for Statistics Page)

If you want to show which features influenced the prediction:

```python
importance = pipeline.get_feature_importance(top_n=5)
# Returns: {"danceability": [{"feature": "ptype_italian_restaurant", "weight": 0.042}, ...], ...}
```

You could display the top features for 2-3 key dimensions as a small table or bar chart.

---

## Parameters You Can Expose in Settings

These have sensible defaults but could be user-adjustable:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `model` | `"ridge"` | `"ridge"` or `"nn"` — which model to use |
| `n_songs` | `25` | Playlist length (20–30 is reasonable) |
| `top_k_genres` | `5` | How many genre buckets to sample from |
| `seed` | `None` | Set an integer for reproducible playlists |

---

## Things That Can Go Wrong

**"Dimension mismatch" error** — The venue's `primary_type` or `neighbourhood` value isn't in the training vocabulary. The pipeline handles this gracefully (zeros out the one-hot slot), but if someone passes a completely made-up type string, predictions will lean toward the global average. Not a crash, just a bland playlist.

**Empty playlist** — Happens if the matched genres have zero songs in the filtered Spotify pool. Extremely unlikely with 91k tracks and 112 genres, but check `len(result["playlist"]) > 0` before rendering.

**Missing venue fields** — The pipeline handles every field being `None`. Missing booleans become "unknown" (-1), missing rating becomes median (4.5), missing text becomes empty. The more fields present, the better the prediction, but it won't break.
