"""
AtmoSound — Playlist Generation Pipeline
=========================================
Standalone module that takes raw Google Maps venue data and produces
a curated playlist of ~20-30 tracks.

Usage:
    from playlist_generator import AtmoSoundPipeline
    pipeline = AtmoSoundPipeline("pipeline_artifacts")
    result = pipeline.generate_playlist(venue_data)

Dependencies: NumPy, Pandas only (no ML libraries).
"""

import numpy as np
import pandas as pd
import pickle
import re
import os
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Reconstructed transformers (same classes from preprocessing)
# ─────────────────────────────────────────────────────────────

class TfidfVectorizerFromScratch:
    """TF-IDF vectorizer using only NumPy and Python stdlib."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
                 min_df=2, max_df_ratio=0.95):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocabulary_ = None
        self.idf_ = None

    def _tokenize(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1]

    def _get_ngrams(self, tokens):
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i + n]))
        return ngrams

    def transform(self, documents):
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        tfidf_matrix = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            ngrams = self._get_ngrams(self._tokenize(doc))
            if len(ngrams) == 0:
                continue
            term_counts = Counter(ngrams)
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = count / len(ngrams)
                    if self.sublinear_tf:
                        tf = 1 + np.log(tf) if tf > 0 else 0
                    tfidf_matrix[i, idx] = tf * self.idf_[idx]

        row_norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        tfidf_matrix = tfidf_matrix / row_norms
        return tfidf_matrix


class TruncatedSVDFromScratch:
    """Truncated SVD — transform only (components loaded from artifacts)."""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


class MinMaxScalerFromScratch:
    """Min-max scaler — transform only (min/range loaded from artifacts)."""

    def __init__(self):
        self.min_ = None
        self.range_ = None

    def transform(self, X):
        return (X - self.min_) / self.range_


class NeuralNetwork:
    """Two-layer neural network — inference only (weights loaded from artifacts)."""

    def __init__(self, weights, use_batch_norm=False):
        for attr, val in weights.items():
            setattr(self, attr, val)
        self.use_batch_norm = use_batch_norm

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _sigmoid(z):
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        if self.use_batch_norm and hasattr(self, "running_mean1"):
            z1 = self.gamma1 * ((z1 - self.running_mean1) /
                                np.sqrt(self.running_var1 + 1e-8)) + self.beta1
        a1 = self._relu(z1)

        z2 = a1 @ self.W2 + self.b2
        if self.use_batch_norm and hasattr(self, "running_mean2"):
            z2 = self.gamma2 * ((z2 - self.running_mean2) /
                                np.sqrt(self.running_var2 + 1e-8)) + self.beta2
        a2 = self._relu(z2)

        z3 = a2 @ self.W3 + self.b3
        return self._sigmoid(z3)


# ─────────────────────────────────────────────────────────────
# Price level mapping
# ─────────────────────────────────────────────────────────────

PRICE_MAP = {
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    "Inexpensive": 1,
    "Moderate": 2,
    "Expensive": 3,
    "Very Expensive": 4,
    1: 1, 2: 2, 3: 3, 4: 4,
}


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

class AtmoSoundPipeline:
    """End-to-end pipeline: venue data → predicted audio profile → playlist.

    Parameters
    ----------
    artifacts_dir : str
        Path to the pipeline_artifacts/ directory containing:
        - deployment_transformers.pkl (TF-IDF, SVD, scaler, feature schema)
        - ridge_model.pkl (Ridge regression weights)
        - nn_model.pkl (Neural network weights)  [optional]
        - genre_profiles.csv (per-genre mean audio profiles)
        - spotify_filtered.csv (song pool for sampling)
    """

    def __init__(self, artifacts_dir="pipeline_artifacts"):
        self.artifacts_dir = artifacts_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all serialized pipeline components."""
        d = self.artifacts_dir

        # ── Deployment transformers ──
        with open(os.path.join(d, "deployment_transformers.pkl"), "rb") as f:
            deploy = pickle.load(f)

        # Reconstruct TF-IDF
        self.tfidf = TfidfVectorizerFromScratch(**deploy["tfidf_config"])
        self.tfidf.vocabulary_ = deploy["tfidf_vocabulary"]
        self.tfidf.idf_ = deploy["tfidf_idf"]

        # Reconstruct SVD
        self.svd = TruncatedSVDFromScratch(n_components=deploy["svd_components"].shape[0])
        self.svd.components_ = deploy["svd_components"]
        self.svd.mean_ = deploy["svd_mean"]

        # Reconstruct scaler
        self.scaler = MinMaxScalerFromScratch()
        self.scaler.min_ = deploy["scaler_min"]
        self.scaler.range_ = deploy["scaler_range"]

        # Schema
        self.feature_names = deploy["feature_names"]
        self.bool_cols = deploy["bool_cols"]
        self.audio_features = deploy["audio_features"]
        self.rating_median = deploy["rating_median"]

        # Extract neighbourhood and primary type lists from feature names
        self.neighbourhood_list = [f.replace("neigh_", "") for f in self.feature_names
                                   if f.startswith("neigh_")]
        self.primary_type_list = [f.replace("ptype_", "") for f in self.feature_names
                                  if f.startswith("ptype_")]

        # ── Ridge model ──
        with open(os.path.join(d, "ridge_model.pkl"), "rb") as f:
            ridge_data = pickle.load(f)
        self.ridge_W = ridge_data["W"]  # shape (d+1, 7), includes bias

        # ── Neural network model (optional) ──
        nn_path = os.path.join(d, "nn_model.pkl")
        if os.path.exists(nn_path):
            with open(nn_path, "rb") as f:
                nn_data = pickle.load(f)
            self.nn_model = NeuralNetwork(
                nn_data["weights"],
                use_batch_norm=nn_data["config"].get("use_batch_norm", False)
            )
        else:
            self.nn_model = None

        # ── Genre profiles ──
        self.genre_profiles = pd.read_csv(os.path.join(d, "genre_profiles.csv"), index_col=0)
        self.genre_centroids = self.genre_profiles.values
        self.genre_names = list(self.genre_profiles.index)

        # ── Spotify song pool ──
        self.songs = pd.read_csv(os.path.join(d, "spotify_filtered.csv"))

        print(f"Pipeline loaded: {len(self.feature_names)} features, "
              f"{len(self.genre_names)} genres, {len(self.songs):,} songs")

    # ─────────────────────────────────────────────────────────
    # Feature Vector Construction
    # ─────────────────────────────────────────────────────────

    def build_feature_vector(self, venue_data):
        """Convert raw Google Maps API venue data into a model-ready feature vector.

        Parameters
        ----------
        venue_data : dict
            Dictionary with keys matching Google Maps Places API fields:
            - "rating" : float (e.g. 4.5)
            - "price_level" : str or int (e.g. "PRICE_LEVEL_MODERATE" or 2)
            - "primary_type" : str (e.g. "italian_restaurant")
            - "neighbourhood" : str (e.g. "West Village")
            - "review_summary" : str (aggregated review text)
            - "generative_summary" : str (Google AI summary)
            - Boolean flags: "good_for_children", "live_music", etc. (True/False/None)

        Returns
        -------
        X : np.ndarray, shape (1, d)
            Scaled feature vector ready for model input.
        venue_info : dict
            Parsed venue attributes for display on the Statistics page.
        """
        # ── 1. Rating ──
        rating = venue_data.get("rating")
        if rating is None or (isinstance(rating, float) and np.isnan(rating)):
            rating = self.rating_median
        rating = float(rating)

        # ── 2. Price level ──
        raw_price = venue_data.get("price_level")
        price_encoded = PRICE_MAP.get(raw_price, 0)
        price_missing = 1 if raw_price is None else 0

        # ── 3. Boolean attributes ──
        bool_values = []
        bool_info = {}
        for col in self.bool_cols:
            val = venue_data.get(col)
            if val is True or val == 1:
                bool_values.append(1)
                bool_info[col] = True
            elif val is False or val == 0:
                bool_values.append(0)
                bool_info[col] = False
            else:
                bool_values.append(-1)  # unknown
                bool_info[col] = None

        # ── 4. Neighbourhood one-hot ──
        neighbourhood = venue_data.get("neighbourhood", "")
        neigh_vector = np.zeros(len(self.neighbourhood_list))
        if neighbourhood in self.neighbourhood_list:
            idx = self.neighbourhood_list.index(neighbourhood)
            neigh_vector[idx] = 1

        # ── 5. Primary type one-hot ──
        primary_type = venue_data.get("primary_type", "")
        ptype_vector = np.zeros(len(self.primary_type_list))
        if primary_type in self.primary_type_list:
            idx = self.primary_type_list.index(primary_type)
            ptype_vector[idx] = 1

        # ── 6. Text TF-IDF → SVD ──
        review = venue_data.get("review_summary", "") or ""
        gen_summary = venue_data.get("generative_summary", "") or ""
        combined_text = (review + " " + gen_summary).strip()

        tfidf_vec = self.tfidf.transform([combined_text])  # (1, 5000)
        text_svd = self.svd.transform(tfidf_vec)            # (1, 50)

        # ── 7. Assemble raw feature vector ──
        # Must match exact order from preprocessing:
        # [rating, price_encoded, price_missing, 18 bools, 33 neighs, 175 ptypes, 50 text_svd]
        raw_vector = np.concatenate([
            [rating],
            [price_encoded, price_missing],
            bool_values,
            neigh_vector,
            ptype_vector,
            text_svd.flatten()
        ]).reshape(1, -1)

        assert raw_vector.shape[1] == len(self.feature_names), (
            f"Feature dimension mismatch: got {raw_vector.shape[1]}, "
            f"expected {len(self.feature_names)}"
        )

        # ── 8. Min-max scale ──
        X_scaled = self.scaler.transform(raw_vector)

        # ── Venue info for Statistics page ──
        venue_info = {
            "rating": rating,
            "price_level": price_encoded,
            "primary_type": primary_type,
            "neighbourhood": neighbourhood,
            "has_text": len(combined_text) > 0,
            "bool_flags": bool_info,
            "active_flags": [k for k, v in bool_info.items() if v is True],
        }

        return X_scaled, venue_info

    # ─────────────────────────────────────────────────────────
    # Model Inference
    # ─────────────────────────────────────────────────────────

    def predict_audio_profile(self, X, model="ridge"):
        """Predict 7-dimensional audio profile for a venue.

        Parameters
        ----------
        X : np.ndarray, shape (1, d)
            Scaled feature vector from build_feature_vector().
        model : str
            "ridge" or "nn"

        Returns
        -------
        profile : np.ndarray, shape (7,)
            Predicted audio profile: [danceability, energy, acousticness,
            valence, instrumentalness, liveness, speechiness]
        """
        if model == "nn" and self.nn_model is not None:
            pred = self.nn_model.predict(X)
        else:
            X_b = np.hstack([X, np.ones((X.shape[0], 1))])
            pred = X_b @ self.ridge_W

        # Clip to valid range
        pred = np.clip(pred, 0, 1)
        return pred.flatten()

    # ─────────────────────────────────────────────────────────
    # Genre Retrieval
    # ─────────────────────────────────────────────────────────

    def get_nearest_genres(self, audio_profile, k=5):
        """Find the K nearest genre centroids to the predicted audio profile.

        Parameters
        ----------
        audio_profile : np.ndarray, shape (7,)
        k : int

        Returns
        -------
        genres : list of dict
            Each dict has keys: "genre", "distance", "profile"
        """
        dists = np.linalg.norm(self.genre_centroids - audio_profile, axis=1)
        top_k = np.argsort(dists)[:k]

        results = []
        for idx in top_k:
            results.append({
                "genre": self.genre_names[idx],
                "distance": float(dists[idx]),
                "profile": self.genre_centroids[idx].tolist(),
            })
        return results

    # ─────────────────────────────────────────────────────────
    # Song Sampling
    # ─────────────────────────────────────────────────────────

    def sample_songs(self, nearest_genres, n_songs=25, seed=None):
        """Sample songs from the nearest genres, weighted by popularity.

        Parameters
        ----------
        nearest_genres : list of dict
            Output from get_nearest_genres().
        n_songs : int
            Total number of songs to include in the playlist.
        seed : int or None
            Random seed for reproducibility. None for random.

        Returns
        -------
        playlist : pd.DataFrame
            Columns: track_name, artists, album_name, track_genre,
            popularity, duration_ms, danceability, energy, acousticness,
            valence, instrumentalness, liveness, speechiness
        """
        rng = np.random.RandomState(seed)

        # Weight genres inversely by distance (closer = more songs)
        distances = np.array([g["distance"] for g in nearest_genres])
        # Convert distances to weights (inverse, softmax-like)
        if distances.max() > 0:
            inv_dists = 1.0 / (distances + 1e-6)
            weights = inv_dists / inv_dists.sum()
        else:
            weights = np.ones(len(nearest_genres)) / len(nearest_genres)

        # Allocate songs per genre proportionally
        songs_per_genre = np.round(weights * n_songs).astype(int)
        # Ensure we hit the target total
        diff = n_songs - songs_per_genre.sum()
        if diff > 0:
            songs_per_genre[0] += diff
        elif diff < 0:
            songs_per_genre[-1] += diff
        songs_per_genre = np.maximum(songs_per_genre, 1)  # at least 1 per genre

        sampled_tracks = []

        for i, genre_info in enumerate(nearest_genres):
            genre_name = genre_info["genre"]
            n_from_genre = int(songs_per_genre[i])

            # Filter song pool to this genre
            genre_songs = self.songs[
                self.songs["track_genre_clean"] == genre_name
            ].copy()

            if len(genre_songs) == 0:
                # Fallback: try original genre column
                genre_songs = self.songs[
                    self.songs["track_genre"] == genre_name
                ].copy()

            if len(genre_songs) == 0:
                continue

            # Sample weighted by popularity
            pop = genre_songs["popularity"].values.astype(float)
            pop = pop - pop.min() + 1  # shift to positive
            probs = pop / pop.sum()

            n_sample = min(n_from_genre, len(genre_songs))
            chosen_idx = rng.choice(len(genre_songs), size=n_sample, replace=False, p=probs)
            sampled_tracks.append(genre_songs.iloc[chosen_idx])

        if not sampled_tracks:
            return pd.DataFrame()

        playlist = pd.concat(sampled_tracks, ignore_index=True)

        # Trim to exact target if we overshot
        if len(playlist) > n_songs:
            playlist = playlist.iloc[:n_songs]

        # Select and order output columns
        output_cols = [
            "track_name", "artists", "album_name", "track_genre_clean",
            "popularity", "duration_ms",
            "danceability", "energy", "acousticness", "valence",
            "instrumentalness", "liveness", "speechiness",
        ]
        available_cols = [c for c in output_cols if c in playlist.columns]
        playlist = playlist[available_cols].copy()
        playlist = playlist.rename(columns={"track_genre_clean": "genre"})

        # Reorder for listening flow: sort by energy (gentle start, build up, cool down)
        n = len(playlist)
        if n > 4:
            playlist = playlist.sort_values("energy").reset_index(drop=True)
            # Arc: start medium, build to peak, then cool down
            mid = n // 2
            ascending_half = playlist.iloc[:mid]
            descending_half = playlist.iloc[mid:][::-1]
            playlist = pd.concat([ascending_half, descending_half], ignore_index=True)

        return playlist

    # ─────────────────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────────────────

    def generate_playlist(self, venue_data, model="ridge", n_songs=25,
                          top_k_genres=5, seed=None):
        """End-to-end: venue data → playlist.

        Parameters
        ----------
        venue_data : dict
            Raw venue data from Google Maps API (see build_feature_vector).
        model : str
            "ridge" or "nn"
        n_songs : int
            Number of songs in the playlist (default: 25).
        top_k_genres : int
            Number of nearest genres to sample from (default: 5).
        seed : int or None
            Random seed for song sampling.

        Returns
        -------
        result : dict
            - "playlist" : pd.DataFrame with track details
            - "audio_profile" : dict with 7 predicted audio dimensions
            - "nearest_genres" : list of matched genres with distances
            - "venue_info" : parsed venue attributes for Statistics page
            - "radar_data" : dict formatted for radar chart rendering
        """
        # Step 1: Build feature vector
        X, venue_info = self.build_feature_vector(venue_data)

        # Step 2: Predict audio profile
        profile = self.predict_audio_profile(X, model=model)

        # Step 3: Find nearest genres
        genres = self.get_nearest_genres(profile, k=top_k_genres)

        # Step 4: Sample songs
        playlist = self.sample_songs(genres, n_songs=n_songs, seed=seed)

        # Step 5: Package results
        audio_profile_dict = dict(zip(self.audio_features, profile.tolist()))

        # Radar chart data (for Streamlit)
        radar_data = {
            "dimensions": self.audio_features,
            "predicted": profile.tolist(),
            "genre_means": {g["genre"]: g["profile"] for g in genres},
        }

        return {
            "playlist": playlist,
            "audio_profile": audio_profile_dict,
            "nearest_genres": genres,
            "venue_info": venue_info,
            "radar_data": radar_data,
        }

    # ─────────────────────────────────────────────────────────
    # Feature Importance (Ridge only)
    # ─────────────────────────────────────────────────────────

    def get_feature_importance(self, top_n=10):
        """Return top-N most important features per audio dimension.

        For explaining predictions on the Statistics page.
        """
        W = self.ridge_W[:-1, :]  # exclude bias row
        importance = {}
        for j, feat_name in enumerate(self.audio_features):
            abs_w = np.abs(W[:, j])
            top_idx = np.argsort(abs_w)[-top_n:][::-1]
            importance[feat_name] = [
                {"feature": self.feature_names[i], "weight": float(W[i, j])}
                for i in top_idx
            ]
        return importance


# ─────────────────────────────────────────────────────────────
# Google Maps API Helper
# ─────────────────────────────────────────────────────────────

def parse_google_maps_response(api_response):
    """Parse a Google Maps Places API (new) response into venue_data dict.

    Parameters
    ----------
    api_response : dict
        Raw JSON response from Google Maps Places API.
        Supports both the new Places API (v1) and the legacy format.

    Returns
    -------
    venue_data : dict
        Dictionary formatted for AtmoSoundPipeline.build_feature_vector().
    """
    venue = {}

    # ── Rating ──
    venue["rating"] = api_response.get("rating")

    # ── Price level ──
    venue["price_level"] = api_response.get("priceLevel",
                                             api_response.get("price_level"))

    # ── Primary type ──
    # New API uses "primaryType", legacy uses "types[0]"
    venue["primary_type"] = api_response.get("primaryType",
                                              api_response.get("primary_type"))
    if venue["primary_type"] is None:
        types = api_response.get("types", [])
        if types:
            venue["primary_type"] = types[0]

    # ── Neighbourhood ──
    # Extract from addressComponents or formatted_address
    venue["neighbourhood"] = ""
    address_components = api_response.get("addressComponents",
                                           api_response.get("address_components", []))
    for comp in address_components:
        comp_types = comp.get("types", [])
        if "neighborhood" in comp_types or "sublocality" in comp_types:
            venue["neighbourhood"] = comp.get("longText",
                                               comp.get("long_name", ""))
            break

    # ── Text summaries ──
    # Review summary from reviews
    reviews = api_response.get("reviews", [])
    if reviews:
        review_texts = [r.get("text", {}).get("text", r.get("text", ""))
                        for r in reviews if r]
        venue["review_summary"] = " ".join([t for t in review_texts if isinstance(t, str)])
    else:
        venue["review_summary"] = ""

    venue["generative_summary"] = api_response.get("generativeSummary", {}).get(
        "overview", {}).get("text", api_response.get("editorial_summary", {}).get(
            "overview", ""))

    # ── Boolean attributes ──
    # Maps API field names → our column names
    BOOL_FIELD_MAP = {
        "goodForChildren":        "good_for_children",
        "goodForGroups":          "good_for_groups",
        "goodForWatchingSports":  "good_for_watching_sports",
        "allowsDogs":             "allows_dogs",
        "liveMusic":              "live_music",
        "outdoorSeating":         "outdoor_seating",
        "reservable":             "reservable",
        "servesBeer":             "serves_beer",
        "servesCocktails":        "serves_cocktails",
        "servesWine":             "serves_wine",
        "servesCoffee":           "serves_coffee",
        "servesBreakfast":        "serves_breakfast",
        "servesBrunch":           "serves_brunch",
        "servesDinner":           "serves_dinner",
        "servesLunch":            "serves_lunch",
        "servesVegetarianFood":   "serves_vegetarian_food",
        "servesDessert":          "serves_dessert",
        "menuForChildren":        "menu_for_children",
    }

    # Also support snake_case keys (legacy format)
    BOOL_SNAKE_MAP = {
        "good_for_children":       "good_for_children",
        "good_for_groups":         "good_for_groups",
        "good_for_watching_sports":"good_for_watching_sports",
        "allows_dogs":             "allows_dogs",
        "live_music":              "live_music",
        "outdoor_seating":         "outdoor_seating",
        "reservable":              "reservable",
        "serves_beer":             "serves_beer",
        "serves_cocktails":        "serves_cocktails",
        "serves_wine":             "serves_wine",
        "serves_coffee":           "serves_coffee",
        "serves_breakfast":        "serves_breakfast",
        "serves_brunch":           "serves_brunch",
        "serves_dinner":           "serves_dinner",
        "serves_lunch":            "serves_lunch",
        "serves_vegetarian_food":  "serves_vegetarian_food",
        "serves_dessert":          "serves_dessert",
        "menu_for_children":       "menu_for_children",
    }

    for api_key, our_key in {**BOOL_FIELD_MAP, **BOOL_SNAKE_MAP}.items():
        val = api_response.get(api_key)
        if val is not None:
            venue[our_key] = val

    return venue


# ─────────────────────────────────────────────────────────────
# Convenience: extract Place ID from Google Maps URL
# ─────────────────────────────────────────────────────────────

def extract_place_id_from_url(url):
    """Extract the Google Maps Place ID from various URL formats.

    Supports:
    - https://www.google.com/maps/place/...
    - https://maps.google.com/?cid=...
    - Direct place IDs (ChIJ...)

    Returns
    -------
    place_id : str or None
    """
    import re

    if not url:
        return None

    # Direct Place ID
    if url.startswith("ChIJ") or url.startswith("Eh"):
        return url.strip()

    # URL with place_id parameter
    match = re.search(r'place_id[=:]([A-Za-z0-9_-]+)', url)
    if match:
        return match.group(1)

    # URL with data parameter containing place id
    match = re.search(r'!1s(0x[0-9a-f]+:0x[0-9a-f]+)', url)
    if match:
        return match.group(1)

    # URL with /place/ segment — need API call to resolve
    # Return None to signal that API resolution is needed
    return None


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("AtmoSound Pipeline — Self Test")
    print("=" * 70)

    pipeline = AtmoSoundPipeline("pipeline_artifacts")

    # Test venue: an Italian restaurant in West Village
    test_venue = {
        "rating": 4.6,
        "price_level": "PRICE_LEVEL_EXPENSIVE",
        "primary_type": "italian_restaurant",
        "neighbourhood": "West Village",
        "review_summary": "Amazing pasta and wine selection. Cozy intimate atmosphere "
                          "with dim lighting. Perfect for date night. The homemade "
                          "ravioli is incredible. Service is attentive and warm.",
        "generative_summary": "An upscale Italian restaurant known for handmade pasta "
                              "and an extensive wine list in a romantic setting.",
        "serves_wine": True,
        "serves_cocktails": True,
        "serves_dinner": True,
        "outdoor_seating": True,
        "reservable": True,
        "good_for_groups": True,
        "live_music": False,
    }

    print("\n── Test Venue ──")
    print(f"  Type: {test_venue['primary_type']}")
    print(f"  Neighbourhood: {test_venue['neighbourhood']}")
    print(f"  Rating: {test_venue['rating']}")

    result = pipeline.generate_playlist(test_venue, model="ridge", n_songs=25, seed=42)

    print("\n── Predicted Audio Profile ──")
    for feat, val in result["audio_profile"].items():
        print(f"  {feat:20s}: {val:.4f}")

    print("\n── Nearest Genres ──")
    for g in result["nearest_genres"]:
        print(f"  {g['genre']:20s}  (distance: {g['distance']:.4f})")

    print(f"\n── Playlist ({len(result['playlist'])} tracks) ──")
    for _, row in result["playlist"].head(10).iterrows():
        dur_min = row.get("duration_ms", 0) / 60000
        print(f"  {row['track_name'][:40]:40s} | {row['artists'][:25]:25s} | "
              f"{row['genre']:15s} | {dur_min:.1f}m")
    if len(result["playlist"]) > 10:
        print(f"  ... and {len(result['playlist']) - 10} more tracks")

    # Test with NN model
    print("\n── Neural Network Prediction ──")
    result_nn = pipeline.generate_playlist(test_venue, model="nn", n_songs=25, seed=42)
    for feat, val in result_nn["audio_profile"].items():
        print(f"  {feat:20s}: {val:.4f}")

    # Test with a sports bar
    print("\n" + "=" * 70)
    print("── Test 2: Sports Bar ──")
    test_bar = {
        "rating": 4.2,
        "price_level": "PRICE_LEVEL_MODERATE",
        "primary_type": "sports_bar",
        "neighbourhood": "Midtown West",
        "review_summary": "Great place to watch the game. Loud and fun atmosphere. "
                          "Good beer selection and wings. Big screens everywhere.",
        "generative_summary": "A lively sports bar with multiple TVs and classic pub fare.",
        "serves_beer": True,
        "good_for_watching_sports": True,
        "good_for_groups": True,
        "live_music": False,
    }

    result_bar = pipeline.generate_playlist(test_bar, model="ridge", n_songs=20, seed=42)
    print(f"  Profile: {result_bar['audio_profile']}")
    print(f"  Top genres: {[g['genre'] for g in result_bar['nearest_genres']]}")
    print(f"  Playlist: {len(result_bar['playlist'])} tracks")

    # Test with a yoga studio
    print("\n── Test 3: Yoga Studio ──")
    test_yoga = {
        "rating": 4.8,
        "primary_type": "yoga_studio",
        "neighbourhood": "SoHo",
        "review_summary": "Peaceful and calming space. Excellent instructors. "
                          "Beautiful studio with natural light.",
        "generative_summary": "A serene yoga studio offering various styles of practice.",
    }

    result_yoga = pipeline.generate_playlist(test_yoga, model="ridge", n_songs=20, seed=42)
    print(f"  Profile: {result_yoga['audio_profile']}")
    print(f"  Top genres: {[g['genre'] for g in result_yoga['nearest_genres']]}")

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)
