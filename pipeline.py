"""
AtmoSound — Playlist Generation Pipeline
=========================================
Loads trained model artifacts and runs: venue data → audio profile → playlist.

Usage:
    from pipeline import AtmoSoundPipeline
    pipeline = AtmoSoundPipeline("pipeline_artifacts")
    result = pipeline.generate_playlist(venue_data)
"""

import numpy as np
import pandas as pd
import pickle
import os
from transformers import TfidfTransformer, SVDTransformer, MinMaxTransformer, NeuralNetwork


# ── Price level mapping (Google Maps API → ordinal integer) ──
PRICE_MAP = {
    "PRICE_LEVEL_INEXPENSIVE": 1, "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3, "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    "Inexpensive": 1, "Moderate": 2, "Expensive": 3, "Very Expensive": 4,
    1: 1, 2: 2, 3: 3, 4: 4,
}


class AtmoSoundPipeline:
    def __init__(self, artifacts_dir="pipeline_artifacts"):
        self._load_artifacts(artifacts_dir)

    def _load_artifacts(self, d):
        with open(os.path.join(d, "deployment_transformers.pkl"), "rb") as f:
            deploy = pickle.load(f)

        self.tfidf = TfidfTransformer(
            deploy["tfidf_vocabulary"], deploy["tfidf_idf"], deploy["tfidf_config"])
        self.svd = SVDTransformer(deploy["svd_components"], deploy["svd_mean"])
        self.scaler = MinMaxTransformer(deploy["scaler_min"], deploy["scaler_range"])

        self.feature_names = deploy["feature_names"]
        self.bool_cols = deploy["bool_cols"]
        self.audio_features = deploy["audio_features"]
        self.rating_median = deploy["rating_median"]

        self.neighbourhood_list = [
            f.replace("neigh_", "") for f in self.feature_names if f.startswith("neigh_")]
        self.primary_type_list = [
            f.replace("ptype_", "") for f in self.feature_names if f.startswith("ptype_")]

        with open(os.path.join(d, "ridge_model.pkl"), "rb") as f:
            self.ridge_W = pickle.load(f)["W"]

        nn_path = os.path.join(d, "nn_model.pkl")
        if os.path.exists(nn_path):
            with open(nn_path, "rb") as f:
                nn_data = pickle.load(f)
            self.nn_model = NeuralNetwork(
                nn_data["weights"],
                use_batch_norm=nn_data["config"].get("use_batch_norm", False))
        else:
            self.nn_model = None

        self.genre_profiles = pd.read_csv(os.path.join(d, "genre_profiles.csv"), index_col=0)
        self.genre_centroids = self.genre_profiles.values
        self.genre_names = list(self.genre_profiles.index)
        self.songs = pd.read_csv(os.path.join(d, "spotify_filtered.csv"))

        print(f"Pipeline loaded: {len(self.feature_names)} features, "
              f"{len(self.genre_names)} genres, {len(self.songs):,} songs")

    def build_feature_vector(self, venue_data):
        rating = venue_data.get("rating")
        if rating is None or (isinstance(rating, float) and np.isnan(rating)):
            rating = self.rating_median
        rating = float(rating)

        raw_price = venue_data.get("price_level")
        price_encoded = PRICE_MAP.get(raw_price, 0)
        price_missing = 1 if raw_price is None else 0

        bool_values, bool_info = [], {}
        for col in self.bool_cols:
            val = venue_data.get(col)
            if val is True or val == 1:
                bool_values.append(1); bool_info[col] = True
            elif val is False or val == 0:
                bool_values.append(0); bool_info[col] = False
            else:
                bool_values.append(-1); bool_info[col] = None

        neigh = venue_data.get("neighbourhood", "")
        neigh_vec = np.zeros(len(self.neighbourhood_list))
        if neigh in self.neighbourhood_list:
            neigh_vec[self.neighbourhood_list.index(neigh)] = 1

        ptype = venue_data.get("primary_type", "")
        ptype_vec = np.zeros(len(self.primary_type_list))
        if ptype in self.primary_type_list:
            ptype_vec[self.primary_type_list.index(ptype)] = 1

        review = venue_data.get("review_summary", "") or ""
        gen_summary = venue_data.get("generative_summary", "") or ""
        combined = (review + " " + gen_summary).strip()
        text_svd = self.svd.transform(self.tfidf.transform([combined]))

        raw = np.concatenate([
            [rating], [price_encoded, price_missing],
            bool_values, neigh_vec, ptype_vec, text_svd.flatten()
        ]).reshape(1, -1)

        assert raw.shape[1] == len(self.feature_names), (
            f"Dimension mismatch: {raw.shape[1]} vs {len(self.feature_names)}")

        venue_info = {
            "rating": rating, "price_level": price_encoded,
            "primary_type": ptype, "neighbourhood": neigh,
            "has_text": len(combined) > 0,
            "active_flags": [k for k, v in bool_info.items() if v is True],
        }
        return self.scaler.transform(raw), venue_info

    def predict_audio_profile(self, X, model="ridge"):
        if model == "nn" and self.nn_model is not None:
            pred = self.nn_model.predict(X)
        else:
            X_b = np.hstack([X, np.ones((X.shape[0], 1))])
            pred = X_b @ self.ridge_W
        return np.clip(pred, 0, 1).flatten()

    def get_nearest_genres(self, profile, k=5):
        dists = np.linalg.norm(self.genre_centroids - profile, axis=1)
        top_k = np.argsort(dists)[:k]
        return [{"genre": self.genre_names[i],
                 "distance": float(dists[i]),
                 "profile": self.genre_centroids[i].tolist()} for i in top_k]

    def sample_songs(self, nearest_genres, n_songs=25, seed=None):
        rng = np.random.RandomState(seed)

        dists = np.array([g["distance"] for g in nearest_genres])
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
        counts = np.maximum(np.round(weights * n_songs).astype(int), 1)
        counts[0] += n_songs - counts.sum()

        sampled = []
        for genre_info, n in zip(nearest_genres, counts):
            pool = self.songs[self.songs["track_genre_clean"] == genre_info["genre"]]
            if pool.empty:
                pool = self.songs[self.songs["track_genre"] == genre_info["genre"]]
            if pool.empty:
                continue

            pop = pool["popularity"].values.astype(float)
            probs = (pop - pop.min() + 1)
            probs /= probs.sum()

            idx = rng.choice(len(pool), size=min(int(n), len(pool)),
                             replace=False, p=probs)
            sampled.append(pool.iloc[idx])

        if not sampled:
            return pd.DataFrame()

        playlist = pd.concat(sampled, ignore_index=True).head(n_songs)

        keep = ["track_id", "track_name", "artists", "album_name", "track_genre_clean",
                "popularity", "duration_ms", "danceability", "energy",
                "acousticness", "valence", "instrumentalness",
                "liveness", "speechiness"]
        playlist = playlist[[c for c in keep if c in playlist.columns]].copy()
        playlist = playlist.rename(columns={"track_genre_clean": "genre"})

        if len(playlist) > 4:
            playlist = playlist.sort_values("energy").reset_index(drop=True)
            mid = len(playlist) // 2
            playlist = pd.concat([playlist.iloc[:mid],
                                  playlist.iloc[mid:][::-1]], ignore_index=True)

        return playlist

    def generate_playlist(self, venue_data, model="ridge",
                          n_songs=25, top_k_genres=5, seed=None):
        X, venue_info = self.build_feature_vector(venue_data)
        profile = self.predict_audio_profile(X, model=model)
        genres = self.get_nearest_genres(profile, k=top_k_genres)
        playlist = self.sample_songs(genres, n_songs=n_songs, seed=seed)

        return {
            "playlist": playlist,
            "audio_profile": dict(zip(self.audio_features, profile.tolist())),
            "nearest_genres": genres,
            "venue_info": venue_info,
            "radar_data": {
                "dimensions": self.audio_features,
                "predicted": profile.tolist(),
                "genre_means": {g["genre"]: g["profile"] for g in genres},
            },
        }

    def get_feature_importance(self, top_n=10):
        W = self.ridge_W[:-1, :]
        importance = {}
        for j, name in enumerate(self.audio_features):
            top_idx = np.argsort(np.abs(W[:, j]))[-top_n:][::-1]
            importance[name] = [
                {"feature": self.feature_names[i], "weight": float(W[i, j])}
                for i in top_idx]
        return importance
