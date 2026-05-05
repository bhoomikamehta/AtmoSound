"""
Venue-Adaptive Playlist Generation — Exploratory Data Analysis
Generates all EDA figures for the project proposal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Style ──
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})
PALETTE = ["#E8634A","#3A8FB7","#6B9F5B","#D4A843","#8B6BAE","#C75B8E","#4AADA8","#D17B46"]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


# LOAD DATA

places = pd.read_csv("manhattan_places.csv", encoding="utf-8-sig")
songs = pd.read_csv("songs_dataset.csv", index_col=0)

print(f"Places: {places.shape[0]} rows, {places.shape[1]} cols")
print(f"Songs:  {songs.shape[0]} rows, {songs.shape[1]} cols")


# FIGURE 1 — Venues by Category

fig, ax = plt.subplots(figsize=(8, 4))
counts = places["category"].value_counts()
bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE[:len(counts)])
ax.set_xlabel("Number of Venues")
ax.set_title("Figure 1: Venue Count by Category")
ax.bar_label(bars, padding=4, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_category_counts.png", dpi=150)
plt.close()


# FIGURE 2 — Rating Distribution (Histogram)

fig, ax = plt.subplots(figsize=(8, 4))
places["rating"].dropna().hist(bins=30, ax=ax, color="#3A8FB7", edgecolor="white", alpha=0.85)
ax.axvline(places["rating"].median(), color="#E8634A", ls="--", lw=2, label=f'Median = {places["rating"].median():.1f}')
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
ax.set_title("Figure 2: Distribution of Venue Ratings")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_rating_hist.png", dpi=150)
plt.close()


# FIGURE 3 — Price Level Distribution

fig, ax = plt.subplots(figsize=(6, 5))
price_counts = places["price_level"].dropna().value_counts()
labels = [l.replace("PRICE_LEVEL_", "").title() for l in price_counts.index]
ax.pie(price_counts.values, labels=labels, autopct="%1.0f%%", colors=PALETTE[3:], startangle=140)
ax.set_title("Figure 3: Price Level Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/fig03_price_pie.png", dpi=150)
plt.close()


# FIGURE 4 — Top 15 Neighbourhoods

fig, ax = plt.subplots(figsize=(10, 5))
nh = places["neighbourhood"].value_counts().head(15)
bars = ax.bar(range(len(nh)), nh.values, color=[PALETTE[i % len(PALETTE)] for i in range(len(nh))])
ax.set_xticks(range(len(nh)))
ax.set_xticklabels(nh.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Number of Venues")
ax.set_title("Figure 4: Top 15 Neighbourhoods by Venue Count")
ax.bar_label(bars, fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{OUT}/fig04_neighbourhood_bar.png", dpi=150)
plt.close()


# FIGURE 5 — Boolean Attribute Prevalence

bool_cols = [
    "serves_dessert","good_for_groups","serves_lunch","serves_coffee",
    "serves_dinner","serves_beer","serves_wine","reservable",
    "good_for_children","serves_cocktails","outdoor_seating",
    "serves_brunch","serves_breakfast","serves_vegetarian_food",
    "menu_for_children","allows_dogs","live_music","good_for_watching_sports"
]
bool_pcts = []
for c in bool_cols:
    col = places[c].dropna()
    if len(col) > 0:
        pct = (col == True).sum() / len(col) * 100
    else:
        pct = 0
    bool_pcts.append(pct)

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = range(len(bool_cols))
colors = ["#6B9F5B" if p > 50 else "#D4A843" if p > 20 else "#E8634A" for p in bool_pcts]
bars = ax.barh(y_pos, bool_pcts, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace("_", " ").title() for c in bool_cols], fontsize=9)
ax.set_xlabel("% True (among non-null)")
ax.set_title("Figure 5: Boolean Attribute Prevalence")
ax.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=8)
ax.set_xlim(0, 110)
plt.tight_layout()
plt.savefig(f"{OUT}/fig05_bool_prevalence.png", dpi=150)
plt.close()


# FIGURE 6 — Missing Values Heatmap

# Select columns with meaningful missingness
miss_cols = [c for c in places.columns if places[c].isnull().mean() > 0.01 and places[c].isnull().mean() < 1.0]
miss_pct = places[miss_cols].isnull().mean().sort_values(ascending=False) * 100

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(miss_pct.index[::-1], miss_pct.values[::-1], color="#D17B46")
ax.set_xlabel("% Missing")
ax.set_title("Figure 6: Missing Value Rates (Places Dataset)")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig06_missing_values.png", dpi=150)
plt.close()


# FIGURE 7 — Geo Scatter (Lat/Lng by Neighbourhood)

top_nh = places["neighbourhood"].value_counts().head(8).index.tolist()
subset = places[places["neighbourhood"].isin(top_nh)].copy()

fig, ax = plt.subplots(figsize=(9, 8))
for i, nh_name in enumerate(top_nh):
    mask = subset["neighbourhood"] == nh_name
    ax.scatter(subset.loc[mask, "longitude"], subset.loc[mask, "latitude"],
               s=12, alpha=0.5, label=nh_name, color=PALETTE[i % len(PALETTE)])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Figure 7: Venue Locations by Neighbourhood")
ax.legend(fontsize=8, markerscale=2, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig07_geo_scatter.png", dpi=150)
plt.close()


# FIGURE 8 — Rating Boxplot by Category

fig, ax = plt.subplots(figsize=(8, 4))
order = places.groupby("category")["rating"].median().sort_values(ascending=False).index
sns.boxplot(data=places, x="category", y="rating", order=order, palette=PALETTE, ax=ax)
ax.set_title("Figure 8: Rating Distribution by Category")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/fig08_rating_boxplot.png", dpi=150)
plt.close()


# FIGURE 9 — Primary Type Breakdown (Top 10)

fig, ax = plt.subplots(figsize=(9, 5))
pt = places["primary_type"].value_counts().head(10)
bars = ax.barh(pt.index[::-1], pt.values[::-1], color=PALETTE[:len(pt)])
ax.set_xlabel("Count")
ax.set_title("Figure 9: Top 10 Primary Types")
ax.bar_label(bars, padding=4, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig09_primary_type.png", dpi=150)
plt.close()

# SONGS DATASET FIGURES

AUDIO_FEATS = ["danceability","energy","acousticness","valence",
               "instrumentalness","liveness","speechiness"]


# FIGURE 10 — Audio Feature Distributions (Histograms)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, feat in enumerate(AUDIO_FEATS):
    songs[feat].hist(bins=50, ax=axes[i], color=PALETTE[i], edgecolor="white", alpha=0.8)
    axes[i].set_title(feat.title(), fontsize=11)
    axes[i].set_xlabel("")
axes[-1].axis("off")
fig.suptitle("Figure 10: Audio Feature Distributions (Songs Dataset)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_audio_feature_hists.png", dpi=150, bbox_inches="tight")
plt.close()


# FIGURE 11 — Audio Feature Boxplots

fig, ax = plt.subplots(figsize=(10, 4))
songs[AUDIO_FEATS].boxplot(ax=ax, vert=True, patch_artist=True,
    boxprops=dict(facecolor="#3A8FB7", alpha=0.6),
    medianprops=dict(color="#E8634A", lw=2))
ax.set_title("Figure 11: Audio Feature Boxplots")
ax.set_ylabel("Value (0–1)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_audio_boxplots.png", dpi=150)
plt.close()


# FIGURE 12 — Popularity Distribution

fig, ax = plt.subplots(figsize=(8, 4))
songs["popularity"].hist(bins=50, ax=ax, color="#8B6BAE", edgecolor="white")
ax.axvline(songs["popularity"].median(), color="#E8634A", ls="--", lw=2, label=f'Median = {songs["popularity"].median()}')
ax.set_xlabel("Popularity")
ax.set_ylabel("Frequency")
ax.set_title("Figure 12: Song Popularity Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_popularity_hist.png", dpi=150)
plt.close()


# FIGURE 13 — Tempo Distribution

fig, ax = plt.subplots(figsize=(8, 4))
songs["tempo"].hist(bins=60, ax=ax, color="#4AADA8", edgecolor="white")
ax.set_xlabel("Tempo (BPM)")
ax.set_ylabel("Frequency")
ax.set_title("Figure 13: Tempo Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/fig13_tempo_hist.png", dpi=150)
plt.close()


# FIGURE 14 — Audio Feature Correlation Matrix

corr_cols = AUDIO_FEATS + ["tempo", "loudness", "popularity"]
corr = songs[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={"size": 9})
ax.set_title("Figure 14: Audio Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}/fig14_correlation_matrix.png", dpi=150)
plt.close()


# FIGURE 15 — Genre Audio Profiles (Grouped Bar)

sample_genres = ["acoustic","ambient","blues","chill","classical","club",
                 "electronic","hip-hop","indie","jazz","latin","metal","pop","rock","soul"]
genre_means = songs[songs["track_genre"].isin(sample_genres)].groupby("track_genre")[AUDIO_FEATS].mean()
genre_means = genre_means.loc[sample_genres]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(sample_genres))
width = 0.12
for i, feat in enumerate(AUDIO_FEATS):
    ax.bar(x + i * width, genre_means[feat], width, label=feat.title(), color=PALETTE[i])
ax.set_xticks(x + width * 3)
ax.set_xticklabels(sample_genres, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean Value")
ax.set_title("Figure 15: Mean Audio Feature Profiles by Genre")
ax.legend(fontsize=8, ncol=4, loc="upper right")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{OUT}/fig15_genre_profiles.png", dpi=150)
plt.close()


# FIGURE 16 — Energy vs Danceability Scatter by Genre

fig, ax = plt.subplots(figsize=(9, 7))
for i, g in enumerate(sample_genres):
    row = genre_means.loc[g]
    ax.scatter(row["energy"], row["danceability"], s=120, color=PALETTE[i % len(PALETTE)],
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.annotate(g, (row["energy"], row["danceability"]), fontsize=8,
                xytext=(6, 4), textcoords="offset points")
ax.set_xlabel("Mean Energy")
ax.set_ylabel("Mean Danceability")
ax.set_title("Figure 16: Genre Landscape — Energy vs Danceability")
ax.set_xlim(0.1, 1.0)
ax.set_ylim(0.2, 0.85)
plt.tight_layout()
plt.savefig(f"{OUT}/fig16_energy_vs_dance_scatter.png", dpi=150)
plt.close()


# FIGURE 17 — Heatmap of Genre x Audio Feature

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(genre_means, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Figure 17: Genre × Audio Feature Heatmap")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/fig17_genre_heatmap.png", dpi=150)
plt.close()


# FIGURE 18 — Rating by Price Level (Box)

price_order = ["PRICE_LEVEL_INEXPENSIVE","PRICE_LEVEL_MODERATE","PRICE_LEVEL_EXPENSIVE","PRICE_LEVEL_VERY_EXPENSIVE"]
fig, ax = plt.subplots(figsize=(8, 4))
sub = places.dropna(subset=["price_level","rating"])
sns.boxplot(data=sub, x="price_level", y="rating", order=price_order, palette=PALETTE, ax=ax)
ax.set_xticklabels(["Inexpensive","Moderate","Expensive","Very Expensive"], fontsize=9)
ax.set_xlabel("")
ax.set_title("Figure 18: Rating by Price Level")
plt.tight_layout()
plt.savefig(f"{OUT}/fig18_rating_by_price.png", dpi=150)
plt.close()

print(f"\nDone — saved 18 figures to '{OUT}/' directory.")
print("Figures list:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
