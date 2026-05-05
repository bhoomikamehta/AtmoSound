"""
AtmoSound — Google Maps API Helpers
====================================
Parse Google Maps Places API responses into the venue_data dict
expected by AtmoSoundPipeline.build_feature_vector().
"""

import re


# Maps API camelCase field → our snake_case column name
_BOOL_FIELD_MAP = {
    "goodForChildren": "good_for_children",
    "goodForGroups": "good_for_groups",
    "goodForWatchingSports": "good_for_watching_sports",
    "allowsDogs": "allows_dogs",
    "liveMusic": "live_music",
    "outdoorSeating": "outdoor_seating",
    "reservable": "reservable",
    "servesBeer": "serves_beer",
    "servesCocktails": "serves_cocktails",
    "servesWine": "serves_wine",
    "servesCoffee": "serves_coffee",
    "servesBreakfast": "serves_breakfast",
    "servesBrunch": "serves_brunch",
    "servesDinner": "serves_dinner",
    "servesLunch": "serves_lunch",
    "servesVegetarianFood": "serves_vegetarian_food",
    "servesDessert": "serves_dessert",
    "menuForChildren": "menu_for_children",
}


def parse_google_maps_response(api_response):
    """Convert a Google Maps Places API JSON response into a venue_data dict.

    Supports both the new Places API (v1) and the legacy format.
    """
    venue = {}

    # Rating
    venue["rating"] = api_response.get("rating")

    # Price level
    venue["price_level"] = api_response.get(
        "priceLevel", api_response.get("price_level"))

    # Primary type
    venue["primary_type"] = api_response.get(
        "primaryType", api_response.get("primary_type"))
    if venue["primary_type"] is None:
        types = api_response.get("types", [])
        if types:
            venue["primary_type"] = types[0]

    # Neighbourhood (from address components)
    venue["neighbourhood"] = ""
    components = api_response.get(
        "addressComponents", api_response.get("address_components", []))
    for comp in components:
        comp_types = comp.get("types", [])
        if "neighborhood" in comp_types or "sublocality" in comp_types:
            venue["neighbourhood"] = comp.get(
                "longText", comp.get("long_name", ""))
            break

    # Text summaries
    reviews = api_response.get("reviews", [])
    if reviews:
        texts = [r.get("text", {}).get("text", r.get("text", ""))
                 for r in reviews if r]
        venue["review_summary"] = " ".join(t for t in texts if isinstance(t, str))
    else:
        venue["review_summary"] = ""

    venue["generative_summary"] = api_response.get(
        "generativeSummary", {}).get("overview", {}).get(
        "text", api_response.get("editorial_summary", {}).get("overview", ""))

    # Boolean attributes (camelCase and snake_case keys)
    for api_key, col in _BOOL_FIELD_MAP.items():
        val = api_response.get(api_key, api_response.get(col))
        if val is not None:
            venue[col] = val

    return venue


def extract_place_id_from_url(url):
    """Try to extract a Google Maps Place ID from a URL.

    Returns the Place ID string, or None if it can't be resolved
    without an API call.
    """
    if not url:
        return None

    # Direct Place ID
    if url.startswith("ChIJ") or url.startswith("Eh"):
        return url.strip()

    # URL with place_id parameter
    match = re.search(r'place_id[=:]([A-Za-z0-9_-]+)', url)
    if match:
        return match.group(1)

    # URL with hex data parameter
    match = re.search(r'!1s(0x[0-9a-f]+:0x[0-9a-f]+)', url)
    if match:
        return match.group(1)

    return None
