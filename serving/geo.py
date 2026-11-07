"""Geographic helpers shared by batch feature engineering and streaming.

The synthetic generator stores a location *name* (``Bangalore_IN``), while
real Okta events store a free-text string such as ``Bangalore, Karnataka, IN``.
Both are normalised here so the streaming path computes the same geo-velocity
feature as the batch pipeline.
"""

import math

GEO_LOCATIONS = {
    "Bangalore_IN": (12.97, 77.59),
    "Mumbai_IN": (19.07, 72.87),
    "Delhi_IN": (28.61, 77.20),
    "London_UK": (51.50, -0.12),
    "NewYork_US": (40.71, -74.00),
    "Frankfurt_DE": (50.11, 8.68),
    "Singapore_SG": (1.35, 103.82),
    "SaoPaulo_BR": (-23.55, -46.63),
    "Sydney_AU": (-33.87, 151.21),
    "Moscow_RU": (55.75, 37.61),
}

# City-level aliases so Okta's "City, State, Country" shape still resolves.
_CITY_ALIASES = {
    "bangalore": "Bangalore_IN",
    "bengaluru": "Bangalore_IN",
    "mumbai": "Mumbai_IN",
    "delhi": "Delhi_IN",
    "new delhi": "Delhi_IN",
    "london": "London_UK",
    "newyork": "NewYork_US",
    "new york": "NewYork_US",
    "frankfurt": "Frankfurt_DE",
    "singapore": "Singapore_SG",
    "saopaulo": "SaoPaulo_BR",
    "sao paulo": "SaoPaulo_BR",
    "sydney": "Sydney_AU",
    "moscow": "Moscow_RU",
}


def coords_for(geo_location):
    """Return ``(lat, lon)`` for a synthetic name or a free-text location.

    Returns ``(None, None)`` when the location is unknown/missing. Missing geo
    is normal in real Okta data (VPNs, proxies) and must not be treated as
    suspicious.
    """
    if not geo_location:
        return None, None

    text = str(geo_location).strip()
    if text in GEO_LOCATIONS:
        return GEO_LOCATIONS[text]

    lower = text.lower()
    if lower in _CITY_ALIASES:
        return GEO_LOCATIONS[_CITY_ALIASES[lower]]

    # "Bangalore, Karnataka, IN" -> try each comma-separated part.
    for part in (p.strip().lower() for p in text.split(",")):
        if part in _CITY_ALIASES:
            return GEO_LOCATIONS[_CITY_ALIASES[part]]

    return None, None


def haversine_coords(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two points."""
    radius_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def haversine_km(geo1, geo2):
    """Distance in km between two known location names."""
    if geo1 not in GEO_LOCATIONS or geo2 not in GEO_LOCATIONS:
        return 0.0
    lat1, lon1 = GEO_LOCATIONS[geo1]
    lat2, lon2 = GEO_LOCATIONS[geo2]
    return haversine_coords(lat1, lon1, lat2, lon2)
