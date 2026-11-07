from serving.geo import coords_for, haversine_km


def test_known_locations_have_positive_distance():
    assert haversine_km("Bangalore_IN", "Mumbai_IN") > 0


def test_free_text_okta_location_resolves():
    assert coords_for("Bangalore, Karnataka, IN") == (12.97, 77.59)
    assert coords_for("New York, NY, US") == (40.71, -74.00)


def test_unknown_location_is_not_flagged():
    assert coords_for("") == (None, None)
    assert coords_for("Somewhere, Nowhere") == (None, None)
