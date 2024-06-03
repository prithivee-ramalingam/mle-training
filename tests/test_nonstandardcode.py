from house_price_prediction import nonstandardcode


def test_get_version():
    assert nonstandardcode.get_version() == '0.1.0'


def test_load_housing_data():
    assert len(nonstandardcode.load_housing_data()) == 20640
    assert list(nonstandardcode.load_housing_data().columns) == [
        'longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income',
        'median_house_value',
        'ocean_proximity',
    ]
