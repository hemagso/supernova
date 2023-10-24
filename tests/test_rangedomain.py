import pytest
from pydantic import ValidationError
from supernova.metadata import RangeDomain

# Assuming the Interval class and parse_interval_string function are in a module named interval_parser


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[1, 3]", (1, True, 3, True)),
        ("(1, 3]", (1, False, 3, True)),
        ("[1, +inf)", (1, True, float("inf"), False)),
    ],
)
def test_valid_intervals_parse(value, expected):
    # Test with valid intervals
    assert RangeDomain.from_str(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[1, 3", "Invalid range interval format"),
        ("1, 3]", "Invalid range interval format"),
        ("[one, 3]", "Invalid number format in the interval"),
        ("[5, 3]", "Start value must be less than or equal to end value"),
    ],
)
def test_invalid_intervals_parse(value, expected):
    # Test with invalid interval formats
    with pytest.raises(ValueError, match=expected):
        RangeDomain.from_str(value)


def test_valid_intervals():
    interval = RangeDomain(value="[1, 3]", description="Test range domain")
    assert interval.start == 1
    assert interval.include_start == True
    assert interval.end == 3
    assert interval.include_end == True

    interval = RangeDomain(value="(1, 3]", description="Test range domain")
    assert interval.start == 1
    assert interval.include_start == False
    assert interval.end == 3
    assert interval.include_end == True


# Execute tests
if __name__ == "__main__":
    pytest.main()
