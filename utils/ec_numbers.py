def valid_ec_number(ec_number: str) -> bool:
    """Return True if ec_number has at least three numeric dot-separated fields."""
    if ec_number == "not found":
        return False
    parts = ec_number.split('.')
    return len(parts) >= 3 and all(part.isdigit() for part in parts[:3])
