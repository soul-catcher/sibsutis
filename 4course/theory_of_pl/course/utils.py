def split_by_commas(string: str) -> list[str]:
    return list(filter(None, map(str.strip, string.split(','))))


# def parse_rules(rules_str: str, lambda_symbol: str) -> dict[str, list[str]]:
