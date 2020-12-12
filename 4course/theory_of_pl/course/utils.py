def split_by(string: str, symbol: str) -> list[str]:
    return list(filter(None, map(str.strip, string.split(symbol))))


class WrongRulesException(Exception):
    def __init__(self, message):
        self.message = message


def parse_rules(rules_str: str, lambda_symbol: str) -> dict[str, list[str]]:
    dictionary = {}
    for rule in rules_str.splitlines():
        rule = rule.strip()
        if not rule:
            continue
        left, sep, right = rule.partition('->')
        left = left.strip()
        right = right.strip()
        if not sep:
            raise WrongRulesException(f'В правиле {rule} отсутствует символ ->')
        if not left:
            raise WrongRulesException(f'В правиле {rule} отсутствует левая часть')
        if not right:
            raise WrongRulesException(f'В правиле {rule} отсутствует правая часть')
        dictionary[left.strip()] = [seq if seq != lambda_symbol else '' for seq in split_by(right, '|')]
    return dictionary
