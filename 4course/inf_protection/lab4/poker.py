from itertools import product

from crypto import *

CARD_NAMES = tuple(range(2, 11)) + tuple('JQKA')
SUIT_NAMES = 'diamond', 'club', 'hearth', 'spade'
DECK_NAMES = tuple(product(CARD_NAMES, SUIT_NAMES))


def encrypt(deck, key, p):
    return [pow(card, key, p) for card in deck]


def cards_to_str(cards):
    return [DECK_NAMES[card - 2] for card in cards]


PLAYERS_COUNT = 11

if PLAYERS_COUNT > 23:
    print("Не может быть больше 23 игроков")
    exit(1)

deck = list(range(2, len(DECK_NAMES) + 2))
assert len(deck) == 52

p = gen_p(10 ** 8, 10 ** 9)
serv_c, serv_d = gen_c_d(p - 1)

players_decryption_keys = []
for _ in range(PLAYERS_COUNT):
    player_c, player_d = gen_c_d(p - 1)
    deck = encrypt(deck, player_c, p)  # Каждый игрок шифрует и перемешивает колоду
    random.shuffle(deck)
    players_decryption_keys.append(player_d)

players_cards = [[deck[i], deck[i + 1]] for i in range(0, PLAYERS_COUNT * 2, 2)]


for i in range(len(players_cards)):
    for decryption_key in players_decryption_keys:
        players_cards[i] = encrypt(players_cards[i], decryption_key, p)

for i, cards in enumerate(players_cards, 1):
    cards = cards_to_str(cards)
    print(f'Игрок {i} имеет карты', *cards)