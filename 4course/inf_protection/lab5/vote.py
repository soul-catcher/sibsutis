from crypto import *
import hashlib
import math
import sys


class ReVoting(Exception):
    def __init__(self, name):
        super().__init__(f"Пользователь {name} уже проголосовал")


def my_sha(n):
    return int.from_bytes(hashlib.sha3_224(n.to_bytes(math.ceil(n.bit_length() / 8), byteorder=sys.byteorder)).digest(),
                          byteorder=sys.byteorder)


class Server:
    def __init__(self):
        P, Q = gen_p(1 << 1023, (1 << 1024) - 1), gen_p(1 << 1023, (1 << 1024) - 1)
        assert P != Q
        self.N = P * Q
        phi = (P - 1) * (Q - 1)
        self._C, self.d = gen_c_d(phi)
        self.voted = set()

    def get_blank(self, name: str, hh: int) -> int:
        if name in self.voted:
            raise ReVoting(name)
        self.voted.add(name)
        return pow(hh, self._C, self.N)


class VoteServer:
    def __init__(self, server):
        self.blanks = set()
        self.server = server

    def send_blank(self, n, s):
        if my_sha(n) == pow(s, server.d, server.N):
            self.blanks.add((n, s))
            print(f'Бланк <{n}, {s}> одобрен')
        else:
            print(f'Бланк <{n}, {s}> отклонён')
            print(my_sha(n))
            print(pow(s, server.d, server.N))


def vote(name: str, server: Server, vote_server: VoteServer, vote_num: int):
    rnd = random.getrandbits(512)
    n = rnd << 512 | vote_num
    r = gen_mutually_prime(server.N)
    h = my_sha(n)
    hh = h * pow(r, server.d, server.N) % server.N
    ss = server.get_blank(name, hh)
    s = ss * inverse(r, server.N) % server.N
    vote_server.send_blank(n, s)


server = Server()
vote_server = VoteServer(Server)
vote('Alice', server, vote_server, 1)
vote('Alice', server, vote_server, 2)
