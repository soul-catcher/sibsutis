import random
import socket
import sys

import crypto


class Client:
    def __init__(self, name: str):
        self.name = name

    def register(self):
        sock = socket.socket()
        sock.connect(('localhost', 8081))
        n = int.from_bytes(sock.recv(16), sys.byteorder)
        sock.send('register'.encode())
        sock.send(self.name.encode())
        self.s = crypto.gen_mutually_prime(n)
        v = pow(self.s, 2, n)
        sock.send(v.to_bytes(16, sys.byteorder))
        resp = sock.recv(1024).decode()
        if resp == 'ok':
            print(f'Пользователь {self.name} успешно зарегистрировался с ключом {v}')
        elif resp == 'not ok':
            print(f'Пользоватль {self.name} уже зарегистрирован')
        sock.close()

    def auth(self):
        print(f'Начата аутентификация пользователя {self.name}')
        sock = socket.socket()
        sock.connect(('localhost', 8081))
        n = int.from_bytes(sock.recv(16), sys.byteorder)
        sock.send('auth'.encode())
        sock.send(self.name.encode())
        status = 'ok'
        while status == 'ok':  # Прохождение раундов пока сервер возвращает ok
            r = random.randrange(1, n)
            x = pow(r, 2, n)
            sock.send(x.to_bytes(16, sys.byteorder))
            e = int.from_bytes(sock.recv(1), sys.byteorder)
            y = r * self.s ** e % n
            sock.send(y.to_bytes(16, sys.byteorder))
            status = sock.recv(1024).decode()
        if status == 'pass':
            print('Аутентификация прошла успешно')
        elif status == 'fail':
            print('В аутентификации отказано. Вы подлый мошенник!')


if __name__ == '__main__':
    alice = Client('Alice')
    alice.register()
    alice.auth()

    # Мошенник, пытающийся авторизоваться как Alice
    eve = Client('Alice')
    eve.s = 999
    eve.auth()
