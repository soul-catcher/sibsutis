import random
import socket
import sys

import crypto

registered = {}
open('registered.txt', 'w').close()  # Создаёт файл. Если он уже есть, очищает его
p, q = crypto.gen_p(10 ** 8, 10 ** 9 - 1), crypto.gen_p(10 ** 8, 10 ** 9 - 1)
n = p * q
print(f'{n = }')
sock = socket.socket()
sock.bind(('', 8081))
sock.listen(10)

while True:
    connection, _ = sock.accept()
    connection.send(n.to_bytes(16, sys.byteorder))  # 16 - длина в байтах числа n
    command = connection.recv(1024).decode()
    if command == 'register':
        name = connection.recv(1024).decode()
        v = int.from_bytes(connection.recv(16), sys.byteorder)
        if name not in registered:
            print(f'Пользователь {name} зарегистрировался с ключом {v}')
            registered[name] = v
            connection.send('ok'.encode())
            open('registered.txt', 'a').write(f'{name} = {v}')
        else:
            print(f'Пользоватль {name} уже зарегистрирован')
            connection.send('not ok'.encode())
    if command == 'auth':
        name = connection.recv(1024).decode()
        for rnd in range(40):  # Цикл из 40 раундов
            x = int.from_bytes(connection.recv(16), sys.byteorder)
            e = random.getrandbits(1)
            connection.send(e.to_bytes(1, sys.byteorder))
            y = int.from_bytes(connection.recv(16), sys.byteorder)
            print(f'{rnd = }, {x = }, {e = }, {y = }')
            if y == 0 or y ** 2 % n != x * registered[name] ** e % n:
                connection.send('fail'.encode())
                break
            connection.send('ok'.encode() if rnd != 39 else 'pass'.encode())
        if rnd == 39:
            print(f'Аутентификация пользователя {name} прошла успешно')
        else:
            print(f'Пользователь с ником {name} мошенник!')

    connection.close()
