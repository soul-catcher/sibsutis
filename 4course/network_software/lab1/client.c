#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int sock, n = 5;
    struct sockaddr_in addr;
    char *ip = "127.0.0.1";

    if (argc < 3) {
        printf("Вы забыли указать порт или сообщение.\n");
        exit(1);
    }

    sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0) {
        printf("He могу получить socket\n");
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(atoi(argv[1]));
    addr.sin_addr.s_addr = inet_addr(ip);

    if (connect(sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        printf("Клиент не может соединиться.\n");
        exit(1);
    }

    for (int i = 1; i <= n; ++i) {
        printf("\nCLIENT: Готов к пересылке\n");
        printf("Отправить на сервер сообщение %d\n", i);
        send(sock, argv[2], strlen(argv[2]), 0);

        printf("CLIENT: Пересылка завершена.\n");
        sleep(1);
    }

    close(sock);
    return 0;
}
