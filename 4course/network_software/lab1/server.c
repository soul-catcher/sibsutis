#include <sys/socket.h>
#include <sys/wait.h>
#include <stdio.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>

void daryl_dixon(int signal) {
    wait(0);
    printf("Дэрил Диксон убил зомби!\n");
}

int BuffWork(int sockClient) {
    char buf[1024] = "";
    while (1) {
        int bytes_read = recv(sockClient, buf, 1024, 0);
        if (bytes_read < 0) {
            printf("Плохое получение дочерним процессом.\n");
            exit(1);
        } else if (bytes_read == 0) exit(1);
        printf("Reciving massege: %s\n", buf);
        printf("Send to client massege\n");
        fflush(stdout);
    }
}

int main() {
    signal(SIGCHLD, daryl_dixon);
    int sockServer, sockClient, child;
    struct sockaddr_in servAddr;

    sockServer = socket(AF_INET, SOCK_STREAM, 0);
    if (sockServer < 0) {
        printf("Сервер не может открыть sockServer :(");
        exit(1);
    }

    servAddr.sin_family = AF_INET;
    servAddr.sin_port = 0;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockServer, (struct sockaddr *) &servAddr, sizeof(servAddr)) < 0) {
        printf("Связывание сервера неудачно");
        exit(1);
    }

    socklen_t length = sizeof(servAddr);

    if (getsockname(sockServer, (struct sockaddr *) &servAddr, &length)) {
        printf("Вызов getsockname неудачен.");
        exit(1);
    }
    printf("Сервер: номер порта - %d\n", ntohs(servAddr.sin_port));
    listen(sockServer, 3);

    while (1) {
        sockClient = accept(sockServer, 0, 0);
        if (sockClient < 0) {
            printf("Неверный socket для клиента.\n");
            exit(1);
        }

        child = fork();
        if (child < 0) {
            printf("Ошибка при порождении процесса\n");
            exit(1);
        }
        if (child == 0) {
            struct timeval tv;
            tv.tv_sec = 2;
            setsockopt(sockClient, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            close(sockServer);
            BuffWork(sockClient);
            close(sockClient);
            exit(0);
        }
        close(sockClient);
    }
}
