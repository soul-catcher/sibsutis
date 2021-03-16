#include <stdio.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>

#define PORT 4444

pthread_mutex_t mutex;
FILE *output;

void handleConnection(void *clientSocket)
{
    pthread_mutex_lock(&mutex);
    int buf;
    while(true)
    {
        int getBuff = recv(*(int*)clientSocket, &buf, 4, 0);
        if(getBuff == 0)
            break;
        printf("%d\n", buf);
        fprintf(output, "%d\n", buf);
        fflush(output);
    }
    close(*(int*)clientSocket);
    free(clientSocket);
    pthread_mutex_unlock(&mutex);
}

int main()
{
    int serverSocket;
    output = fopen("output.txt", "w");
    pthread_mutex_init(&mutex, NULL);

    struct sockaddr_in serverAddr;
    socklen_t serverAddr_size = sizeof(serverAddr);

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if(serverSocket < 0){
		printf("Error in connection.\n");
		exit(1);
	}
	printf("Server Socket is created.\n");

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    int ret = bind(serverSocket, (struct sockaddr*)&serverAddr, serverAddr_size);
    if(ret < 0)
    {
        printf("Error in binding.\n");
        exit(1);
    }

    if(listen(serverSocket, 3) == 0)
    {
        printf("Listening...\n");
    }
    else
    {
        printf("Error in binding.\n");
    }
    
    while(true)
    {
        struct sockaddr_in clientAddr;
        socklen_t clientAddr_size = sizeof(clientAddr);
        int *clientSocket = (int*)malloc(sizeof(int));
        *clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddr_size);

        pthread_t thread;
        pthread_create(&thread, NULL, handleConnection, (void *) clientSocket);
    }
}
