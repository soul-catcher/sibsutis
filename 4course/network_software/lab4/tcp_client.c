#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#define PORT 4444
#define SIZE 1024

void send_file(FILE *fp, int clientSocket) {
    int n;
    char data[SIZE] = {0};

    while(fgets(data, SIZE, fp) != NULL) {
        if(send(clientSocket, data, sizeof(data), 0) == -1) {
            perror("Error in sending file");
            exit(1);
        }
        bzero(data, SIZE);
    }
}

int main() {
    FILE *fp;
    int clientSocket;
    struct sockaddr_in client;

    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if(clientSocket < 0) {
		perror("Error in connection.\n");
		exit(1);
	}
	printf("Client Socket is created.\n");

    client.sin_family = AF_INET;
    client.sin_port = htons(PORT);
    client.sin_addr.s_addr = inet_addr("127.0.0.1");

    int ret = connect(clientSocket, (struct sockaddr*)&client, sizeof(client));
    if(ret < 0) {
        perror("Error in connection.\n");
        exit(1);
    }
    printf("Connected to the Server.\n");

    fp = fopen("file_read.txt", "r");
    if(fp == NULL) {
        perror("Error in reading file.\n");
    }
    send_file(fp, clientSocket);
    printf("File send successfully.\n");
    close(clientSocket);

    return 0;
}