#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 4444

int main()
{
    int clientSocket;
    struct sockaddr_in client;

    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if(clientSocket < 0){
		printf("Error in connection.\n");
		exit(1);
	}
	printf("Client Socket is created.\n");

    client.sin_family = AF_INET;
    client.sin_port = htons(PORT);
    client.sin_addr.s_addr = inet_addr("127.0.0.1");

    int ret = connect(clientSocket, (struct sockaddr*)&client, sizeof(client));
    if(ret < 0)
    {
        printf("Error in connection.\n");
        exit(1);
    }
    printf("Connected to the Server.\n");

    for(int i = 1; i <= 10; i++)
    {
        send(clientSocket, &i, 4, 0);
        sleep(1);
    }
    close(clientSocket);

    return 0;
}
