#include <stddef.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#define PORT 4444
#define MAXLINE 1024

int max(int x, int y) 
{ 
    if (x > y) 
        return x; 
    else
        return y; 
} 

int main() {
    int serverSocketTCP, serverSocketUDP;
    struct sockaddr_in serverAddrTCP, serverAddrUDP; 

    socklen_t sockLenTCP = sizeof(serverAddrTCP);
    socklen_t sockLenUDP = sizeof(serverAddrUDP);

    serverSocketTCP = socket(AF_INET, SOCK_STREAM, 0);
    if(serverSocketTCP < 0) {
		perror("Error in TCP connection.\n");
		exit(1);
	}
	printf("Server TCP Socket is created.\n");
    
    serverSocketUDP = socket(AF_INET, SOCK_DGRAM, 0);
    if(serverSocketUDP < 0) {
		perror("Error in UDP connection.\n");
		exit(1);
	}
	printf("Server UDP Socket is created.\n");

    serverAddrUDP.sin_family = AF_INET;
    serverAddrUDP.sin_port = htons(PORT);
    serverAddrUDP.sin_addr.s_addr = htonl(INADDR_ANY);
    int retUDP = bind(serverSocketUDP, (struct sockaddr*)&serverAddrUDP, sockLenUDP);
    if(retUDP < 0) {
        perror("Error in binding UDP.\n");
        exit(1);
    }

    serverAddrTCP.sin_family = AF_INET;
    serverAddrTCP.sin_port = htons(PORT);
    serverAddrTCP.sin_addr.s_addr = htonl(INADDR_ANY);
    int retTCP = bind(serverSocketTCP, (struct sockaddr*)&serverAddrTCP, sockLenTCP);
    if(retTCP < 0) {
        perror("Error in binding TCP.\n");
        exit(1);
    }

    if(listen(serverSocketTCP, 4) == 0) {
        printf("Listening...\n");
    }
    else {
        perror("Error in binding.\n");
        exit(1);
    }

    fd_set rSet;

    FD_ZERO(&rSet);
    int maxfdp1 = max(serverSocketUDP, serverSocketTCP) + 1;
    while(1) {
        FD_SET(serverSocketTCP, &rSet);
        FD_SET(serverSocketUDP, &rSet);

        select(FD_SETSIZE, &rSet, NULL, NULL, NULL);

        if(FD_ISSET(serverSocketTCP, &rSet)) {
            struct sockaddr_in clientAddr;
            socklen_t sockLen = sizeof(clientAddr);
            int sockCl = accept(serverSocketTCP, (struct sockaddr*)&clientAddr, &sockLen);
            int n;
            FILE *fp;
            char buffer[MAXLINE];

            fp = fopen("file_write_TCP.txt", "w");
            while(1) {
                n = recv(sockCl, &buffer, MAXLINE, 0);
                if(n <= 0) {
                    break;
                }
                fprintf(fp, "%s", buffer);
                bzero(buffer, MAXLINE);
            }
            fclose(fp);
            printf("Wrote TCP file.\n");
        }

        if(FD_ISSET(serverSocketUDP, &rSet)) {
            struct sockaddr_in clientAddr;
            socklen_t sockLen = sizeof(clientAddr);

            int n;
            FILE *fp;
            char buffer[MAXLINE];
            fp = fopen("file_write_UDP.txt", "a");
            n = recvfrom(serverSocketUDP, &buffer, MAXLINE, 0, (struct sockaddr*)&clientAddr, &sockLen);
            fprintf(fp, "%s", buffer);
            bzero(buffer, MAXLINE);
            printf("Wrote UDP file.\n");
            fclose(fp);
        }
    }
    return 0;
}