#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <argp.h>
#include <unistd.h>

int n_opt = 10;
char *ip_arg, i_opt;

struct argp_option options[] = {{NULL,      'i', "NUM", 0, "Transmission delay in seconds"},
                                {NULL, 'n', "NUM", 0, "Number of iterations"},
                                {0}};

static int parse_opt(int key, char *arg, struct argp_state *state) {
    int *arg_count = state->input;

    int tmp = arg ? strtol(arg, NULL, 10) : 0;
    if (tmp < 1 && (key == 'i' || key == 'n')) {
        fprintf(stderr, "error parsing option -%c", key);
        exit(EXIT_FAILURE);
    }

    switch (key) {
        case 'i':
            i_opt = tmp;
            break;
        case 'n':
            n_opt = tmp;
            break;
        case ARGP_KEY_ARG:
            if (*arg_count > 0)
                ip_arg = arg;
            (*arg_count)--;
            break;
        case ARGP_KEY_END:
            printf("\n");
            if (*arg_count > 0)
                argp_failure(state, 1, 0, "too few arguments");
            else if (*arg_count < 0)
                argp_failure(state, 1, 0, "too many arguments");
            break;
    }
    return 0;
}

int main(int argc, char **argv) {
    int sockfd, arg_count = 1;
    struct sockaddr_in serv_addr;
    char *srv_ip = NULL, *srv_port = NULL;

    struct argp argp = {options, parse_opt, "IP:PORT"};
    argp_parse(&argp, argc, argv, 0, 0, &arg_count);

    for (int i = 0, n = strlen(ip_arg); i < n; i++)
        if (ip_arg[i] == ':') {
            srv_ip = malloc(15 * sizeof(char));
            srv_port = malloc(5 * sizeof(char));
            strncpy(srv_ip, ip_arg, i);
            strncpy(srv_port, ip_arg + i + 1, n - i - 1);
            break;
        }

    if (!srv_ip || !srv_port) {
        fprintf(stderr, "failed to parse ip\n");
        exit(EXIT_FAILURE);
    }

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        fprintf(stderr, "socket failed\n");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = strtol(srv_port, NULL, 10);

    if (inet_pton(AF_INET, srv_ip, &serv_addr.sin_addr) <= 0) {
        fprintf(stderr, "pton failed\n");
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr))) {
        fprintf(stderr, "connect failed\n");
        exit(EXIT_FAILURE);
    }


    for (int i = 0; i < n_opt; i++) {
        send(sockfd, &i_opt, 1, 0);
        sleep(i_opt);
    }

    return 0;
}