#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

void oldman() {
    fprintf(stdout, "I'm not yet dead! My ID is %i\n", (int) getpid());
}

void recreation() {
    fprintf(stdout, "Who I am? My ID is %i\n", (int) getpid());
}

int main() {
    pid_t child_pid, parent_pid;
    int i = 0;
    fprintf(stdout, "Before RECREATION %i\n",
            parent_pid = (int) getpid());
    child_pid = fork();
    while (i++ < 30)
        if (child_pid != 0)
            oldman();
        else
            recreation();
    return 0;
}