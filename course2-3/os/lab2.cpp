#include <windows.h>
#include <tchar.h>
#include <stdio.h>

#define INFO_BUFFER_SIZE 32767

void main() {
    TCHAR infoBuf[INFO_BUFFER_SIZE];

    GetSystemDirectory(infoBuf, INFO_BUFFER_SIZE);
    _tprintf(TEXT("\nSystem dir:\t\t\t%s"), infoBuf);
    GetTempPath(MAX_PATH, infoBuf);
    _tprintf(TEXT("\nTemp path:\t\t\t%s"), infoBuf);

    SYSTEM_INFO sys_inf;
    GetSystemInfo(&sys_inf);
    printf("\nCores:\t\t\t\t%d\n", sys_inf.dwNumberOfProcessors);
    switch(sys_inf.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_INTEL:
        printf("Processor Architecture:\t\tIntel x86 \t\n");
        break;
    case PROCESSOR_ARCHITECTURE_IA64:
        printf("Processor Type:\t\t\tIntel x64\n");
        break;
    case PROCESSOR_ARCHITECTURE_AMD64:
        printf("Processor Type:\t\t\tAMD 64\n");
        break;
    default:
        printf("Unknown processor architecture\n");
    }
    printf("Minimum application address:\t%#X\n", (int)sys_inf.lpMinimumApplicationAddress);
    printf("Maximum application address:\t%#X\n", (int)sys_inf.lpMaximumApplicationAddress);

    MEMORYSTATUSEX mem_st = {sizeof(mem_st)};
    GlobalMemoryStatusEx(&mem_st);
    printf("Physical memory:\n");
    printf("\t* Total:\t\t%lld\n", mem_st.ullTotalPhys);
    printf("\t* Availble:\t\t%lld\n", mem_st.ullAvailPhys);
    printf("Virtual memory:\n");
    printf("\t* Total:\t\t%lld\n", mem_st.ullTotalVirtual);
    printf("\t* Availble:\t\t%lld\n", mem_st.ullAvailVirtual);
}
