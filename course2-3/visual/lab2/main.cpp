#include "startdialog_samsonov_matafonov.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    StartDialog_Samsonov_Matafonov startDialog;
    startDialog.show();

    return a.exec();
}
