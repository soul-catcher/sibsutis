#include "mainwindow_samsonov_matafonov.h"
#include "ui_mainwindow_samsonov_matafonov.h"

MainWindow_Samsonov_Matafonov::MainWindow_Samsonov_Matafonov(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow_Samsonov_Matafonov)
{
    ui->setupUi(this);
}

MainWindow_Samsonov_Matafonov::~MainWindow_Samsonov_Matafonov()
{
    delete ui;
}
