#ifndef MAINWINDOW_SAMSONOV_MATAFONOV_H
#define MAINWINDOW_SAMSONOV_MATAFONOV_H

#include <QMainWindow>

namespace Ui {
class MainWindow_Samsonov_Matafonov;
}

class MainWindow_Samsonov_Matafonov : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow_Samsonov_Matafonov(QWidget *parent = nullptr);
    ~MainWindow_Samsonov_Matafonov();

private:
    Ui::MainWindow_Samsonov_Matafonov *ui;
};

#endif // MAINWINDOW_SAMSONOV_MATAFONOV_H
