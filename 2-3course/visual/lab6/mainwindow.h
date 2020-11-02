#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "samsonov_matafonov.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    Samsonov_Matafonov *myform;
signals:
    void sendData(QString str);

private slots:
    void onButtonSend();
    void on_pushButton_load_clicked();
};

#endif // MAINWINDOW_H
