#ifndef SAMSONOV_MATAFONOV_H
#define SAMSONOV_MATAFONOV_H

#include <QWidget>
#include <QAbstractButton>

namespace Ui {
class Samsonov_Matafonov;
}

class Samsonov_Matafonov : public QWidget
{
    Q_OBJECT

public:
    explicit Samsonov_Matafonov(QWidget *parent = nullptr);
    ~Samsonov_Matafonov();

private:
    Ui::Samsonov_Matafonov *ui;
public slots:
    void recieveData(QString str);
private slots:
    void on_buttonBox_clicked(QAbstractButton *button);
};

#endif // SAMSONOV_MATAFONOV_H
