#ifndef STARTDIALOG_SAMSONOV_MATAFONOV_H
#define STARTDIALOG_SAMSONOV_MATAFONOV_H

#include <QWidget>
#include <QPushButton>
#include <QMessageBox>
#include "inputdialog_samsonov_matafonov.h"

class StartDialog_Samsonov_Matafonov : public QPushButton
{
    Q_OBJECT
public:
    StartDialog_Samsonov_Matafonov(QWidget* pwgt = nullptr) : QPushButton("нажми", pwgt)
    {
        connect(this, SIGNAL(clicked()), SLOT(slotButtonClicked()));
    }

signals:

public slots:
    void slotButtonClicked()
    {
        InputDialog_Samsonov_Matafonov* pInputDialog = new InputDialog_Samsonov_Matafonov;
        if (pInputDialog->exec() == QDialog::Accepted) {
            QMessageBox::information(nullptr, "Ваша информация: ", "Имя: " + pInputDialog->firstName() + " Фамилия: " + pInputDialog->lastName());
        }
        delete pInputDialog;
    }
};

#endif // STARTDIALOG_SAMSONOV_MATAFONOV_H
