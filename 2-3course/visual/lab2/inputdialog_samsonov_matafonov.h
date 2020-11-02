#ifndef INPUTDIALOG_SAMSONOV_MATAFONOV_H
#define INPUTDIALOG_SAMSONOV_MATAFONOV_H

#include <QDialog>
#include <QLineEdit>

class InputDialog_Samsonov_Matafonov: public QDialog
{
    Q_OBJECT
private:
    QLineEdit* m_ptxtFirstName;
    QLineEdit* m_ptxtLastName;
public:
    InputDialog_Samsonov_Matafonov(QWidget* pwgt = nullptr);

    QString firstName() const;
    QString lastName() const;
};

#endif // INPUTDIALOG_SAMSONOV_MATAFONOV_H
