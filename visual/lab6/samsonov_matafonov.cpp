#include "samsonov_matafonov.h"
#include "ui_samsonov_matafonov.h"
#include <QFileDialog>
#include <QTextStream>
#include <QTextEdit>
#include <QMessageBox>

Samsonov_Matafonov::Samsonov_Matafonov(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Samsonov_Matafonov)
{
    ui->setupUi(this);
}

void Samsonov_Matafonov::recieveData(QString str)
{
    QStringList lst = str.split("*");
    ui->textEdit->setText(lst.at(1) + "\n" + lst.at(0));
    if (lst.size() > 1) {
        QImage image1(lst.at(0));
        ui->label->setPixmap(QPixmap::fromImage(image1));
    }
}

Samsonov_Matafonov::~Samsonov_Matafonov()
{
    delete ui;
}

void Samsonov_Matafonov::on_buttonBox_clicked(QAbstractButton *button)
{
    if (button->text() == "Reset") {
        ui->textEdit->clear();
        ui->label->clear();
    } else if (button->text() == "Save") {
        QString filename = QFileDialog::getSaveFileName(nullptr, "Сохранить как", QDir::currentPath());
        QFile file(filename);
        if (file.open(QIODevice::WriteOnly)) {
            QTextStream(&file) << ui->textEdit->toPlainText();
            file.close();
            QMessageBox::information(this, "Файл сохранён", "Файл успешно сохранён");

        }
    } else if (button->text() == "Load") {
        QStringList inf = ui->textEdit->toPlainText().split("\n");
        QImage image2(inf.at(4));
        ui->label->setPixmap(QPixmap::fromImage(image2));
    }
}
