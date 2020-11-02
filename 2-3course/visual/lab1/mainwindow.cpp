#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "ui_auth.h"
#include "auth.h"
#include "QFileDialog"
#include "QTextDocumentWriter"

void MainWindow::About_Lab1()
{
    Auth *dg = new Auth();
    dg->show();
}

void MainWindow::slotOpen()
{
    QString filename = QFileDialog::getOpenFileName(nullptr, "Открыть файл", QDir::currentPath(), "*.cpp *.txt");
    QFile file(filename);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        ui->textEdit->setPlainText(file.readAll());
}

void MainWindow::slotSave()
{
    QString filename = QFileDialog::getSaveFileName(nullptr, "Сохранить файл", QDir::currentPath(), "*.cpp *.txt");
    QTextDocumentWriter writer;
    writer.setFileName(filename);
    writer.write(ui->textEdit->document());
}

void MainWindow::slotClear()
{
    ui->textEdit->clear();
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
    {
        ui->setupUi(this);
        connect(ui->action_2, SIGNAL(triggered()), this, SLOT(About_Lab1()));

        QAction* pactOpen = new QAction("file open action", nullptr);
        pactOpen->setText("&Открыть");
        pactOpen->setShortcut(QKeySequence("CTRL+O"));
        pactOpen->setToolTip("Открытие документа");
        pactOpen->setStatusTip("Открыть файл");
        pactOpen->setWhatsThis("Открыть файл");
        pactOpen->setIcon(QPixmap("1.png"));
        connect(pactOpen, SIGNAL(triggered()), SLOT(slotOpen()));
        QMenu* pmnuFile=new QMenu("&Файл");
        pmnuFile->addAction(pactOpen);
        menuBar()->addMenu(pmnuFile);

        QAction* pactSave = new QAction("file save action", nullptr);
        pactSave->setText("&Сохранить");
        pactSave->setShortcut(QKeySequence("CTRL+S"));
        pactSave->setToolTip("Сохранение документа");
        pactSave->setStatusTip("Сохранить файл");
        pactSave->setWhatsThis("Сохранить файл");
        pactSave->setIcon(QPixmap("1.png"));
        connect(pactSave, SIGNAL(triggered()), SLOT(slotSave()));
        pmnuFile->addAction(pactSave);
        menuBar()->addMenu(pmnuFile);


        QAction* pactClear = new QAction("field clear action", nullptr);
        pactClear->setText("&Очистить");
        pactClear->setToolTip("Очистить поле");
        pactClear->setStatusTip("Очистить поле");
        pactClear->setWhatsThis("Очистить поле");
        connect(pactClear, SIGNAL(triggered()), SLOT(slotClear()));
//        pmnuFile->addAction(pactClear);
//        menuBar()->addMenu(pmnuFile);



        ui->mainToolBar->addAction(pactOpen);
        ui->mainToolBar->addAction(pactSave);
        ui->mainToolBar->addAction(pactClear);

    }



MainWindow::~MainWindow()
{
    delete ui;
}


