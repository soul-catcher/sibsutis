#ifndef SDIPROGRAM_SAMSONOV_MATAFONOV_H
#define SDIPROGRAM_SAMSONOV_MATAFONOV_H

#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QMenuBar>
#include <QStatusBar>
#include <QApplication>
#include "docwindow_samsonov_matafonov.h"

namespace Ui {
class SDIProgram_Samsonov_Matafonov;
}

class SDIProgram_Samsonov_Matafonov : public QMainWindow
{
    Q_OBJECT

public:
    SDIProgram_Samsonov_Matafonov(QWidget *pwgt = nullptr) : QMainWindow(pwgt)
    {
        QMenu* pmnuFile = new QMenu("&File");
        QMenu* pmnuHelp = new QMenu("&Help");
        DocWindow_Samsonov_Matafonov* pdoc = new DocWindow_Samsonov_Matafonov;
        pmnuFile->addAction("&Open...", pdoc, SLOT(slotLoad()), QKeySequence("CTRL+O"));
        pmnuFile->addAction("&Save",  pdoc, SLOT(slotSave()), QKeySequence("CTRL+S"));
        pmnuFile->addAction("&Save As...",  pdoc, SLOT(slotSaveAs()));
        pmnuFile->addSeparator();
        pmnuFile->addAction("&Quit", qApp, SLOT(quit()), QKeySequence("CTRL+Q"));

        pmnuHelp->addAction("&About", this, SLOT(slotAbout()), Qt::Key_F1);

        menuBar()->addMenu(pmnuFile);
        menuBar()->addMenu(pmnuHelp);
        setCentralWidget(pdoc);

        connect(pdoc, SIGNAL(changeWindowTitle(const QString&)), SLOT(slotChangeWindowTitle(const QString&)));
        statusBar()->showMessage("Ready", 2000);

        pmnuFile->addAction("&Color", pdoc, SLOT(slotColor()));
    }

    ~SDIProgram_Samsonov_Matafonov();
public slots:
    void slotAbout()
    {
        QMessageBox::about(this, "Application", "Samsonov, Matafonov");
    }
    void slotChangeWindowTitle(const QString& str)
    {
        setWindowTitle(str);
    }
private:
    Ui::SDIProgram_Samsonov_Matafonov *ui;
};

#endif // SDIPROGRAM_SAMSONOV_MATAFONOV_H
