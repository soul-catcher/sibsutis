#ifndef DOCWINDOW_SAMSONOV_MATAFONOV_H
#define DOCWINDOW_SAMSONOV_MATAFONOV_H

#include <QTextEdit>

class DocWindow_Samsonov_Matafonov : public QTextEdit
{
    Q_OBJECT
private:
    QString m_strFileName;
public:
    DocWindow_Samsonov_Matafonov(QWidget* pwgt = nullptr);
signals:
    void changeWindowTitle(const QString&);
public slots:
    void slotLoad();
    void slotSave();
    void slotSaveAs();
    void slotColor();
};

#endif // DOCWINDOW_SAMSONOV_MATAFONOV_H
