# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_course(object):
    def setupUi(self, course):
        if not course.objectName():
            course.setObjectName(u"course")
        course.resize(783, 667)
        self.action_open = QAction(course)
        self.action_open.setObjectName(u"action_open")
        self.action_help = QAction(course)
        self.action_help.setObjectName(u"action_help")
        self.action_author = QAction(course)
        self.action_author.setObjectName(u"action_author")
        self.action_theme = QAction(course)
        self.action_theme.setObjectName(u"action_theme")
        self.action_save = QAction(course)
        self.action_save.setObjectName(u"action_save")
        self.centrwidget = QWidget(course)
        self.centrwidget.setObjectName(u"centrwidget")
        self.gridLayout_2 = QGridLayout(self.centrwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.groupBox_3 = QGroupBox(self.centrwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.formLayout = QFormLayout(self.groupBox_3)
        self.formLayout.setObjectName(u"formLayout")
        self.groupBox = QGroupBox(self.groupBox_3)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.spinBox_min_len = QSpinBox(self.groupBox)
        self.spinBox_min_len.setObjectName(u"spinBox_min_len")

        self.gridLayout.addWidget(self.spinBox_min_len, 1, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)

        self.spinBox_max_len = QSpinBox(self.groupBox)
        self.spinBox_max_len.setObjectName(u"spinBox_max_len")
        self.spinBox_max_len.setValue(5)

        self.gridLayout.addWidget(self.spinBox_max_len, 1, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)


        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.groupBox)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setWordWrap(True)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_8)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setWordWrap(True)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_9)

        self.plainTextEdit_non_canon_chains = QPlainTextEdit(self.groupBox_3)
        self.plainTextEdit_non_canon_chains.setObjectName(u"plainTextEdit_non_canon_chains")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.plainTextEdit_non_canon_chains)

        self.plainTextEdit_canon_chains = QPlainTextEdit(self.groupBox_3)
        self.plainTextEdit_canon_chains.setObjectName(u"plainTextEdit_canon_chains")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.plainTextEdit_canon_chains)

        self.pushButton_compare_chains = QPushButton(self.groupBox_3)
        self.pushButton_compare_chains.setObjectName(u"pushButton_compare_chains")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.pushButton_compare_chains)

        self.label_status = QLabel(self.groupBox_3)
        self.label_status.setObjectName(u"label_status")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_status)


        self.gridLayout_2.addWidget(self.groupBox_3, 0, 1, 1, 1)

        self.groupBox_4 = QGroupBox(self.centrwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.plainTextEdit_actions = QPlainTextEdit(self.groupBox_4)
        self.plainTextEdit_actions.setObjectName(u"plainTextEdit_actions")
        self.plainTextEdit_actions.setReadOnly(True)

        self.verticalLayout_4.addWidget(self.plainTextEdit_actions)


        self.gridLayout_2.addWidget(self.groupBox_4, 1, 1, 1, 1)

        self.groupBox_2 = QGroupBox(self.centrwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout = QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout.addWidget(self.label_2)

        self.lineEdit_terminals = QLineEdit(self.groupBox_2)
        self.lineEdit_terminals.setObjectName(u"lineEdit_terminals")

        self.verticalLayout.addWidget(self.lineEdit_terminals)

        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout.addWidget(self.label_5)

        self.lineEdit_non_terminals = QLineEdit(self.groupBox_2)
        self.lineEdit_non_terminals.setObjectName(u"lineEdit_non_terminals")

        self.verticalLayout.addWidget(self.lineEdit_non_terminals)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_6)

        self.label_7 = QLabel(self.groupBox_2)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.label_7)

        self.comboBox_stars_sym = QComboBox(self.groupBox_2)
        self.comboBox_stars_sym.setObjectName(u"comboBox_stars_sym")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.comboBox_stars_sym)

        self.lineEdit_lambda = QLineEdit(self.groupBox_2)
        self.lineEdit_lambda.setObjectName(u"lineEdit_lambda")
        self.lineEdit_lambda.setMaxLength(1)
        self.lineEdit_lambda.setAlignment(Qt.AlignCenter)

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.lineEdit_lambda)


        self.verticalLayout.addLayout(self.formLayout_2)

        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.rules = QPlainTextEdit(self.groupBox_2)
        self.rules.setObjectName(u"rules")

        self.verticalLayout.addWidget(self.rules)

        self.pushButton_claculate = QPushButton(self.groupBox_2)
        self.pushButton_claculate.setObjectName(u"pushButton_claculate")

        self.verticalLayout.addWidget(self.pushButton_claculate)

        self.line = QFrame(self.groupBox_2)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.label_10 = QLabel(self.groupBox_2)
        self.label_10.setObjectName(u"label_10")

        self.verticalLayout.addWidget(self.label_10)

        self.plainTextEdit_canon_grammar = QPlainTextEdit(self.groupBox_2)
        self.plainTextEdit_canon_grammar.setObjectName(u"plainTextEdit_canon_grammar")
        self.plainTextEdit_canon_grammar.setReadOnly(True)

        self.verticalLayout.addWidget(self.plainTextEdit_canon_grammar)


        self.gridLayout_2.addWidget(self.groupBox_2, 0, 0, 2, 1)

        course.setCentralWidget(self.centrwidget)
        self.menubar = QMenuBar(course)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 783, 21))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.menu_2 = QMenu(self.menubar)
        self.menu_2.setObjectName(u"menu_2")
        course.setMenuBar(self.menubar)
        QWidget.setTabOrder(self.lineEdit_terminals, self.lineEdit_non_terminals)
        QWidget.setTabOrder(self.lineEdit_non_terminals, self.comboBox_stars_sym)
        QWidget.setTabOrder(self.comboBox_stars_sym, self.rules)
        QWidget.setTabOrder(self.rules, self.spinBox_min_len)
        QWidget.setTabOrder(self.spinBox_min_len, self.spinBox_max_len)

        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menu.addAction(self.action_open)
        self.menu.addAction(self.action_save)
        self.menu_2.addAction(self.action_help)
        self.menu_2.addAction(self.action_author)
        self.menu_2.addAction(self.action_theme)

        self.retranslateUi(course)

        QMetaObject.connectSlotsByName(course)
    # setupUi

    def retranslateUi(self, course):
        course.setWindowTitle(QCoreApplication.translate("course", u"course", None))
        self.action_open.setText(QCoreApplication.translate("course", u"\u041e\u0442\u043a\u0440\u044b\u0442\u044c", None))
        self.action_help.setText(QCoreApplication.translate("course", u"\u0421\u043f\u0440\u0430\u0432\u043a\u0430", None))
        self.action_author.setText(QCoreApplication.translate("course", u"\u0410\u0432\u0442\u043e\u0440", None))
        self.action_theme.setText(QCoreApplication.translate("course", u"\u0422\u0435\u043c\u0430", None))
        self.action_save.setText(QCoreApplication.translate("course", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("course", u"\u0426\u0435\u043f\u043e\u0447\u043a\u0438", None))
        self.groupBox.setTitle(QCoreApplication.translate("course", u"\u0414\u0438\u0430\u043f\u0430\u0437\u043e\u043d \u0434\u043b\u0438\u043d \u0446\u0435\u043f\u043e\u0447\u0435\u043a", None))
        self.label_4.setText(QCoreApplication.translate("course", u"\u0414\u043e", None))
        self.label_3.setText(QCoreApplication.translate("course", u"\u041e\u0442", None))
        self.label_8.setText(QCoreApplication.translate("course", u"\u0426\u0435\u043f\u043e\u0447\u043a\u0438, \u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0435 \u0438\u0437 \u043d\u0435\u043a\u0430\u043d\u043e\u043d\u0438\u0447\u0435\u0441\u043a\u043e\u0439 \u0433\u0440\u0430\u043c\u043c\u0430\u0442\u0438\u043a\u0438", None))
        self.label_9.setText(QCoreApplication.translate("course", u"\u0426\u0435\u043f\u043e\u0447\u043a\u0438, \u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0435 \u0438\u0437 \u043a\u0430\u043d\u043e\u043d\u0438\u0447\u0435\u0441\u043a\u043e\u0439 \u0433\u0440\u0430\u043c\u043c\u0430\u0442\u0438\u043a\u0438", None))
        self.pushButton_compare_chains.setText(QCoreApplication.translate("course", u"\u0421\u0440\u0430\u0432\u043d\u0438\u0442\u044c \u0446\u0435\u043f\u043e\u0447\u043a\u0438", None))
        self.label_status.setText(QCoreApplication.translate("course", u"\u0421\u0442\u0430\u0442\u0443\u0441:", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("course", u"\u0421\u043e\u0431\u044b\u0442\u0438\u044f", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("course", u"\u0413\u0440\u0430\u043c\u043c\u0430\u0442\u0438\u043a\u0430", None))
        self.label_2.setText(QCoreApplication.translate("course", u"\u0422\u0435\u0440\u043c\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0441\u0438\u043c\u0432\u043e\u043b\u044b", None))
        self.lineEdit_terminals.setText(QCoreApplication.translate("course", u"a, b, c", None))
        self.label_5.setText(QCoreApplication.translate("course", u"\u041d\u0435\u0442\u0435\u0440\u043c\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0441\u0438\u043c\u0432\u043e\u043b\u044b", None))
        self.lineEdit_non_terminals.setText(QCoreApplication.translate("course", u"S, A", None))
        self.label_6.setText(QCoreApplication.translate("course", u"\u0421\u0442\u0430\u0440\u0442\u043e\u0432\u044b\u0439 \u0441\u0438\u043c\u0432\u043e\u043b", None))
        self.label_7.setText(QCoreApplication.translate("course", u"\u0421\u0438\u043c\u0432\u043e\u043b \u043b\u044f\u043c\u0431\u0434\u044b", None))
        self.lineEdit_lambda.setText(QCoreApplication.translate("course", u"@", None))
        self.label.setText(QCoreApplication.translate("course", u"\u041f\u0440\u0430\u0432\u0438\u043b\u0430", None))
        self.rules.setPlainText(QCoreApplication.translate("course", u"S -> aaS | Ab\n"
"A -> cAbb | @", None))
        self.pushButton_claculate.setText(QCoreApplication.translate("course", u"\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0440\u0430\u0441\u0441\u0447\u0451\u0442\u044b", None))
        self.label_10.setText(QCoreApplication.translate("course", u"\u0413\u0440\u0430\u043c\u043c\u0430\u0442\u0438\u043a\u0430 \u0432 \u043a\u043e\u043d\u043e\u043d\u0438\u0447\u0435\u0441\u043a\u043e\u043c \u0432\u0438\u0434\u0435", None))
        self.menu.setTitle(QCoreApplication.translate("course", u"\u0424\u0430\u0439\u043b", None))
        self.menu_2.setTitle(QCoreApplication.translate("course", u"\u041e \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0435", None))
    # retranslateUi

