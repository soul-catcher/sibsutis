#inc
int main() {
    cout << "\nTFracEditor:" << endl;
    TFracEditor fracEditor;
    cout << "init: " << fracEditor.getFraction() << endl;
    fracEditor.editFraction(Operations::ADD_DIGIT);
    cout << "Add Number: " << fracEditor.getFraction() << endl;
    fracEditor.editFraction(Operations::ADD_DIVIDER);
    cout << "Add divider: " << fracEditor.getFraction() << endl;
    fracEditor.editFraction(Operations::ADD_SIGN);
    cout << "Add sign: " << fracEditor.getFraction() << endl;
    fracEditor.editFraction(Operations::ADD_DIGIT);
    cout << "Add Number: " << fracEditor.getFraction() << endl;
    fracEditor.editFraction(Operations::REMOVE_LAST_DIGIT);
    cout << "Remove digit: " << fracEditor.getFraction() << endl;

    cout << "\nTPNumEditor:" << endl;
    TPNumEditor pNumEditor;
    cout << "init: " << pNumEditor.getNumberString() << endl;
    pNumEditor.menu(_addDigit);
    cout << "add digit: " << pNumEditor.getNumberString() << endl; // int 0-15
    pNumEditor.menu(_addDigit);
    cout << "add digit: " << pNumEditor.getNumberString() << endl; // int 0-15
    pNumEditor.menu(_backspace);
    cout << "backspace: " << pNumEditor.getNumberString() << endl;
    pNumEditor.menu(_editNumber);
    cout << "editNumber: " << pNumEditor.getNumberString() << endl; // 10 or greater than 10
}