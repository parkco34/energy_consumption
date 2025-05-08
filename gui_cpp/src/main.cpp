#include <QApplication> // QT application framework
#include <QFileDialog> // QT File dialog
#include <QWidget> // Basic GUI widget
#include <iostream>

int main(int argc, char *argv[])
{
    // Instantiate QT application, creating UI event loop
    QApplication app(argc, argv);

    // Create widget (parent for dialogs)
    QWidget window;

    QString file_name = QFileDialog::getOpenFileName(
            &window,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
            );

    // User's choice
    if (!file_name.isEmpty())
    {
        std::cout << "Selected file: " << file_name.toStdString() << std::endl;
    } else 
    {
        std::cout << "No file selected." << std::endl;
    }
    return 0;
}

