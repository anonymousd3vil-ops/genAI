#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    if (b == 0) {
        cout << "Error: Division by zero\n";
        return -1;
    }
    return a / b;
}

int main() {
    int a, b;
    char op;
    cout << "Enter two numbers and an operation (+, -, *, /): ";
    cin >> a >> op >> b;
    switch(op) {
        case '+': cout << "Result: " << add(a, b) << endl; break;
        case '-': cout << "Result: " << subtract(a, b) << endl; break;
        case '*': cout << "Result: " << multiply(a, b) << endl; break;
        case '/': cout << "Result: " << divide(a, b) << endl; break;
        default: cout << "Invalid operation\n";
    }
    return 0;
}