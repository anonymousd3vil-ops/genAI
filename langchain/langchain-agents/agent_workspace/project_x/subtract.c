#include <stdio.h>

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
        printf("Error: Division by zero\n");
        return -1;
    }
    return a / b;
}

int main() {
    int a, b;
    char op;
    printf("Enter two numbers and an operation (+, -, *, /): ");
    scanf("%d %c %d", &a, &op, &b);
    switch(op) {
        case '+': printf("Result: %d\n", add(a, b)); break;
        case '-': printf("Result: %d\n", subtract(a, b)); break;
        case '*': printf("Result: %d\n", multiply(a, b)); break;
        case '/': printf("Result: %d\n", divide(a, b)); break;
        default: printf("Invalid operation\n");
    }
    return 0;
}