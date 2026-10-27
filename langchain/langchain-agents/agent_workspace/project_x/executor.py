def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        print("Error: Division by zero")
        return -1
    return a // b

def main():
    print("Enter two numbers and an operation (+, -, *, /): ", end="")
    data = input().split()
    if len(data) < 3:
        print("Invalid input")
        return

    try:
        a = int(data[0])
        op = data[1]
        b = int(data[2])
    except ValueError:
        print("Invalid input")
        return

    if op == '+':
        result = add(a, b)
    elif op == '-':
        result = subtract(a, b)
    elif op == '*':
        result = multiply(a, b)
    elif op == '/':
        result = divide(a, b)
    else:
        print("Invalid operation")
        return

    print(f"Result: {result}")

if __name__ == "__main__":
    main()