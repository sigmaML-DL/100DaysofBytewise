
number1 = float(input("Enter the first number: "))
number2 = float(input("Enter the second number: "))

print("Choose an operation:")
print("1. Addition (+)")
print("2. Subtraction (-)")
print("3. Multiplication (*)")
print("4. Division (/)")

operation = input("Enter the operation (+, -, *, /): ")

if operation == '+':
    result = number1 + number2
    print(f"The result of {number1} + {number2} is: {result}")
elif operation == '-':
    result = number1 - number2
    print(f"The result of {number1} - {number2} is: {result}")
elif operation == '*':
    result = number1 * number2
    print(f"The result of {number1} * {number2} is: {result}")
elif operation == '/':
    # Handle division by zero
    if number2 == 0:
        print("Error: Division by zero is not allowed.")
    else:
        result = number1 / number2
        print(f"The result of {number1} / {number2} is: {result}")
else:
    print("Invalid operation. Please select +, -, *, or /.")
