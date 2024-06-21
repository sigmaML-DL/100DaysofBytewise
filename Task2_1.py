# 1. Palindrome Checker (Word)
user=input("WORD: ")
reversed=user[::-1]
print("The answer is = ",reversed)
if user==reversed:
    print("Word is a Palindrome.")
elif user != reversed:
    print("Word is not a Palindrome.")

# 2. FizzBuzz

for n in range(1, 101):
    
    if n % 3 == 0 and n % 5 == 0:
        print("FizzBuzz")
    elif n % 3 == 0:
        print("Fizz")
    elif n % 5 == 0:
        print("Buzz")
    else:
        print(n)

# 3. Nth Fibonacci Number

n=int(input("Entre your Desired Number : "))
def f(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    
    for _ in range(2, n + 1):
        a,b=b,a+b
        
    return b

  
print(f"The {n}th Fibonacci number is: {f(n)}")




