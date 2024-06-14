
n = int(input("How many terms? "))
first_term = 0
second_term = 1

if n >= 1:
    print("Fibonacci sequence:")
    print(first_term, end=" ")  
if n >= 2:
    print(second_term, end=" ") 


for _ in range(2, n):
    next_term = first_term + second_term
    print(next_term, end=" ")
    first_term = second_term
    second_term = next_term

