import random # Random library for random printing number 

print("Welcome to the Number Guessing Game!")
    
while True:
    n = random.randint(1, 100)
    attempts = 0
        
    while True:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            if guess < n:
                print("Too low! Try again.")
            elif guess > n:
                print("Too high! Try again.")
            else:
                print("Congratulations! You've guessed the right number.")
                break
        except ValueError:
                print("Invalid input. Please enter a valid number.")


 
