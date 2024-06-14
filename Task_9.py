
user_string = input("Enter a string: ")
user_string = user_string.replace(" ", "").lower()


reversed_string = user_string[::-1] # Slicing operation 


if user_string == reversed_string:
    print(f"'{user_string}' is a palindrome.")
else:
    print(f"'{user_string}' is not a palindrome.")
