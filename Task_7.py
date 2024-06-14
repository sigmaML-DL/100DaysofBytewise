
user_string = input("Enter a string: ")
vowels = "aeiouAEIOU"

vowel_count = 0
found_vowels=[]


for char in user_string:
    # Check if the character is a vowel
    if char in vowels:
        vowel_count += 1
        found_vowels=found_vowels.append(char)


print("The number of vowels in the string is:", vowel_count)
print("The vowels in the string are:", found_vowels)
