# Reversed Sentence 
words=input("Enter your Sentence : ")
words = words.split() # split the sentence into words 
r_w= words[::-1] # Slicing process 
reversed_sentence = ' '.join(r_w) # Joining the reversed words into sentence

print("Reversed Sentence:", reversed_sentence)


# Palindrome Sentence 
import string

def p_check(s):
    cle = ''.join(char.lower() for char in s if char.isalnum())
    return cle == cle[::-1]

inp= "A man, a plan, a canal, Panama"
if p_check(inp):
    print(f'"{inp}" is a palindrome.')
else:
    print(f'"{inp}" is not a palindrome.')
    


    

