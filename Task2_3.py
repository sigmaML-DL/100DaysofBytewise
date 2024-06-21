def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def celsius_to_kelvin(c):
    return c + 273.15

def kelvin_to_celsius(k):
    return k - 273.15

def fahrenheit_to_kelvin(f):
    return (f - 32) * 5/9 + 273.15

def kelvin_to_fahrenheit(k):
    return (k - 273.15) * 9/5 + 32

def display_menu():
    print("Temperature Conversion Menu:")
    print("1. Celsius to Fahrenheit")
    print("2. Fahrenheit to Celsius")
    print("3. Celsius to Kelvin")
    print("4. Kelvin to Celsius")
    print("5. Fahrenheit to Kelvin")
    print("6. Kelvin to Fahrenheit")
    print("7. Exit")

def temperature_conversion():
    while True:
        display_menu()
        choice = input("Please select a conversion option (1-7): ")
        
        if choice == '7':
            print("Thank you for using the temperature conversion program. Goodbye!")
            break
        
        try:
            if choice in {'1', '2', '3', '4', '5', '6'}:
                temp = float(input("Enter the temperature to convert: "))
                
                if choice == '1':
                    result = celsius_to_fahrenheit(temp)
                    print(f"{temp}°C is equal to {result}°F")
                elif choice == '2':
                    result = fahrenheit_to_celsius(temp)
                    print(f"{temp}°F is equal to {result}°C")
                elif choice == '3':
                    result = celsius_to_kelvin(temp)
                    print(f"{temp}°C is equal to {result}K")
                elif choice == '4':
                    result = kelvin_to_celsius(temp)
                    print(f"{temp}K is equal to {result}°C")
                elif choice == '5':
                    result = fahrenheit_to_kelvin(temp)
                    print(f"{temp}°F is equal to {result}K")
                elif choice == '6':
                    result = kelvin_to_fahrenheit(temp)
                    print(f"{temp}K is equal to {result}°F")
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a valid temperature.")

temperature_conversion()
