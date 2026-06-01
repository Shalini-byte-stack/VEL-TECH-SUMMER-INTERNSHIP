birth_year = int(input("Enter your birth year: "))

if birth_year > 2026:
    print("Error: Invalid birth year")
else:
    age = 2026 - birth_year
    age_after_10_years = age + 10

    print(f"Your age is {age} years.")
    print(f"Your age after 10 years will be {age_after_10_years} years.")