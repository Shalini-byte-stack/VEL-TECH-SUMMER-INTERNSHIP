def sort_numbers(numbers):
    even = []
    odd = []

    for num in numbers:
        if num % 2 == 0:
            even.append(num)
        else:
            odd.append(num)

    return even, odd

numbers = [12, 7, 5, 18, 22, 9, 14, 3, 8, 11]

even_numbers, odd_numbers = sort_numbers(numbers)

print("Even Numbers:", even_numbers)
print("Odd Numbers:", odd_numbers)
