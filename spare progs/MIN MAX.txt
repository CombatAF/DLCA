import random

def guess_the_number():
    number_to_guess = random.randint(1, 100)
    attempts = 0
    max_attempts = 5

    print("Welcome to Guess the Number!")
    print("I have selected a number between 1 and 100. Can you guess it?")

    while attempts < max_attempts:
        guess = int(input("Enter your guess:"))

        if guess < number_to_guess:
            print("Too low! Try again.")
        elif guess > number_to_guess:
            print("Too high! Try again.")
        else:
            print(f"Congratulations! You guessed the number ({number_to_guess}) correctly!")
            return

        attempts += 1

    print(f"Sorry, you've run out of attempts. The number was {number_to_guess}.")

guess_the_number()