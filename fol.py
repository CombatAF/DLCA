
Colors = ['Red', 'Green', 'Blue']


def Likes(x, y):
    return (x == 'Alice' and y == 'Green') or \
           (x == 'Bob' and y == 'Red') or \
           (x == 'Charlie' and y == 'Blue')

def FavoriteColor(x, c):
    return Likes(x, c)

person = input("Enter a person's name (Alice, Bob, Charlie): ")
color = input("Enter a color (Red, Green, Blue): ")


print(f"Does {person} like the color {color}?", Likes(person, color))


favorite_color = next((c for c in Colors if FavoriteColor(person, c)), None)
print(f"What is {person}'s favorite color?", favorite_color)
