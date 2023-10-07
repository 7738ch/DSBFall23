# question 1
def giveVowels(x):
    if isinstance(x, str):
        vowels = ["a", "e", "i", "o", "u"]
        out = 0
        lst = list(x)
        for i in lst:
            if i in vowels:
                out += 1
    return(out)

#question 2
animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']

for item in animals:
    smallItem = list(item)
    out = list()
    for letter in smallItem:
        out += letter.capitalize()
    realOut = ''.join(out)
    print(realOut)

#question 3
for number in range(1,16):
    if number%2 == 0:
        print(number,"is even")
    else:
        print(number, "is odd")

#question 4
def summ():
    val1 = int(input("Enter your first integer: "))
    val2 = int(input("Enter your second integer: "))
    su = val1 + val2
    return(su)








