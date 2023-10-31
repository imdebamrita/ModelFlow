import math


a = float(input("Enter the length of 1st side: "))
b = float(input("Enter the length of 2nd side: "))
c = float(input("Enter the length of 3rd side: "))

s = (a + b + c) / 2.0

x = s * (s - a) * (s - b) * (s - c)

area = math.sqrt(x)

print("The area of the triangle is " + str(area))