x = {1:1, 5:5}
if (x.__contains__(5)):
    x[5] += 1
else:
    x[5] = 5
print(x)