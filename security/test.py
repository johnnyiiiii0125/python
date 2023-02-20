
z = 'dd2940c04462b4dd7c450528835cca15'
x = list(z)
x[2] = chr((ord(x[2]) + ord(x[3])) - 50)
x[4] = chr((ord(x[2]) + ord(x[5])) - 48)
x[30] = chr((ord(x[31]) + ord(x[9])) - 48)
x[14] = chr((ord(x[27]) + ord(x[28])) - 97)
for i in range(0, 16):
    a = x[31 - i]
    x[31 - i] = x[i]
    x[i] = a
x_str = ''.join(i for i in x)
print(x_str)
