import base64

str2 = "e3nifIH9b_C@n@dH"
dest = ""
for i in range(len(str2)):
    dest += chr(ord(str2[i])-i)
flag = base64.b64decode(dest)
print(flag)
