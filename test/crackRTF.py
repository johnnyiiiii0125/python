import hashlib
String1 = "6E32D0943418C2C33385BC35A1470250DD8923A9"
suffix = "@DBApp"
for i in range(100000, 1000000):
    dest = str(i) + suffix
    hashed = hashlib.sha1(dest.encode("utf-8")).hexdigest()
    if hashed.lower() == String1.lower():
        print(dest)
# dest: 123321@DBApp
    