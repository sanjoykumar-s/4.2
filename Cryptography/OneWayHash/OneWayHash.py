import hashlib

def getMD5Hash(data):
    hash = hashlib.md5(data.encode())
    return hash.hexdigest()

def getSHA256Hash(data):
    hash = hashlib.sha256(data.encode())
    return hash.hexdigest()

def getHash(data, choice):
    if choice == '1':
        return "MD5 hash : " + getMD5Hash(msg)
    else if choice == '2':
        return "SHA-256 hash: " + getSHA256Hash(msg)

# Take message from user
msg = input("Enter text to hash: ")

# Ask user to choose the hashing method
print("Choose hashing method:")
print("1. MD5")
print("2. SHA-256")
choice = input("Enter 1 or 2: ")


print(getHash(msg, choice))