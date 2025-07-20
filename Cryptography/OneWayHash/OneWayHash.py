import hashlib

def getMD5Hash(data):
    hash = hashlib.md5(data.encode())
    return hash.hexdigest()

def getSHA256Hash(data):
    hash = hashlib.sha256(data.encode())
    return hash.hexdigest()

# Take message from user
msg = input("Enter text to hash: ")

# Ask user to choose the hashing method
print("Choose hashing method:")
print("1. MD5")
print("2. SHA-256")
choice = input("Enter 1 or 2: ")

# Compute hash based on user choice
if choice == '1':
    hash_val = getMD5Hash(msg)
    print("MD5 Hash: " + hash_val)
elif choice == '2':
    hash_val = getSHA256Hash(msg)
    print("SHA-256 Hash: " + hash_val)