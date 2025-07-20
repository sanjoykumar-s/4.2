import random

def is_prime(n, k=5):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Step 1: Write n-1 as 2^b * m
    m = n - 1
    b = 0
    while m % 2 == 0:
        m //= 2
        b += 1

    # Step 2: Repeat k times (number of random tests)
    for _ in range(k):
        a = random.randint(2, n - 2)
        z = pow(a, m, n)  # z = a^m mod n
        if z == 1 or z == n - 1:
            continue

        for _ in range(b - 1):
            z = pow(z, 2, n)
            if z == n - 1:
                break
        else:
            return False  # Composite
    return True  # Probably prime


num = int(input("Enter a number to test for primality: "))
if is_prime(num):
    print(f"{num} is probably a prime number.")
else:
    print(f"{num} is not a prime number.")