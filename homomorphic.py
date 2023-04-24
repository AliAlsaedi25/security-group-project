"""
Partially-Homomorphic Encryption

This is an implementation of a Paillier cryptosystem. It is a type of partially homomorphic encryption which supports:
- Ciphertext Addition
- Ciphertext Multiplication by a Plaintext number

Ref: https://en.wikipedia.org/wiki/Paillier_cryptosystem
"""

import math
import random

# Choose two large distinct prime numbers p and q
p = 19
q = 7

n = p * q
# e.g. n = 19 * 7 = 133, closest value to 127 (ASCII limit) using prime factors

# Euler's Totient Function: Counts the number of positive primes up to n
# O(1) formula is given by XXXX, which states that if n is the product of two primes
# p and q, then its totient is equal to the product (p-1)*(q-1)
phi = (p-1)*(q-1)

def lx(x):
    """
    """
    y = (x-1) / n
    return int(y)

g = 1 + n
lmbda = phi * 1
mu = pow(phi, -1, n)
print(f"Private key (lambda): {lmbda}")
print(f"Public key: g={g}, n={n}, mu={mu}")

def encrypt(m,r):
    """
    """
    assert math.gcd(r,n*n) == 1
    c = pow(g,m,n*n)*pow(r,n,n*n) % (n*n)
    return c

def decrypt(c):
    """
    """
    p = (lx(pow(c, lmbda, n*n))*mu) % n
    return p

with open('./bigram-language-model/gpt_at_home/input.txt', 'r', encoding="utf-8") as input_file:
    print(f"Reading from {input_file.name}")
    text = input_file.read()
    num_text = [ ord(c) for c in text ]

r = 66
cipher_text = [ encrypt(x, r) for x in num_text ]
cipher_string = ''.join([ chr(c) for c in cipher_text])
print(cipher_text)

with open('cipher.txt', 'w', encoding="utf-8") as output_file:
    print(f"Writing to {output_file.name}")
    output_file.write(cipher_string)

with open('cipher.txt', 'r', encoding="utf-8") as input_file:
    print(f"Reading from {input_file.name}")
    text = input_file.read()
    num_text = [ ord(c) for c in text ]

original_text = [ decrypt(x) for x in num_text ]
original_text = [ chr(c) for c in original_text ]

with open('decrypt.txt', 'w', encoding="utf-8") as output_file:
    print(f"Decrypted text written to {output_file.name}")
    output_file.write(original_text)
