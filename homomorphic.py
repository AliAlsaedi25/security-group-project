import math
import random

p = 19
q = 7

n = p * q
# n = 133, closest value to 127(ascii limit) using prime factors

phi = (p-1)*(q-1)
print(f"phi = {phi}")


def lx(x):
    y = (x-1) / n
    return int(y)

g = 1 + n
lmbda = phi * 1
mu = pow(phi, -1, n)

print(f"private key is lmbda={lmbda}")
print(f"public key = g={g}, n={n}, mu={mu}")

def encrypt(m,r):
    assert math.gcd(r,n*n) == 1
    c = ( pow(g,m,n*n)*pow(r,n,n*n) % (n*n))
    return c

def decrypt(c):
    p = (lx(pow(c, lmbda, n*n))*mu) % n
    return p

convert = list()
def stabalize_ascii(ascii_num):
    div, mod = divmod(ascii_num, 126)
    if mod < 32:
        mod+=32
        convert.append([div*126, 32])
    else:
        convert.append([div*126, 0])
    return mod

def unstabalize_ascii(ascii_num, index):
    ascii_num = ascii_num - convert[index][1] + convert[index][0]
    return ascii_num


def alpha_to_num_list(text):
    #text = text.upper()
    num_list = []
    i = 0
    while i < len(text):
        num_list.append(ord(text[i]))
        i = i+1
    return num_list

def num_list_to_text(num_list):
    text = ''
    i = 0
    while i < len(num_list):
        text = text + chr(num_list[i])
        i = i + 1
    return text

def modInverse(A, M):
    for X in range(1, M):
        if (((A % M) * (X % M)) % M == 1):
            return X
    return -1

# encrypt(ord(h) + 3, r1*r2) === (cipher(ord(h)) * cipher(3)) % (n*n)
with open('./bigram-language-model/gpt_at_home/input.txt', 'r', encoding="utf-8") as f_in:
    # Open the output file in write mode
    text = f_in.read()
    num_text = alpha_to_num_list(text)
print(f"{f_in.name} has been opened and read")
cipher_text = list()
i = 0
r = 66
while i < len(num_text):
    c = encrypt(num_text[i], r)
    cipher_text.append(c)
    i+=1
cipher_string = num_list_to_text(cipher_text)
with open('cipher.txt', 'w', encoding="utf-8") as f_out:
    # Read and write each line in the input file
    for num in cipher_string:
        f_out.write(num)
print(f"Cipher text has been written to {f_out.name}")

with open('cipher.txt', 'r', encoding="utf-8") as f_in2:
    text = f_in2.read()
    num_text = alpha_to_num_list(text)
print(f"{f_in2.name} has been opened and read")

original_text = list()
i = 0
while i < len(num_text):
    p = decrypt(num_text[i])
    original_text.append(c)
    i+=1
original_text = num_list_to_text(original_text)

with open('decrypt.txt', 'w', encoding="utf-8") as f_out2:
    # Read and write each line in the input file
    for num in original_text:
        f_out2.write(num)
print(f"Decrypted text has been written to {f_out2.name}")

#alpha_list = alpha_to_num_list('hello')
#print(f"hello as number list with ascii integer values = {alpha_list}")
#cipher_list = list()
#i = 0
#r1 = 66
#print(f"r1 = {r1}")
#while i < len(alpha_list):
#    c = encrypt(alpha_list[i], r1)
    #c = stabalize_ascii(c)
#    cipher_list.append(c)
#    i+=1
#print(f"cipher values of hello as number list = {cipher_list}")
#cipher_string = num_list_to_text(cipher_list)
#print(f"cipher hello as string = {cipher_string}")
#i = 0
#test_list = alpha_to_num_list(' test')
#print(f"test(with preceding space) as number list with ascii integer values = {test_list}")
#test_cipher_list = list()
#i = 0
#r2 = 92
#print(f"r2 = {r2}")
#while i < len(test_list):
#    c = encrypt(test_list[i], r2)
    #c = stabalize_ascii(c)
#    test_cipher_list.append(c)
#    i+=1
#print(f"test encrypted = {test_cipher_list}")
#i = 0
#r3 = 102
#print(f"r3 = {r3}")
#encrypt_3 = encrypt(3, r3)
#encrypt_3 = stabalize_ascii(encrypt_3)
#print(f"integer 3 cipher value = {encrypt_3}")
#while i < len(cipher_list):
#    cipher_list[i] = (cipher_list[i] * encrypt_3) % (n*n)
    #cipher_list[i] = stabalize_ascii(cipher_list[i])
#    i+=1
#print(f"cipher values of hello as number list modified with +3 cipher value = {cipher_list}")
#cipher_string = num_list_to_text(cipher_list)
#print(f"cipher hello +3 as string = {cipher_string}")
#decrypt_list = list()
#i = 0
#while i < len(cipher_list):
    #d1 = unstabalize_ascii(cipher_list[i], i)
    #d1 = unstabalize_ascii(d1, i)
    #d1 = unstabalize_ascii(d1, i)
    #d = decrypt(d1)
#    d= decrypt(cipher_list[i])
    #d = unstabalize_ascii(d, i)
    #d = stabalize_ascii(d)
#    decrypt_list.append(d)
#    i+=1
#print(f"decrypted list with hello shifted 3 spaces (ascii values here) = {decrypt_list}")
#i = 0
#while i < len(test_cipher_list):
    #d1 = unstabalize_ascii(test_cipher_list[i], i)
    #d = decrypt(d1)
#    d= decrypt(test_cipher_list[i])
# #   decrypt_list.append(d)
#    i+=1
#print(f"decrypted list with hello shifted 3 spaces with the addition of the decrypted value of 'test' added (ascii values here) = {decrypt_list}")
#decrypt_string = num_list_to_text(decrypt_list)
#print(f"decrypt list changed to a string (ascii values -> characters) = {decrypt_string}")
# expecting KHOOR TEST
# for some reason * === +
