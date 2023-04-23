import string

FILE_TO_OPEN = 'new.txt'
FILE_TO_WRITE = 'hope.txt'
ALPHABET_LOWER_BOUND = 0
ALPHABET_UPPER_BOUND = ord('~') + 1

def encryptCaesarCipher(plain_text, shift = 2):
    # Remove redundant shift amount
    shift = shift % ALPHABET_UPPER_BOUND

    # Creating an alphabet for all ASCII characters
    alphabet = ''.join(chr(i) for i in range(ALPHABET_UPPER_BOUND))

    # Create a shifted version of that alphabet
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]

    # Create a dictionary where keys are ASCII values and the values are shifted ASCII values
    table = str.maketrans(alphabet, shifted_alphabet)

    # Use table to map each character in the plaintext to its shifted correspondence
    return plain_text.translate(table)


def decryptCaesarCipher(cipher_text, shift = 2):
    # Encrypt with inverted direction
    return encryptCaesarCipher(cipher_text, -shift)


if __name__ == "__main__":
    operation = input("Encrypt or Decrypt [e] or [d]: ")
    shift = int(input("Input Shift Value: "))

    with open(FILE_TO_OPEN, "r") as f:
        print(f"Reading from {FILE_TO_OPEN}")
        input_text = f.read()

        if operation == 'e':
            result = encryptCaesarCipher(input_text, shift)
        elif operation == 'd':
            result = decryptCaesarCipher(input_text, shift)

        print(f"Writing to {FILE_TO_WRITE}")
        output_file = open(FILE_TO_WRITE, "w")
        output_file.write(result)
        output_file.close()
        f.close()
