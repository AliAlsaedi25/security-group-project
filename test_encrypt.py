import string

fileToOpen = 'new.txt' #input("What file would you like to open: ")
fileToWrite = 'hope.txt' #input("What file would you like to write to: ")

inputFile = open(fileToOpen, "r")
plain_text = inputFile.read()
altered_text = open(fileToWrite, "w")


operation = input("Encrypt or Decrypt [e] or [d]: ")
shift_val = int(input("Input Shift Value: "))




def CC(text, shift = 2):

    shift = shift % 128

    #creating an alphabet for all ascii characters
    alphabet = ''.join(chr(i) for i in range(128))
    
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    altered = text.translate(table)

    altered_text.write(altered)



def DCC(text, shift = 2):
    shift = (shift * -1) % 128
    CC(text, shift)


if operation == 'e':
    CC(plain_text, shift_val)
elif operation == 'd':
    DCC(plain_text, shift_val)



inputFile.close()