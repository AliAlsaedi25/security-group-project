### security-group-project
# Protecting integrity in L.L.M.

### By - Ali, Victor, Calvin, Ian, Aaron 

## Overview
- This repository holds our language model which is trained to mimic and produce text based on the given input. The output is meant to be both syntactically and idiolectically.

## Initial Set Up
1. Make sure you have torch installed on your machine
   - 'pip install torch'
2. In order to train this model and see the same level of testing we had you will need access to a A100 GPU or better on you machine. If not the model will take 2-3 days to finish training and running on your local CPU.

## How To Run
1. Choose a data set and encrypt it
  - for us input_to_encrypt.txt holds the orginal data 
  - put_here.txt holds the the encrypted data set 
  - new.txt holds the data generated by the model 
  - hope.txt holds the decrypted data the model created 
2. After encrpyting your data, make sure you change the name of the file in the bigram file and then you can start running the code 
3. After training for 5000 (max_iters) steps it will generate a new file that has encrypted output 
4. Take that file and decrypt it with the same key you encrypted it with 
5. You will see the model was able to mimic the data in which it was trained on 

## Other Files
- homomorphic.py is a file which contains code of a Paillier Cryptosystem which encrypts and decrypts data homomorphically
  - The full implementation of this encryption scheme was out of scope for this project and altered decrypted values when trying to restrict the bounds of encrypted cipher values
- cipher.txt & decrypt.txt are files which show where and how we missed the mark on trying to implement homorphic encryption within this project. 
- shift_cipher.py is the Shift Cipher encryption used for this project and as well as our language model. It is used during decryption as well.

## External Resources 
- Attention is All You Need paper: 
   https://arxiv.org/abs/1706.03762
- OpenAI GPT-3 paper: 
   https://arxiv.org/abs/2005.14165
- Torch Documentation give that a read to undersand what I have done here: 
   https://pytorch.org/docs/stable/index.html
