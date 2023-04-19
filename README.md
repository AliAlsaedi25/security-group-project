# security-group-project

you will have to install torch on you computer 'pip install torch' 

to trian this model and see the same level of testing we had you will need access to a A100 GPU or better if not the code will take 2-3 days to finish runing on a local CPU

you need to choose a data set and encrypt it 

for us input_to_encrypt holds the orginal data 
put_here holds the the encrypted data set 
new_text holds the data generaated by the model 
hope hold the decrypted data the model created 

after encrpyting you data make sure you change the name of the file in the bigram file and than you can start running the code 

after training for 5000 (max_iters) steps it will generate a new file that has encrypted output 

take that file aand decrpt it with the same key you encrypted it with 

you will see the model was able to mimic the data in which it was trained on 

