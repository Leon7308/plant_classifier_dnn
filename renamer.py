import os
x = 1
while x!=0:
    x = input('Enter file name \n')

    os.chdir('C:\\Users\\archi\\plant_classifier\\Dataset\\Validation\\'+ str(x))
    i=1
    for file in os.listdir():
        src=file
        dst=str(i)+".jpg"
        os.rename(src,dst)
        i+=1