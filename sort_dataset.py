import os

directory = "C:/Users/ganes/Pictures/facerec_pics/aryan/"
i = 1;
for filename in os.listdir(directory):
    os.rename(directory+filename,directory+str(i)+".jpg")
    i += 1
