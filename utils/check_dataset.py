import os

orig = len(os.listdir("../dataset/original"))
fake = len(os.listdir("../dataset/fake"))

print("Original QRs:", orig)
print("Fake QRs:", fake)
