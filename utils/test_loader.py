from dataset_loader import QRDataset

dataset = QRDataset("../dataset")
print("Total samples:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)
