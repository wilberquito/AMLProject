import matplotlib.pyplot as plt

def show_dataset(dataloader):
    features, labels = next(iter(dataloader))

    for i in range (0,10):
        img = features[i].squeeze().permute(1,2,0)
        label = labels[i]
        plt.imshow(img)
        plt.show()
        print(f'Label: {label}')
        print("hola")