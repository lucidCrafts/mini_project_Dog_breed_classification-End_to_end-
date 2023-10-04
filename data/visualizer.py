import matplotlib.pyplot as plt

def display_sample_images(train_data, ds_info, num_images=9):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(train_data.take(num_images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(ds_info.features["label"].int2str(label))
        plt.axis("off")
    plt.show()
