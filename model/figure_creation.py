import matplotlib.pyplot as plt


def save_first_9(train, class_names):
    """Creates image of first 9 images input into the model training set

    Variable(s):
        train: Training data set to gather images
        class_names: Possible classifications
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.savefig("visuals/First 9 Images.png")


def save_training_results(history, epochs):
    """Creates image of training graph history through epochs

        Variable(s):
            history: Training history and accuracy values
            epochs: Number of total runs throughout model compilation
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("visuals/Training and Validation Graphs.png")


def save_augmented_images(train, a_images):
    """Creates image of first 9 augmented images input into the model training set

        Variable(s):
            train: Training data set to gather images
            a_images: List of augmented images
    """
    plt.figure(figsize=(10, 10))
    for images, _ in train.take(1):
        for i in range(9):
            augmented_images = a_images(images)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
            plt.savefig("visuals/Augmented Images.png")
