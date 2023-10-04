from pathlib import Path
from data.loader import load_data
from data.visualizer import display_sample_images
from data.preprocessors import resize_image, augment_data
from models.custom_models import create_model
import tensorflow_datasets as tfds


def main():
    
    train_data, test_data, ds_info = load_data()
    display_sample_images(train_data, ds_info)

    train_data = train_data.map(resize_image)
    test_data = test_data.map(resize_image)

    data_augmentation = augment_data()
    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))
    test_data = test_data.map(lambda x, y: (data_augmentation(x, training=False), y))

    model = create_model()
    history = model.fit(train_data.batch(32), validation_data=test_data.batch(32), epochs=10)
    
    save_path = Path("trained_model")
    model.save(save_path)

if __name__ == "__main__":
    main()
