import os
import tensorflow as tf
from pathlib import Path
import json

class Training:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        # Clear any existing TensorFlow/Keras session
        tf.keras.backend.clear_session()

        # Load the model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Compile the model with a fresh optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate= self.config.params_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="binary" 
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        print("Valid Dir ", self.config.training_data)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        print("Train Dir ", self.config.training_data)
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        print(self.train_generator.class_indices)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model using `get_base_model()` before training.")

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=1,  
            # self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def save_classes(self):
        save_dir = "model_with_classes"
        os.makedirs(save_dir, exist_ok=True) 
        with open("model_with_classes/class_indices.json", "w") as f:
            json.dump(self.train_generator.class_indices, f)
    