import tensorflow as tf
import numpy as np
import json

class Prediction:
    
    def __init__(self, config, img):
        self.config = config
        self.img = img

    def predict(self):
     
        # Path to your image
        # img_path = "artifacts/data_ingestion/Chest-CT-Scan-data/normal/3.png"

        # Load the image with same size as training
        img = tf.keras.utils.load_img(
            self.img,
            target_size= self.config.params_image_size[:-1]  # e.g. (224, 224)
        )

        # Convert to array
        img_array = tf.keras.utils.img_to_array(img)

        # Scale (same as rescale=1./255 in generator)
        img_array = img_array / 255.0

        # Add batch dimension: (1, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Load later
        model = tf.keras.models.load_model(self.config.trained_model_path)

        try:
            with open("model_with_classes/class_indices.json") as f:
                class_indices = json.load(f)

            # Predict
            pred = model.predict(img_array)

            # For binary classification
            if pred.shape[1] == 1:  # sigmoid output
                predicted_class = (pred > 0.5).astype("int32")[0][0]
            else:  # softmax output
                predicted_class = np.argmax(pred, axis=1)[0]

            # Map back to class name
            class_labels = list(class_indices.keys())
            print("Predicted class:", class_labels[predicted_class])

            return class_labels[predicted_class]

        except FileNotFoundError:
            return "Class labels file not found !" 




