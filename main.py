import tensorflow as tf
import keras
import numpy as np
photo = "Early-blight-2.png"
img = keras.utils.load_img(photo, target_size=(160, 160))
model = keras.models.load_model('plant_diseaseV1.keras')

def extract_features(image):
    image = keras.utils.load_img(image,target_size=(160,160))
    feature = keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    # print(prediction)
    prediction_label = class_names[prediction.argmax()]
    return prediction_label

# Convert to array
img_array = keras.utils.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)

img_array = img_array / 255.0

predictions = model.predict(img_array) # type: ignore
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

predicted_index = np.argmax(model.predict(img_array)) # type: ignore

predicted_class = class_names[predicted_index]

print("Predicted class:", model_predict(photo))
