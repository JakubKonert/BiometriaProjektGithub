import keras
import cv2
import numpy as np

model = keras.models.load_model("Models/EfficientNetB7_BIO.h5")
image = cv2.imread("Dataset/test/BartekZewOtwa/KlatkaNR3038_jpg.rf.7f13ad588c1c0819d38a7687ef33fdfd.jpg")


# testDSPath = "Dataset/test"

# testDS =  keras.utils.image_dataset_from_directory(
#     directory=testDSPath,
#     labels='inferred',
#     label_mode='categorical',
#     batch_size=32,
#     image_size=(600, 600))

# result = model.evaluate(testDS)
# dictionary=dict(zip(model.metrics_names, result))

classesNames = ['AdamWewOtwa', 'AdamWewZamk', 'AdamZewOtwa', 'AdamZewZamk', 'BartekWewOtwa', 'BartekWewZamk', 'BartekZewOtwa', 'BartekZewZamk', 'KasiaWewOtwa', 'KasiaWewZamk', 'KasiaZewOtwa', 'KasiaZewZamk', 'KubaWewOtwa', 'KubaWewZamk', 'KubaZewOtwa','KubaZewZamk']

imagePred = cv2.resize(image,(600,600),interpolation=cv2.INTER_LANCZOS4)
imagePred = imagePred.reshape(1,600,600,3)
imagePred = imagePred/255

prediction = model.predict(imagePred)

predMax = np.argmax(prediction)
predictionResult = classesNames[predMax]
print(f"Prediction: {prediction}")
print(f"PredMax: {predMax}")
print(f"PredictionResult: {predictionResult}")

cv2.putText(image,predictionResult,(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,180,20),2,cv2.LINE_AA)
cv2.imshow("Classification Complication",image)
cv2.waitKey(0)



