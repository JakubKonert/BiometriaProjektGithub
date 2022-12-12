import cv2

kasiaImage = cv2.imread("Dataset/toCopy/kasia.jpg")
kubaImage = cv2.imread("Dataset/toCopy/Kuba.jpg")
bartekImage = cv2.imread("Dataset/toCopy/bartek.jpg")
adamImage = cv2.imread("Dataset/toCopy/adam.jpg")

(h, w) = kasiaImage.shape[:2]
(cX, cY) = (w // 2, h // 2)

for i in range(300):
    
    M = cv2.getRotationMatrix2D((cX, cY), i, 1.0)
    kasiaImage = cv2.warpAffine(kasiaImage, M, (w, h))
    kubaImage = cv2.warpAffine(kubaImage, M, (w, h))
    bartekImage = cv2.warpAffine(bartekImage, M, (w, h))
    adamImage = cv2.warpAffine(adamImage, M, (w, h))



    cv2.imwrite("Dataset/Copied/Kasia"+str(i)+".jpg",kasiaImage)
    cv2.imwrite("Dataset/Copied/Kuba"+str(i)+".jpg",kubaImage)
    cv2.imwrite("Dataset/Copied/Bartek"+str(i)+".jpg",bartekImage)
    cv2.imwrite("Dataset/Copied/Adam"+str(i)+".jpg",adamImage)