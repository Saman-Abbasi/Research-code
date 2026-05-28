import cv2

img = cv2.imread("data/room.jpg")

resized = cv2.resize(img, (800, 600))

print("Image shape:", img.shape)

cv2.imshow("Loaded Image", resized)

cv2.imwrite("data/output.jpg", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()