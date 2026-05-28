import cv2

img = cv2.imread("data/room.jpg")

resized = cv2.resize(img, (800, 600))

blurred = cv2.GaussianBlur(resized, (15, 15), 0)

cv2.imshow("Blurred", blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()