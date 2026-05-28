import cv2

img = cv2.imread("data/room.jpg")

resized = cv2.resize(img, (800, 600))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()