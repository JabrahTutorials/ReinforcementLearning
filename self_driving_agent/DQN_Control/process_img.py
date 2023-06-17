import cv2

img = cv2.imread('../output/000970.png')
print(img.shape)

scale_percent = 25
width = int(img.shape[1] * scale_percent/100)
height = int(img.shape[0] * scale_percent/100)

dim = (128, 128)

resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
cv2.imshow('', img_gray)
cv2.waitKey(5000)
cv2.destroyAllWindows()
