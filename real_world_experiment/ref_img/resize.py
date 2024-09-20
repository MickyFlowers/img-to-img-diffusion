import cv2

img_path = "/home/cyx/chenyuxi/project/img-to-img-diffusion/real_world_experiment/ref_img/Out_00837.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 480))
cv2.imwrite("/home/cyx/chenyuxi/project/img-to-img-diffusion/real_world_experiment/ref_img/Out_00837_resized.jpg", img)
