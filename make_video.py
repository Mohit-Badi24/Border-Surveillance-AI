import os
import cv2

image_folder = "data/visdrone_val"
output_path = "data/visdrone_val.mp4"

images = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")])

if not images:
    print("No images found!")
    exit()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape
fps = 15  # adjust as needed

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    out.write(img)

out.release()
print("Video created at:", output_path)