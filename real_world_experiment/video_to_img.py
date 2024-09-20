import cv2
video_path = "../data/real_world_data_4/video/video.avi"
output_folder = "../data/real_world_data_4/img/"

# Open the video file
video = cv2.VideoCapture(video_path)

# Read the frames and save them as images
frame_count = 0
while True:
    # Read the next frame
    ret, frame = video.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Save the frame as an image
    image_path = f"{output_folder}/img-{frame_count}.jpg"
    cv2.imwrite(image_path, frame)

    # Increment the frame count
    frame_count += 1

# Release the video file
video.release()