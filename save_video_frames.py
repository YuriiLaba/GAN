import cv2

def save_video_frames(path, step=1, name="data/simp1_frame", size=128, lower_bound=100, upper_bound=100):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        if count > upper_bound:
            break

        success,image = vidcap.read()
        print('Read a new frame: ', success)
        if count > lower_bound and not(count % step):
            resized_im = cv2.resize(image, (size, size))
            cv2.imwrite(name +  str(count) + ".jpg", resized_im)     # save frame as JPEG file
        count += 1

if __name__ == "__main__":
    save_video_frames("Simp_1.mp4",lower_bound=100, upper_bound=30000, step=10)