import os
import cv2


PATH_INPUT = "C:\\Users\\a774880\\Desktop\\serialized\\mona_lisa"
PATH_OUTPUT = "C:\\Users\\a774880\\Desktop\\videos"
P1, P2, P3 = os.path.join(PATH_INPUT, "serialized_0"), os.path.join(PATH_INPUT, "serialized_1"), os.path.join(PATH_INPUT, "serialized_2")


def make_a_video():
    video_name = os.path.join(PATH_OUTPUT, 'generated.mp4')
    mapped_images = {}
    images = sorted([img for img in os.listdir(PATH_INPUT) if img.endswith(".jpg")])
    for image in images:
        mapped_images[int(image.split(".")[0].replace("the_best_", ""))] = image

    frame = cv2.imread(os.path.join(PATH_INPUT, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 50, (width, height))
    for i, image in enumerate(sorted(list(mapped_images.keys()))):
        if (i+1) % 100 == 0:
            print(f"{i+1}/{len(mapped_images)}")
        video.write(cv2.imread(os.path.join(PATH_INPUT, mapped_images[image])))
    print("FINISHED!")


if __name__ == "__main__":
    make_a_video()
