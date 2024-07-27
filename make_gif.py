import os
import tqdm
import cv2
import numpy as np
from suncompass import SunCompass


def make_gif(image_directory: str, frame_duration_ms: int):

    suncompass = SunCompass()
    suncompass.set_eval(dropout=False)

    # Iterate over the files in the images directory and create a gif
    image_names = os.listdir(image_directory)
    # Sort the images by name
    image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))
    images = []
    for file in tqdm.tqdm(image_names):
        if file.endswith(".jpg"):
            image = cv2.imread(f"{image_directory}/{file}")
            img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
            # Write the theta_rad on the image
            cv2.putText(img_with_suncompass, f"Sun direction: {np.rad2deg(theta_rad):.1f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            images.append(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))

    # Save the images as a gif
    import imageio
    imageio.mimsave(f"{image_directory}.gif", images, duration=frame_duration_ms, loop=0)


if __name__ == "__main__":
    make_gif("menton", 1000//30)
