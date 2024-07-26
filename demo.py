import numpy as np
import cv2
import matplotlib.pyplot as plt

from suncompass import SunCompass


if __name__ == "__main__":
    image_path = "croatia.jpg"
    suncompass = SunCompass()
    suncompass.set_eval(dropout=False)

    image = cv2.imread(image_path)
    img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))
    plt.title(f"Sun direction: {np.degrees(theta_rad):.1f} degrees")

    plt.subplot(2, 2, 2)
    image_path = "menton.jpg"
    image = cv2.imread(image_path)
    img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
    plt.imshow(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))
    plt.title(f"Sun direction: {np.degrees(theta_rad):.1f} degrees")

    plt.subplot(2, 2, 3)
    image_path = "gargnano.jpg"
    image = cv2.imread(image_path)
    img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
    plt.imshow(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))
    plt.title(f"Sun direction: {np.degrees(theta_rad):.1f} degrees")

    plt.subplot(2, 2, 4)
    image_path = "japan.jpg"
    image = cv2.imread(image_path)
    img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
    plt.imshow(cv2.cvtColor(img_with_suncompass, cv2.COLOR_BGR2RGB))
    plt.title(f"Sun direction: {np.degrees(theta_rad):.1f} degrees")

    plt.tight_layout()
    plt.show()
