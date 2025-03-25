import cv2
import numpy as np

def expand_and_color_light_spot(image_path, output_path, threshold=200, expansion_size=30, alpha=0.5):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_size, expansion_size))

    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    red_layer = np.zeros_like(colored_image)
    red_layer[expanded_mask == 255] = [0, 0, 255]  # BGR формат для красного

    output_image = np.where(red_layer == 0, colored_image, red_layer)

    cv2.imwrite(output_path, output_image)

# For example
expand_and_color_light_spot('input_image.png', 'output_image.png', threshold=150, expansion_size=30, alpha=0.5)
