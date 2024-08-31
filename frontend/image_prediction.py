import numpy as np
import cv2


"""
This script has functions to be able to predict where object are in a picture:
- getting_bounding_boxes will use the model to predict what objects are found and where they are
- create_image will create an image putting togheter both the original picture and the prediction
- save_image will save the image when the code is runned localy
- full_process will run the 3 functions
"""


def create_image(original_image_array: np.array, bound_boxes: dict) -> np.array:
    """
    Takes both:
    - The original image array
    - The result from the bounding boxes

    And returns an image with both elements in array format
    """

    print(len(bound_boxes))

    # Create an OpenCV image from the numeric array
    opencv_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)

    # Annotate bounding boxes on the OpenCV image
    for box_info in bound_boxes:
        coordinates = box_info["Coordinates"]
        object_type = box_info["Object type"]
        probability = box_info["Probability"]

        # Convert float coordinates to integers
        coordinates = [int(coord) for coord in coordinates]

        # Draw rectangle on the image around the face
        cv2.rectangle(
            opencv_image,
            pt1=(coordinates[0], coordinates[1]),
            pt2=(coordinates[2], coordinates[3]),
            color=(92, 201, 116),
            thickness=2,
        )

        # Rectangle holding the face text
        cv2.rectangle(
            opencv_image,
            (coordinates[0], coordinates[1] - 5),  # X1, Y1
            (coordinates[0] + 65, coordinates[1] - 30),  # X2, Y2
            (76, 166, 96),  # Color RGB
            -1,  # thickness = -1 to fill the entire thing
        )

        # Rectangle holding the score
        cv2.rectangle(
            opencv_image,
            (coordinates[0], coordinates[1] - 35),  # X1, Y1
            (coordinates[0] + 65, coordinates[1] - 60),  # X2, Y2
            (92, 201, 116),  # Color RGB
            -1,  # thickness = -1 to fill the entire thing
        )

        # Annotate with object type and probability
        cv2.putText(
            opencv_image,
            object_type,
            (coordinates[0] + 5, coordinates[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        # cv2.FONT_HERSHEY_SIMPLEX, size, color, width

        # Annotate with object type and probability
        cv2.putText(
            opencv_image,
            str(probability),
            (coordinates[0] + 5, coordinates[1] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

    # Convert the annotated image back to RGB format
    annotated_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Display or save the annotated image as needed
    return annotated_image
