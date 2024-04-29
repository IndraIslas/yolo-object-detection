from PIL import Image
from collections import Counter
import cv2
import numpy as np
import pytesseract
from send_requests import ridinghood

# classes = ["card", "hand", "point", "camera", "rock", "ok"]
# ["action", "stop", "jump", "left", "point", "card"]
processed_ids = {"card": set(), "camera": set(), "hand": set(), "point": set(), "rock": set(), "ok": set()}
# processed_ids = {"action": set(), "stop": set(), "jump": set(), "left": set(), "point": set(), "card": set()}

def plot_bounding_boxes(image, data):
    for box in data['predictions']['box_data']:
        position = box['position']
        class_id = box['class_id']
        box_caption = box['box_caption']
        
        class_label = data['predictions']['class_labels'][class_id]

        minX, minY, maxX, maxY = position['minX'], position['minY'], position['maxX'], position['maxY']
        minX, minY, maxX, maxY = int(minX), int(minY), int(maxX), int(maxY)
        
        cv2.rectangle(image, (minX, minY), (maxX, maxY), (255, 0, 0), 2)
        
        label = f'{class_label}: {box_caption}'
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(image, (minX, minY), (minX + label_width, minY - label_height - baseline), (255, 0, 0), thickness=cv2.FILLED)        
        cv2.putText(image, label, (minX, minY - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def highest_surface(candidates):
    max_area = 0
    best_crop = None
    best_index = None
    for i, crop in enumerate(candidates):
        crop_width = crop[2] - crop[0]
        crop_height = crop[3] - crop[1]
        crop_area = crop_width * crop_height
        if crop_area > max_area:
            max_area = crop_area
            best_crop = crop
            best_index = i
    return best_index, best_crop

def crop_non_overlapping_part(image, card, point):
    # Extract the positions
    card_minX, card_maxX = card['position']['minX'], card['position']['maxX']
    card_minY, card_maxY = card['position']['minY'], card['position']['maxY']
    point_minX, point_maxX = point['position']['minX'], point['position']['maxX']
    point_minY, point_maxY = point['position']['minY'], point['position']['maxY']

    # Find overlapping region coordinates
    overlap_minX = max(card_minX, point_minX)
    overlap_maxX = min(card_maxX, point_maxX)
    overlap_minY = max(card_minY, point_minY)
    overlap_maxY = min(card_maxY, point_maxY)

    # Initialize candidates for non-overlapping parts
    candidates = []

    # Check for non-overlapping on the left side of the card
    if overlap_minX > card_minX:
        left_crop = (card_minX, card_minY, overlap_minX, card_maxY)
        candidates.append(left_crop)

    # Check for non-overlapping on the right side of the card
    if overlap_maxX < card_maxX:
        right_crop = (overlap_maxX, card_minY, card_maxX, card_maxY)
        candidates.append(right_crop)

    # Check for non-overlapping on the top side of the card
    if overlap_minY > card_minY:
        top_crop = (card_minX, card_minY, card_maxX, overlap_minY)
        candidates.append(top_crop)

    # Check for non-overlapping on the bottom side of the card
    if overlap_maxY < card_maxY:
        bottom_crop = (card_minX, overlap_maxY, card_maxX, card_maxY)
        candidates.append(bottom_crop)

    # Evaluate the candidates to find the one with the largest area
    _, best_crop = highest_surface(candidates)

    # If a valid non-overlapping crop is found
    if best_crop:
        non_overlapping_part = image.crop(best_crop)
        return non_overlapping_part

    # If no non-overlapping part is found (fully overlapped), return None
    return None

def check_and_crop_overlap(image, predictions):
    box_data = predictions['box_data']
    class_labels = predictions['class_labels']
    
    # Finding the index for 'card' and 'point'
    card_index = [i for i, label in class_labels.items() if label == 'card']
    point_index = [i for i, label in class_labels.items() if label == 'point']
    
    cards = [box for box in box_data if box['class_id'] in card_index]
    points = [box for box in box_data if box['class_id'] in point_index]
    
    # Function to check if point overlaps on the top or bottom of the card
    def is_overlapping(card, point):
        # Coordinates for card
        card_minX, card_maxX = card['position']['minX'], card['position']['maxX']
        card_minY, card_maxY = card['position']['minY'], card['position']['maxY']
        # Coordinates for point
        point_minX, point_maxX = point['position']['minX'], point['position']['maxX']
        point_minY, point_maxY = point['position']['minY'], point['position']['maxY']
        
        # Checking overlap on the X axis
        overlap_x = (point_minX < card_maxX and point_maxX > card_minX)
        # Checking overlap on the Y axis (top or bottom)
        overlap_y = (point_maxY > card_minY and point_minY < card_maxY) or (point_minY < card_maxY and point_maxY > card_minY)
        
        return overlap_x and overlap_y
    
    overlapping_cards = []
    
    # Checking for any overlap
    for point in points:
        candidates = []
        point_overlapping_card = []
        for card in cards:
            if is_overlapping(card, point):
                # Crop the image of the 'card' if an overlap is found
                position = card['position']
                minX, minY, maxX, maxY = int(position['minX']), int(position['minY']), int(position['maxX']), int(position['maxY'])
                card_coords = (minX, minY, maxX, maxY)
                cropped_card = image.crop(card_coords)
                #get surface of cropped card

                non_overlapped_part = crop_non_overlapping_part(image, card, point)
                surface_area_overlapped = (cropped_card.size[0] * cropped_card.size[1])
                if non_overlapped_part is not None:
                    surface_area_overlapped -= (non_overlapped_part.size[0] * non_overlapped_part.size[1])
                percentage_overlap = (surface_area_overlapped / (cropped_card.size[0] * cropped_card.size[1])) * 100
                
                if non_overlapped_part is None:
                    non_overlapped_part = cropped_card
                candidates.append(percentage_overlap)
                point_overlapping_card.append([cropped_card, non_overlapped_part])
        if len(candidates) == 0:
            continue
        i = candidates.index(max(candidates))
        overlapping_cards.append(point_overlapping_card[i])
    
    # Return the list of cropped images of cards that overlap
    return overlapping_cards

def is_within(card_coords, camera_coords):
    card_minX, card_minY, card_maxX, card_maxY = card_coords
    camera_minX, camera_minY, camera_maxX, camera_maxY = camera_coords

    # Check if all corners of the card are within the camera bounds
    return (card_minX >= camera_minX and card_maxX <= camera_maxX and
            card_minY >= camera_minY and card_maxY <= camera_maxY)

def find_closest_object(objects, object_class, bbox):
    """
    Finds the object ID with the centroid closest to the centroid of a given bounding box.
    
    Args:
    objects (dict): A dictionary containing objects and their coordinates in OrderDicts.
    object_class (str): The class of objects to compare against (e.g., 'card', 'hand').
    bbox (tuple): The bounding box coordinates as (minX, minY, maxX, maxY).

    Returns:
    int: The object ID of the closest object. Returns None if no objects of the class are found.
    """
    if object_class not in objects or not objects[object_class]:
        return None  # No objects of the specified class

    # Calculate the centroid of the bounding box
    bbox_centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    min_distance = float('inf')
    closest_id = None

    # Iterate through each object of the specified class
    for obj_id, obj_centroid in objects[object_class].items():

        # Calculate Euclidean distance from the bbox centroid to this object's centroid
        distance = np.linalg.norm(bbox_centroid - obj_centroid)

        # Check if this is the closest object found so far
        if distance < min_distance:
            min_distance = distance
            closest_id = obj_id

    return closest_id

def check_camera_gesture(image, predictions, objects):
    box_data = predictions['box_data']
    class_labels = predictions['class_labels']
    
    # Finding the index for 'card' and 'point'
    card_index = [i for i, label in class_labels.items() if label == 'card']
    camera_index = [i for i, label in class_labels.items() if label == 'camera']
    
    cards = [box for box in box_data if box['class_id'] in card_index]
    cameras = [box for box in box_data if box['class_id'] in camera_index]

    camera_cards = []
    for camera in cameras:
        camera_coords = (camera['position']['minX'], camera['position']['minY'], camera['position']['maxX'], camera['position']['maxY'])
        camera_ct_id = find_closest_object(objects, 'camera', camera_coords)
        
        for card in cards:
            card_coords = (card['position']['minX'], card['position']['minY'], card['position']['maxX'], card['position']['maxY'])
            if is_within(card_coords, camera_coords):
                card_crop = image.crop(card_coords)
                camera_cards.append(card_crop)
                process_camera_gesture(card_crop, camera_ct_id)
    return camera_cards

def process_camera_gesture(card_crop, camera_ct_id):
    if camera_ct_id and camera_ct_id in processed_ids["camera"]:
        return
    # get text and get color
    text = read_text_from_image(card_crop)
    print(text)
    #crop center of card
    center = crop_center(card_crop)
    color = most_frequent_color(center)
    ridinghood(color, text)
    processed_ids["camera"].add(camera_ct_id)



def crop_center(pil_img):
    width, height = pil_img.size  # Get the dimensions of the image
    new_width, new_height = width // 2, height // 2  # Define the size of the crop
    
    # Calculate the top-left corner of the cropped area
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    
    # Crop the center of the image
    center_cropped_img = pil_img.crop((left, top, right, bottom))
    return center_cropped_img

def most_frequent_color(crop):
    # Convert the image to RGB if it's not
    if crop.mode != 'RGB':
        crop = crop.convert('RGB')
    
    # Get all pixels from the image
    pixels = list(crop.getdata())
    
    # Count the frequency of each color
    color_counts = Counter(pixels)
    if len(color_counts.most_common(1)) == 0:
        return None
    # Find the most frequent color
    most_frequent = color_counts.most_common(1)[0][0]  # Returns the color with the highest count
    # most_frequent =(most_frequent[2], most_frequent[1], most_frequent[1])
    return most_frequent

def get_average_color(image, minX, minY, maxX, maxY):
    """
    Extracts a 10x10 crop centered on the centroid of the bounding box defined by
    (minX, minY, maxX, maxY) in the provided image and computes the average color.

    Args:
    - image (np.array): The image from which to extract the color.
    - minX, minY, maxX, maxY (int): Coordinates defining the bounding box.

    Returns:
    - average_color (tuple): The average BGR color of the crop as a tuple (B, G, R).
    """
    # Calculate the centroid of the bounding box
    cX = int((minX + maxX) / 2)
    cY = int((minY + maxY) / 2)

    # Calculate the start and end points for the 10x10 crop
    start_x = max(cX - 5, 0)
    start_y = max(cY - 5, 0)
    end_x = min(cX + 5, image.shape[1])
    end_y = min(cY + 5, image.shape[0])

    # Extract the 10x10 crop from the image
    # Adjust the crop size if near the borders
    crop = image[start_y:end_y, start_x:end_x]

    # Calculate the average color of the crop
    average_color = np.mean(crop, axis=(0, 1))

    # Return the average color as a tuple, converting it to integer values
    return tuple(int(color) for color in average_color)

def display_color_rectangle(color):
    # Create an image of size 100x100
    width, height = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color the image with the most frequent color
    image[:] = color  # Convert RGB to BGR for OpenCV

    return image

def add_color_squares(image, colors, square_size=50):
    """
    Modifies an image by adding a row of squares of the given BGR colors to the top-left corner.

    Parameters:
        image (numpy.ndarray): The image on which to draw the squares.
        colors (list of tuples): A list of tuples, each representing a BGR color.
        square_size (int): The size of each side of the squares in pixels.

    Returns:
        numpy.ndarray: The modified image.
    """
    # Ensure the square size does not exceed image dimensions
    square_size = min(square_size, image.shape[0], image.shape[1])

    # Draw each square in the list next to each other
    for i, color in enumerate(colors):
        top_left_corner = (i * square_size, 0)  # Move each square to the right
        bottom_right_corner = ((i + 1) * square_size, square_size)

        # Draw the square on the image
        cv2.rectangle(image, top_left_corner, bottom_right_corner, color, thickness=-1)  # thickness=-1 fills the rectangle

    return image

def read_text_from_image(image):
    # Optional: If Tesseract is not in your PATH, include the following line:
    # pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
    # Example:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)

    return text

def change_hue_based_on_color(image, colors):
    # Convert image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Height and width of the image
    height, width = image.shape[:2]

    # Number of sections is equal to the number of colors
    num_sections = len(colors)
    section_width = width // num_sections

    # Apply each color's hue to a section of the image
    for i, color in enumerate(colors):
        # Convert the color from BGR to HSV
        color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = color_hsv[0]

        # Calculate section boundaries
        start_x = i * section_width
        end_x = (start_x + section_width) if i < num_sections - 1 else width

        # Adjust hue in the section
        hsv_image[:, start_x:end_x, 0] = hue

    # Convert back to BGR
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return final_image

def draw_centroids(image, objects, class_label):
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
    # draw both the ID of the object and the centroid of the
    # object on the output frame
        text = f"{class_label} ID {objectID}"
        # centroid = centroid[0]
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

def process_frame(frame, boxes, objects):
    img = frame.copy()
    pointed_cards = check_and_crop_overlap(Image.fromarray(img), boxes['predictions'])
    camera_cards = check_camera_gesture(Image.fromarray(img), boxes['predictions'], objects)
    plot_bounding_boxes(img, boxes)
    colors = []
    for card, non_overlapped_part in pointed_cards:
        # cv2.imshow('Pointed card', np.array(card))
        # cv2.imshow('Non-overlapping part', np.array(non_overlapped_part))
        card_center = crop_center(non_overlapped_part)
        # cv2.imshow('Center cropped non overlapped card', np.array(card_center))
        color = most_frequent_color(card_center)
        if color is not None:
            colors.append(color)
        # color_block = display_color_rectangle(color)
        # cv2.imshow('Most frequent color', color_block)
    # for card in camera_cards:
    #     text = read_text_from_image(card)
    #     print(text)
    #     cv2.imshow(f'Camera card: {text}', np.array(card))
    # if len(camera_cards) > 0:
    #     cv2.imshow('Processed Image', img)
    #     cv2.waitKey(0)
    if len(colors) > 0:
        img = change_hue_based_on_color(img, colors)
    img = add_color_squares(img, colors)

    for class_name in objects:
        draw_centroids(img, objects[class_name], class_name)
    # if len(colors) > 0:
    #     cv2.imshow('Processed Image', img)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow('Processed Image')

    return img