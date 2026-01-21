import requests
import base64
import json
import os
import cv2
import numpy as np
import argparse
import random
import math

# Configuration
API_URL_FORMAT = "http://{}:{}/predict"
DEFAULT_SERVER_IP = "112.51.6.147"
DEFAULT_PORT = 8069
TEST_IMAGE_PATH = "test.jpg"
OUTPUT_DIR = "test_results"

# Red Circle Geometry
CIRCLE_CENTER = (250, 250)
CIRCLE_RADIUS = 100

def encode_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Creating a dummy image...")
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.circle(dummy_img, CIRCLE_CENTER, CIRCLE_RADIUS, (0, 0, 255), -1) # Red circle
        cv2.imwrite(image_path, dummy_img)
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def visualize_result(image_path, polygons, points, boxes, output_name):
    """Draws polygons, points, and boxes on the image and saves it."""
    img = cv2.imread(image_path)
    
    # Draw generated points
    if points:
        for p in points:
            cv2.circle(img, (int(p['point'][0]), int(p['point'][1])), 5, (0, 255, 0), -1) # Green dots for points

    # Draw generated boxes
    if boxes:
        for b in boxes:
            box = b['box']
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2) # Blue boxes

    # Draw result polygons
    if polygons:
        for poly in polygons:
            poly_points = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(img, [poly_points], isClosed=True, color=(0, 255, 255), thickness=2)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(out_path, img)
    print(f"Saved visualization to: {out_path}")

def generate_random_points_in_circle(n=10):
    """Generates n random points inside the defined circle."""
    points = []
    for _ in range(n):
        r = CIRCLE_RADIUS * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        x = int(CIRCLE_CENTER[0] + r * math.cos(theta))
        y = int(CIRCLE_CENTER[1] + r * math.sin(theta))
        points.append({"point": [x, y], "label": 1}) # All positive points
    return points

def generate_random_boxes_for_circle(n=5):
    """Generates n random bounding boxes that contain the circle."""
    boxes = []
    min_x, min_y = CIRCLE_CENTER[0] - CIRCLE_RADIUS, CIRCLE_CENTER[1] - CIRCLE_RADIUS
    max_x, max_y = CIRCLE_CENTER[0] + CIRCLE_RADIUS, CIRCLE_CENTER[1] + CIRCLE_RADIUS
    for _ in range(n):
        x1 = min_x - random.randint(5, 30)
        y1 = min_y - random.randint(5, 30)
        x2 = max_x + random.randint(5, 30)
        y2 = max_y + random.randint(5, 30)
        boxes.append({"box": [x1, y1, x2, y2]})
    return boxes

def send_request(api_url, payload, description):
    print(f"--- Testing: {description} ---")
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        print("Status: Success")
        print(f"Polygons found: {len(data['polygons'])}")
        return data['polygons']
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {api_url}.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals():
            print(response.text)
        return []

def main():
    parser = argparse.ArgumentParser(description="Send randomized test requests to the SAM3 annotation server.")
    parser.add_argument("--ip", type=str, default=DEFAULT_SERVER_IP, help=f"Server IP address (default: {DEFAULT_SERVER_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    api_url = API_URL_FORMAT.format(args.ip, args.port)
    print(f"Targeting server at: {api_url}")

    # 1. Prepare Image
    image_b64 = encode_image(TEST_IMAGE_PATH)
    
    # 2. Test Case A: 10 Random Points in Circle
    random_points = generate_random_points_in_circle(10)
    payload_points = {
        "image_base64": image_b64,
        "points": random_points,
        "epsilon_ratio": 0.005
    }
    polys_points = send_request(api_url, payload_points, "10 Random Points in Circle")
    visualize_result(TEST_IMAGE_PATH, polys_points, random_points, [], "result_random_points.jpg")

    # 3. Test Case B: 5 Random Boxes around Circle
    random_boxes = generate_random_boxes_for_circle(5)
    payload_boxes = {
        "image_base64": image_b64,
        "boxes": random_boxes,
        "epsilon_ratio": 0.005
    }
    polys_boxes = send_request(api_url, payload_boxes, "5 Random Boxes around Circle")
    visualize_result(TEST_IMAGE_PATH, polys_boxes, [], random_boxes, "result_random_boxes.jpg")

if __name__ == "__main__":
    main()