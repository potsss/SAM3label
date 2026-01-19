import requests
import base64
import json
import os
import cv2
import numpy as np

# Configuration
API_URL = "http://localhost:8000/predict"
TEST_IMAGE_PATH = "test.jpg"  # Please make sure this image exists!
OUTPUT_DIR = "test_results"

def encode_image(image_path):
    if not os.path.exists(image_path):
        # Create a dummy image if not exists
        print(f"Image {image_path} not found. Creating a dummy image...")
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.circle(dummy_img, (250, 250), 100, (0, 0, 255), -1) # Red circle
        cv2.rectangle(dummy_img, (50, 50), (150, 150), (0, 255, 0), -1) # Green rect
        cv2.imwrite(image_path, dummy_img)
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def visualize_result(image_path, polygons, output_name):
    """Draws the polygons on the image and saves it."""
    img = cv2.imread(image_path)
    
    for poly in polygons:
        points = np.array(poly["points"], dtype=np.int32)
        label = poly["label"]
        
        # Draw polygon
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Draw label
        if len(points) > 0:
            start_point = tuple(points[0])
            cv2.putText(img, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(out_path, img)
    print(f"Saved visualization to: {out_path}")

def send_request(payload, description):
    print(f"\n--- Testing: {description} ---")
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        print("Status: Success")
        print(f"Polygons found: {len(data['polygons'])}")
        return data['polygons']
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is 'python app.py' running?")
        return []
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals():
            print(response.text)
        return []

def main():
    # 1. Prepare Image
    image_b64 = encode_image(TEST_IMAGE_PATH)
    
    # 2. Test Case A: Text Prompt (Find 'red circle')
    payload_text = {
        "image_base64": image_b64,
        "texts": [{"text": "red circle"}],
        "epsilon_ratio": 0.005
    }
    polys_text = send_request(payload_text, "Text Prompt ('red circle')")
    visualize_result(TEST_IMAGE_PATH, polys_text, "result_text.jpg")

    # 3. Test Case B: Point Prompt (Click on center)
    payload_point = {
        "image_base64": image_b64,
        "points": [{"point": [250, 250], "label": 1}], # Center of red circle
        "epsilon_ratio": 0.005
    }
    polys_point = send_request(payload_point, "Point Prompt (Center Click)")
    visualize_result(TEST_IMAGE_PATH, polys_point, "result_point.jpg")

    # 4. Test Case C: Box Prompt (Box around green rect)
    payload_box = {
        "image_base64": image_b64,
        "boxes": [{"box": [40, 40, 160, 160]}], # Around green rect
        "epsilon_ratio": 0.005
    }
    polys_box = send_request(payload_box, "Box Prompt (Around Green Rect)")
    visualize_result(TEST_IMAGE_PATH, polys_box, "result_box.jpg")

if __name__ == "__main__":
    main()
