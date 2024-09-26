import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained MNIST model (ensure it's already trained or use a provided one)
model = load_model('lab2_model.h5')  # Load your pre-trained model here

# Function to preprocess the captured image for the model
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28, as expected by the MNIST model
    resized = cv2.resize(gray, (28, 28))

    # Invert the image (black background, white digits)
    inverted = cv2.bitwise_not(resized)
    
    # Display the live video
    #cv2.imshow("Webcam - Press 'c' to capture the image", inverted)

    # Normalize the image to the range [0, 1]
    normalized = inverted.astype('float32') / 255.0

    # Reshape the image to fit the input shape of the model
    reshaped = normalized.reshape(1, 28, 28, 1)

    return reshaped

def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Wait for keypress
        key = cv2.waitKey(1)

        # Press 'c' to capture the image
        if key == ord('c'):
            # Save the frame as an image
            img = frame
            break

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return img

# Main function to capture and predict the digit
def main():
    print("Starting webcam...")

    # Capture an image
    img = capture_image()

    # Preprocess the image for digit recognition
    processed_img = preprocess_image(img)
    
    cv2.imshow("Image from Array", preprocess_image)

    # Make a prediction using the pre-trained model
    prediction = model.predict(processed_img).argmax()

    # Display the result
    print(f"Predicted digit: {prediction}")

    # Show the captured image with the prediction
    cv2.imshow(f"Predicted digit: {prediction}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
