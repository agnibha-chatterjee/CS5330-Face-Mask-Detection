from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.preprocessing.image import img_to_array
from keras.api.models import load_model
import numpy as np
import argparse
import cv2
import os

def parse_args():
	"""Parse command line arguments for image path, model paths, and options."""
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to input image")
	ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str, default="mask_detector.keras", help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	ap.add_argument("-o", "--output", type=str, help="(optional) path to save output image")
	return ap.parse_args()

def load_face_detector(face_dir: str):
	"""
	Load the face detector model from disk.
	Raises FileNotFoundError if required files are missing.
	"""
	prototxtPath = os.path.sep.join([face_dir, "deploy.prototxt"])
	weightsPath = os.path.sep.join([face_dir, "res10_300x300_ssd_iter_140000.caffemodel"])
	if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
		print(f"[Error] Face detector model files not found in {face_dir}.")
		raise FileNotFoundError("Face detector model files not found.")
	print(f"[Info] Face detector model loaded from {face_dir}.")
	return cv2.dnn.readNet(prototxtPath, weightsPath)

def detect_and_predict_mask(image, net, model, confidence_threshold):
	"""
	Detect faces in the image and predict mask usage for each detected face.
	Returns a list of results: ((startX, startY, endX, endY), mask_prob, no_mask_prob)
	"""
	(h, w) = image.shape[:2]
	# Prepare the image for face detection
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	results = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > confidence_threshold:
			# Compute bounding box coordinates
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = image[startY:endY, startX:endX]
			if face.size == 0:
				print("[WARNING] Detected face region is empty, skipping.")
				continue
			# Preprocess the face ROI for mask prediction
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			# Predict mask/no mask
			(mask, withoutMask) = model.predict(face)[0]
			results.append(((startX, startY, endX, endY), mask, withoutMask))
			print(f"[Info] Face detected at [{startX}, {startY}, {endX}, {endY}] - Mask: {mask:.2f}, No Mask: {withoutMask:.2f}")
	if not results:
		print("[Warning] No faces detected with confidence above threshold.")
	return results

def draw_results(image, results):
	"""
	Draw bounding boxes and labels on the image for each detection.
	"""
	for (startX, startY, endX, endY), mask, withoutMask in results:
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# Draw label and bounding box on the image
		cv2.putText(
			image, label_text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA
		)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	return image

def mask_image(args):
	"""
	Main function to process the image:
	- Loads models
	- Detects faces and predicts mask usage
	- Draws results and displays/saves the output
	"""
	print("[Info] Loading face detector model...")
	net = load_face_detector(args.face)
	print("[Info] Loading face mask detector model...")
	model = load_model(args.model)
	image = cv2.imread(args.image)
	if image is None:
		print(f"[Error] Input image {args.image} not found or cannot be read.")
		raise FileNotFoundError(f"Input image {args.image} not found.")
	orig = image.copy()
	print("[Info] Computing face detections...")
	results = detect_and_predict_mask(orig, net, model, args.confidence)
	output = draw_results(orig, results)
	# Display the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# Optionally save the output image
	if args.output:
		cv2.imwrite(args.output, output)
		print(f"[Info] Output image saved to {args.output}")

if __name__ == "__main__":
	args = parse_args()
	try:
		mask_image(args)
	except KeyboardInterrupt:
		print("\n[Info] Interrupted by user. Exiting gracefully.")
		cv2.destroyAllWindows()
	except Exception as e:
		print(f"[Error] An error occurred: {str(e)}")
