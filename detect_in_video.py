import argparse
import imutils
import time
import cv2
import os
import numpy as np
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.preprocessing.image import img_to_array
from keras.api.models import load_model
from imutils.video import VideoStream


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str, default="face_detector", help="Path to face detector model directory")
	ap.add_argument("-m", "--model", type=str, default="mask_detector.keras", help="Path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
	return vars(ap.parse_args())

def load_face_detector(face_dir):
	prototxtPath = os.path.sep.join([face_dir, "deploy.prototxt"])
	weightsPath = os.path.sep.join([face_dir, "res10_300x300_ssd_iter_140000.caffemodel"])
	if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
		print(f"[Error] Face detector model files not found in '{face_dir}'.")
		raise FileNotFoundError("Face detector model files not found.")
	print(f"[Info] Face detector model loaded from '{face_dir}'.")
	return cv2.dnn.readNet(prototxtPath, weightsPath)

def draw_results(frame, locs, preds):
	"""Draw bounding boxes and labels for each detected face."""
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(
			frame, label_text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA
		)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	return frame

def detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold):
	"""
	Detect faces in the frame and predict mask usage for each.
	Returns bounding boxes and predictions.
	"""
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces, locs, preds = [], [], []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > confidence_threshold:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			if face.size == 0:
				continue
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	else:
		print("[Info] No faces detected in this frame.")
	return (locs, preds)

def main():
	args = parse_args()
	print("[Info] Loading face detector model...")
	faceNet = load_face_detector(args["face"])
	print("[Info] Loading face mask detector model...")
	maskNet = load_model(args["model"])
	print("[Info] Starting video stream. Press 'q' to quit.")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	try:
		while True:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, args["confidence"])
			frame = draw_results(frame, locs, preds)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				print("[Info] 'q' pressed. Exiting video stream.")
				break
	except KeyboardInterrupt:
		print("\n[Info] Interrupted by user. Exiting gracefully.")
	finally:
		cv2.destroyAllWindows()
		vs.stop()
		print("[Info] Video stream stopped. Resources released.")

if __name__ == "__main__":
	main()
