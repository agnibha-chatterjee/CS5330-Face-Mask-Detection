# train.py
#
# Trains a face mask detector model using a pre-trained MobileNetV2 model.
#
# Authors
# - Agnibha Chatterjee
# - Om Agarwal

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.api import mixed_precision
from keras.api.applications import MobileNetV2
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api import mixed_precision
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def set_mixed_precision():
	# Enable mixed precision only if a GPU is available
	if tf.config.list_physical_devices('GPU'):
		print("[Info] Enabling mixed precision for GPU.")
		mixed_precision.set_global_policy('mixed_float16')
	else:
		print("[Info] Mixed precision not enabled (no GPU detected).")

def parse_args():
	ap = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	ap.add_argument("-d", "--dataset", required=True,
		help="Path to input dataset directory")
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="Path to output loss/accuracy plot (PNG)")
	ap.add_argument("--plot_pdf", type=str, default="plot.pdf",
		help="Path to output loss/accuracy plot (PDF)")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.keras",
		help="Path to output face mask detector model")
	ap.add_argument("--epochs", type=int, default=20,
		help="Number of training epochs")
	ap.add_argument("--batch_size", type=int, default=64,
		help="Batch size")
	ap.add_argument("--learning_rate", type=float, default=1e-4,
		help="Initial learning rate")
	return ap.parse_args()

def main():
	args = parse_args()
	set_mixed_precision()

	INIT_LR = args.learning_rate
	EPOCHS = args.epochs
	BS = args.batch_size

	# Data augmentation and generators
	aug = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest",
		validation_split=0.2
	)

	print(f"[INFO] Loading images from {args.dataset} ...")
	train_gen = aug.flow_from_directory(
		args.dataset,
		target_size=(224, 224),
		batch_size=BS,
		class_mode="categorical",
		subset="training",
		shuffle=True
	)
	val_gen = aug.flow_from_directory(
		args.dataset,
		target_size=(224, 224),
		batch_size=BS,
		class_mode="categorical",
		subset="validation",
		shuffle=False
	)

	# Print class distribution
	print(f"[INFO] Class indices: {train_gen.class_indices}")
	print(f"[INFO] Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

	# Build the model
	print("[INFO] Building model...")
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
							input_tensor=Input(shape=(224, 224, 3)))
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(train_gen.num_classes, activation="softmax")(headModel)
	model = Model(inputs=baseModel.input, outputs=headModel)

	for layer in baseModel.layers:
		layer.trainable = False

	print("[INFO] Compiling model...")
	opt = Adam(learning_rate=INIT_LR)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	# Train the model
	print("[INFO] Training head...")
	steps_per_epoch = int(np.ceil(train_gen.samples / BS))
	validation_steps = int(np.ceil(val_gen.samples / BS))
	H = model.fit(
		train_gen,
		steps_per_epoch=steps_per_epoch,
		validation_data=val_gen,
		validation_steps=validation_steps,
		epochs=EPOCHS
	)

	# Evaluate the model
	print("[INFO] Evaluating network...")
	val_gen.reset()
	predIdxs = model.predict(val_gen, batch_size=BS)
	predIdxs = np.argmax(predIdxs, axis=1)
	trueIdxs = val_gen.classes

	print(classification_report(trueIdxs, predIdxs, target_names=list(val_gen.class_indices.keys())))

	# Save the model
	print(f"[INFO] Saving mask detector model to {args.model} ...")
	model.save(args.model)

	# Plot training loss and accuracy
	N = EPOCHS
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.tight_layout()
	plt.savefig(args.plot)
	plt.savefig(args.plot_pdf)
	print(f"[INFO] Training plot saved as {args.plot} and {args.plot_pdf}")

if __name__ == "__main__":
	main()

