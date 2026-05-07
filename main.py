import os
from model import *
from data import *
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# Data Augmentation Params
# =========================
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# =========================
# Paths
# =========================
train_path = 'data/membrane/train'
test_path = "data/membrane/test"

# =========================
# Training Generator
# =========================
myGene = trainGenerator(
    batch_size=1,
    train_path=train_path,
    image_folder='image',
    mask_folder='label',
    aug_dict=data_gen_args,
    save_to_dir=None
)

# =========================
# Model
# =========================
model = unet()

# =========================
# Callbacks (FORCE SAVE)
# =========================
model_checkpoint = ModelCheckpoint(
    filepath='unet_membrane.keras',
    monitor='loss',
    verbose=1,
    save_best_only=False   # 🔥 always save (important for testing)
)

# =========================
# Training (FAST MODE)
# =========================
print("Starting FAST training...")

model.fit(
    myGene,
    steps_per_epoch=5,   # 🔥 VERY SMALL → ~2 min training
    epochs=1,
    callbacks=[model_checkpoint]
)

print("Training completed!")

# =========================
# Prediction
# =========================
testGene = testGenerator(test_path)

print("Starting prediction...")

results = model.predict(
    testGene,
    steps=30,
    verbose=1
)

print("Prediction completed!")

# =========================
# Save Results
# =========================
saveResult(test_path, results)

print("Results saved successfully!")
