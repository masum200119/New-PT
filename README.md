Custom Invisible Watermarking Prototype

Project Overview

This project implements a custom invisible image watermarking prototype inspired by HiDDeN. It uses a deep learning-based encoder to embed a binary identity (ID) into an image and a decoder to recover the embedded bits.

The system includes differentiable distortions for training, batch image embedding, watermark extraction, statistical detection using p-values, and robustness evaluation under real-world image attacks.

Project Files

model.py — Implements the Encoder (image + bits → watermarked image) and Decoder (image → recovered bits).

distortions.py — Implements differentiable attacks used during training, including noise, blur, resize, crop, dropout, and JPEG compression.

train.py — Trains the model and generates training logs (log.csv), sample watermarked images (sample.png), and model checkpoints (ckpt.pt).

embed.py — Embeds an identity into all images in a folder and saves the watermarked images as PNG files.

extract.py — Extracts the embedded identity from images and reports a DETECTED or NOT DETECTED decision with a p-value.

test_robust.py — Evaluates watermark robustness against real-world attacks, including PIL-based JPEG compression, resizing, cropping, blurring, noise, and rotation.

1. Training

Hardware requirements: A GPU is recommended for training. Google Colab's free NVIDIA T4 GPU is suitable for initial experiments, depending on dataset size and training configuration.

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Prepare the dataset

Prepare a dataset containing at least 5,000–10,000 images in a folder. Suitable sources include:

COCO

DIV2K

A large collection of your own images

Provide separate directories for training and validation images.

Step 3: Train the model

Run the following command:

python train.py --data DATA_DIR --val VAL_DIR --out runs/exp1 --bits 32 --size 128 --batch 16 --epochs 40 --warmup 2000 --workers 2

Parameter descriptions:

Parameter

Description

--data

Training dataset directory

--val

Validation dataset directory

--out

Output directory for logs, samples, and checkpoints

--bits 32

Number of watermark identity bits

--size 128

Input image size of 128 × 128 pixels

--batch 16

Training batch size

--epochs 40

Number of training epochs

--warmup 2000

Warmup setting, as defined by the training script

--workers 2

Number of data-loading workers

Training targets

Monitor the following metrics during training:

Identity recovery accuracy: Target at least 95%.

Image quality: Target a PSNR of at least 33 dB.

If these targets are not achieved, consider training for more epochs and adjusting the loss weights and other hyperparameters.

The w_img parameter controls the trade-off between image quality and watermark robustness. Increasing w_img generally prioritizes visual similarity, potentially reducing robustness, while decreasing it may strengthen the watermark at the expense of image quality. The exact behavior depends on how the loss function is implemented in train.py.

These values are experimental targets, not guaranteed outcomes.

2. Watermark Embedding

After training, embed a watermark identity into all supported images in the input folder.

python embed.py --ckpt runs/exp1/ckpt.pt --input images --output marked --id 123456 --key mysecret

Arguments:

--ckpt: Path to the trained model checkpoint.

--input: Directory containing the original images.

--output: Directory for the watermarked images.

--id: Identity to embed.

--key: Secret key used by the implementation.

The output images are saved in PNG format to preserve the embedded watermark as much as possible.

Assign a different ID to each image

To embed a separate identity into each image, use the --per_image option:

python embed.py --ckpt runs/exp1/ckpt.pt --input images --output marked --id 123456 --key mysecret --per_image

The script generates an ids.csv file containing the per-image identity information, provided this functionality is implemented in the script.

Keep this file secure if it is needed to associate images with their embedded identities.

3. Watermark Extraction and Detection

To recover and verify the embedded identity, run:

python extract.py --ckpt runs/exp1/ckpt.pt --input marked --id 123456 --key mysecret

The extraction process uses the trained decoder to recover the watermark bits and evaluates the detection result.

The expected detection outcomes are:

DETECTED — The extracted watermark meets the implementation's detection criteria.

NOT DETECTED — The extracted watermark does not meet the detection criteria.

The p-value provides a statistical measure of how compatible the observed result is with the specified null hypothesis. Its interpretation depends on the statistical test and threshold implemented in extract.py; it should not be treated as a guarantee of authenticity or security.

4. Robustness Evaluation

To evaluate the robustness of the embedded watermark against common image-processing operations, run:

python test_robust.py --ckpt runs/exp1/ckpt.pt --input marked --id 123456 --key mysecret --limit 20

This command evaluates up to 20 images from the input directory.

The test suite includes:

JPEG compression

Image resizing

Cropping

Gaussian or other configured blurring

Noise addition

Image rotation

The purpose of this evaluation is to determine whether the watermark can still be detected after common image transformations.

5. Expected Outputs

After training and evaluation, the project may generate the following outputs:

File or directory

Purpose

runs/exp1/log.csv

Training and validation metrics

runs/exp1/sample.png

Sample watermarked image

runs/exp1/ckpt.pt

Trained model checkpoint

marked/

Watermarked PNG images

ids.csv

Per-image identity mapping when --per_image is enabled

Robustness report

Detection and recovery results under different attacks
