# Model B: MediaPipe Landmarks + MLP (Goal 1, ASL Letters)

Model B classifies the 24 static ASL letters (A–Y, no J or Z) from a single image.
It's the team's main Goal 1 model. The raw-image CNN (Model A) is the comparison.

## The big idea

We don't feed the model pictures. We feed it the hand's shape.

```
Image  →  MediaPipe  →  Normalize  →  MLP  →  Letter
          21 points     63 numbers     1 of 24
```

Background, skin tone, and lighting all get thrown away. Only the 21 hand points
are left, so the model should hold up better on new signers than a CNN on raw images.

---

## Step 1. Extract landmarks (Colab, one time)

- Run MediaPipe `HandLandmarker` on all 72,000 training images (24 letters × 3,000).
- Upscale each image to about 400 px first. Kaggle images are 200×200, and MediaPipe misses small hands.
- Use a lower detection confidence (about 0.3) during extraction.
- Save the points to Drive, one `.npz` per letter (features, frame numbers, handedness).
  That way nobody has to redo this step.
- Log the detection rate per letter. Images with no hand found are skipped.

Data location: `/content/drive/MyDrive/ASL-interp-Data/`
Output: `/content/drive/MyDrive/ASL-interp-Data/model_b/`

## Step 2. Normalize

This makes every hand the same size and position. For each point $\mathbf{p}_i$,
with $\mathbf{p}_0$ as the wrist:

$$\mathbf{p}_i' = \frac{\mathbf{p}_i - \mathbf{p}_0}{\max_j \lVert \mathbf{p}_j - \mathbf{p}_0 \rVert}$$

- Flip left hands so they look like right hands.
- Subtract the wrist, so the wrist sits at the origin.
- Divide by the farthest point, so hand size doesn't matter.
- Result: 21 points × 3 coordinates = **63 numbers**.

> **Important:** training and the live app must use the exact same `normalize_hand`
> function in `core/features.py`. If they differ even slightly, the live app breaks.

## Step 3. Split the data

| Set | What's in it | Used for |
|---|---|---|
| Train | Kaggle frames 1–2400 per letter | Learning |
| Validation | Kaggle frames 2401–3000 per letter | Tuning and picking the best checkpoint |
| Test | Kaggle "ASL Alphabet Test" (danrasband) | Final score, looked at **once** |

The Kaggle images are video frames, so neighboring frames are near-copies.
We split by frame number, not randomly, so copies can't land on both sides.

The test set is a different person with different backgrounds. It's the only
set that answers the real question: does this work on a new hand?

## Step 4. Train the MLP

- **Layers:** $63 \to 256 \to 128 \to 24$, with ReLU and dropout 0.3
- **Loss:** cross-entropy
- **Optimizer:** Adam, mini-batches
- **Augmentation:** random rotation (±15°), stretching (0.9–1.1), small jitter ($\sigma \approx 0.01$).
  This fakes "different hands."
- **Checkpoint:** save the best one by validation accuracy
- **Seed:** 42
- **Hardware:** CPU is fine. 63 numbers per sample is tiny.

## Step 5. Evaluate

Run on validation and on the test set. Report:

- Top-1 accuracy
- Macro F1
- Confusion matrix (PNG)
- Per-letter accuracy and the 3 most-confused letter pairs
- Detection rate (how often MediaPipe found a hand)
- Average inference time per sample

Everything is written to `results.json` so it drops straight into the report.

## Step 6. Compare with the CNN (Model A)

Run the CNN on the same test set. This comparison is the headline result for the report.

## Step 7. Live demo

- Webcam → MediaPipe → the same `normalize_hand` → MLP
- Only show a letter if confidence is at least 0.7 **and** it wins most of the last 5 frames.
  Otherwise show "no sign."
- Draws the hand points and FPS on screen. Press `q` to quit.

---

## Repo layout (planned)

```
core/
  features.py        normalize_hand + MediaPipe extraction helper
  model.py           LetterMLP, save/load checkpoint
  augment.py         rotation, stretch, jitter
training/
  extract_landmarks.py   Step 1
  train_mlp.py           Step 4
  evaluate.py            Step 5
notebooks/
  model_b_landmarks_mlp.ipynb   Colab front end for Steps 1–5
app/
  live_demo.py       Step 7
tests/               pytest checks for normalize, augment, model, split
```

`core/` holds pure logic and never knows who calls it. The notebook,
the desktop app, and a future website all import from it.
