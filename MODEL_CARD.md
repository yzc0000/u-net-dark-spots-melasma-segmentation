# Model card

## Model description

This model performs binary semantic segmentation of visible dark spots and melasma-like pigmentation in facial images. It accepts 640 × 640 RGB input and produces a single-channel segmentation map.

The network uses a U-Net encoder-decoder architecture with 64 base channels, bilinear upsampling, and skip connections between corresponding resolution levels.

## Data

Dataset development included the manual annotation of more than 5,000 images. The annotated dataset is not distributed with the repository.

## Evaluation

Evaluation on a 49-image test set at a probability threshold of 0.5 produced the following results:

| Metric | Result |
| --- | ---: |
| Dice | 0.7286 |
| IoU | 0.5939 |
| Precision | 0.7619 |
| Recall | 0.7525 |
| F1 | 0.7571 |

These values measure performance on the project test set and do not represent independent clinical validation.

## Intended use

- Educational computer-vision experiments
- Segmentation research and benchmarking
- Human-reviewed prototypes

## Out-of-scope use

The model is not a medical device and must not be used to diagnose, grade, or recommend treatment for any condition. Its output requires human interpretation and may be incorrect.

## Limitations

- Performance may vary across skin tones and pigmentation patterns.
- Lighting, camera settings, makeup, compression, and image quality can affect predictions.
- Resizing images to a square can distort their original aspect ratio.
- Fine boundaries and small disconnected regions remain difficult to segment.
- The model does not determine the cause of pigmentation.
- Dataset provenance and subgroup coverage require further evaluation before real-world use.
