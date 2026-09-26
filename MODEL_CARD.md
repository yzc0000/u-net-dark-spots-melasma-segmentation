# Model card

## Model description

This repository contains a binary U-Net segmentation model that marks pixels associated with visible dark spots and melasma-like facial pigmentation. The network accepts RGB images resized to 640 × 640 pixels and returns one logit channel.

The released checkpoint uses 64 base channels and bilinear upsampling. It was exported from epoch 45 of the selected training run. Optimizer state was removed from the public artifact; every released model tensor is identical to the selected local checkpoint.

The broader dataset-development effort included more than 5,000 images annotated manually by the project author. Those source images and annotations are not included here because they require separate privacy and licensing review. This figure describes the overall annotation effort and should not be interpreted as the size of the selected checkpoint's held-out evaluation split.

## Intended use

- Educational computer-vision experiments
- Portfolio demonstrations
- Research prototypes that remain under human review

## Out-of-scope use

This model is not a medical device and has not undergone clinical validation. It must not be used to diagnose, grade, or recommend treatment for any condition. Predictions can be wrong, particularly under lighting, camera, skin-tone, makeup, compression, or imaging conditions that differ from the training data.

## Evaluation

The selected checkpoint was evaluated at a 0.5 probability threshold on a local held-out split of 49 images.

| Metric | Result |
| --- | ---: |
| Dice | 0.7286 |
| IoU | 0.5939 |
| Precision | 0.7619 |
| Recall | 0.7525 |
| F1 | 0.7571 |

These results describe one internal split, not an independent or clinical benchmark. The dataset is not distributed in this repository because its images require separate privacy and licensing review.

## Limitations

- Performance may vary across skin tones and pigmentation patterns.
- Resizing to a square can distort the original aspect ratio.
- Fine boundaries and small disconnected regions remain difficult for the model.
- The model does not distinguish causes of pigmentation and cannot establish a diagnosis.
- Dataset provenance and subgroup coverage must be audited before any real-world use.
