from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import Subset
from model.generation import image_caption
from model.dataset import clean_caption
import torch
import os


def evaluate_bleu(model, test_data, flick, device="cuda", max_examples=None , confidance = 1):
    model.eval()
    smoother = SmoothingFunction().method4

    all_references = []
    all_candidates = []

    # -----------------------------
    # Get dataset
    # -----------------------------
    if isinstance(test_data, torch.utils.data.DataLoader):
        dataset = test_data.dataset
    else:
        dataset = test_data

    # -----------------------------
    # Avoid evaluating same image 5 times
    # -----------------------------
    visited_images = set()

    with torch.no_grad():

        for i in range(len(dataset)):

            if max_examples is not None and len(all_candidates) >= max_examples:
                break

            if isinstance(dataset, Subset):
                idx = dataset.indices[i]
                base_dataset = dataset.dataset
            else:
                idx = i
                base_dataset = dataset

            image_name = base_dataset.images.iloc[idx]

            # Skip duplicated image
            if image_name in visited_images:
                continue

            visited_images.add(image_name)

            image_path = os.path.join(
                base_dataset.image_path,
                image_name
            )

            # -----------------------------------------
            # Prediction
            # -----------------------------------------
            pred_caption = image_caption(
                model,
                image_path,
                flick.vocab.W2i,
                device=device,
                show_image=False ,
                confidance = confidance 
            )

            pred_caption = clean_caption(pred_caption)
            pred_tokens = flick.vocab.tokinization(pred_caption)

            all_candidates.append(pred_tokens)

            # -----------------------------------------
            # Get ALL captions for this image
            # -----------------------------------------
            refs = base_dataset.x_y[
                base_dataset.x_y["image"] == image_name
            ]["caption"].tolist()

            ref_tokens = []

            for cap in refs:
                cap = clean_caption(cap)
                ref_tokens.append(
                    flick.vocab.tokinization(cap)
                )

            all_references.append(ref_tokens)

            print("=" * 80)
            print("References:")
            for r in ref_tokens:
                print(r)

            print("\nPrediction:")
            print(pred_tokens)

        print("=" * 80)
        print(f"Images evaluated : {len(all_candidates)}")

    bleu1 = corpus_bleu(
        all_references,
        all_candidates,
        weights=(1, 0, 0, 0),
        smoothing_function=smoother
    )

    bleu4 = corpus_bleu(
        all_references,
        all_candidates,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoother
    )

    return bleu1, bleu4