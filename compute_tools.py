from modelscope import snapshot_download
from transformers import CLIPProcessor, CLIPModelimport torch
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import entropy
from diffusers.utils import load_image
from transformers import CLIPFeatureExtractor

def process_brightness(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    avg_rgb = np.mean(arr)
    hsv_arr = np.array(img.convert('HSV'))
    h_channel, s_channel, v_channel = hsv_arr[..., 0], hsv_arr[..., 1], hsv_arr[..., 2]
    average_value = np.mean(v_channel) / 255.0
    return average_value

def compute_brightness(count, data_src_folder, data_id):
    data_df = pd.read_csv(f'{data_src_folder}/output_captions/captions_{data_id}.csv')
    print(data_df)
    results = []
    rows = data_df.iterrows() if count == -1 else data_df.head(count).iterrows()
    for _, row in rows:
        image_id = str(row['image_id']).strip()
        original_caption = str(row['caption']).strip()
        image_path = os.path.join(f'{data_src_folder}/output_images_{data_id}', f"{image_id.zfill(6)}.png")
        average_value = process_brightness(image_path)
        print("id,average_value:", _, average_value)
        results.append({
            'image_id': image_id,
            'original_caption': original_caption,
            'prefer_value': average_value,
        })
    return results

def process_realism(image_path,preference_model,type):
    if type == 'realism':
        contrast_captions = [
            ("prefer1", "A real photograph, realistic details and natural lighting." ),
            ("prefer2", "A cartoon image, is a human-created artistic representation, such as an illustration or painting.")
        ]
    image = Image.open(image_path)
    all_scores = {}
    for label, caption in contrast_captions:
        similarity_score = preference_model.score(image, caption)
        all_scores[label] = similarity_score[0]
    diff_value = all_scores["prefer1"] - all_scores["prefer2"]
    return diff_value

def compute_realism(folder, count, data_src_folder, preference_model, src_image_folder):
    results = []
    data_df = pd.read_csv(f'{data_src_folder}/output_captions/captions_{data_id}.csv')
    rows = data_df.iterrows() if count == -1 else data_df.head(count).iterrows()
    for _, row in rows:
        image_id = str(row['image_id']).strip()
        original_caption = str(row['caption']).strip() 
        image_path = os.path.join(f'{data_src_folder}/output_images_{data_id}', f"{image_id.zfill(6)}.png")
        if not os.path.exists(image_path):
            print(f"Warning: figure {image_path} not exist, skip!")
            continue

        diff_value = process_realism(image_path,preference_model,folder)
        print(f"image_id:{image_id},diff_value:{diff_value}")

        results.append({
            'image_id': image_id,
            'original_caption': original_caption,
            'diff_value': diff_value,
            **all_scores
        })

    return results

def process_detail(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    return entropy(hist.flatten()) 

def compute_detail(folder, count, data_src_folder):
    data_df = pd.read_csv(f'{data_src_folder}/output_captions/captions_{data_id}.csv')
    rows = data_df.iterrows() if count == -1 else data_df.head(count).iterrows()
    for _, row in rows:
        image_id = str(row['image_id']).strip()
        original_caption = str(row['caption']).strip() 
        image_path = os.path.join(f'{data_src_folder}/output_images_{data_id}', f"{image_id.zfill(6)}.png")
        if not os.path.exists(image_path):
            print(f"Warning: figure {image_path} not exist, skip!")
            continue

        detail = process_detail(image_path)
        print(f"image_id:{image_id},entropy_value:{detail}")

        results.append({
            'image_id': image_id,
            'original_caption': original_caption,
            'detail': detail,
        })
    return results

    