import torch
import numpy as np

CKPT_PATH = {
    "siglip-so400m-patch14-384": "/mnt/ckpts/google/siglip-so400m-patch14-384",
    "clip-vit-large-patch14": "/mnt/ckpts/openai/clip-vit-large-patch14"
}

def load_embedding_model(model_name, device="cuda"):
    if model_name == "siglip-so400m-patch14-384":
        # google/siglip-so400m-patch14-384
        from transformers import SiglipProcessor, SiglipModel
        model = SiglipModel.from_pretrained(
            CKPT_PATH[model_name], 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            device_map=device
        )
        processor = SiglipProcessor.from_pretrained(CKPT_PATH[model_name])

        def preprocess(texts=None, imgs=None):
            text_inputs = processor(text=texts, padding="max_length", return_tensors="pt") if texts is not None else None
            img_inputs = processor(images=imgs, padding="max_length", return_tensors="pt") if imgs is not None else None
            return text_inputs, img_inputs

        def predict(texts=None, imgs=None):
            text_inputs, img_inputs = preprocess(texts, imgs)
            with torch.inference_mode():
                if text_inputs is not None:
                    text_inputs = text_inputs.to(device)
                    text_features = model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                else:
                    text_features = None
                if img_inputs is not None:
                    img_inputs = img_inputs.to(device, torch.float16)
                    img_features = model.get_image_features(**img_inputs)
                    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                else:
                    img_features = None
            return text_features, img_features
        
        return predict

    elif model_name == "clip-vit-large-patch14":
        # openai/clip-vit-large-patch14
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(
            CKPT_PATH[model_name], 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            device_map=device
        )
        processor = CLIPProcessor.from_pretrained(CKPT_PATH[model_name])

        def preprocess(texts=None, imgs=None):
            text_inputs = processor(text=texts, padding="max_length", return_tensors="pt") if texts is not None else None
            img_inputs = processor(images=imgs, padding="max_length", return_tensors="pt") if imgs is not None else None
            return text_inputs, img_inputs

        def predict(texts=None, imgs=None):
            text_inputs, img_inputs = preprocess(texts, imgs)
            with torch.inference_mode():
                if text_inputs is not None:
                    text_inputs = text_inputs.to(device)
                    text_features = model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                else:
                    text_features = None
                if img_inputs is not None:
                    img_inputs = img_inputs.to(device)
                    img_features = model.get_image_features(**img_inputs)
                    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                else:
                    img_features = None
            return text_features, img_features
        
        return predict
    
    else:
        raise ValueError(f"Model {model_name} not found")
        

def merge_items(similarity_matrix, threshold=0.9):
    """
    Merge items based on similarity threshold and update the similarity matrix iteratively.
    
    Parameters:
        similarity_matrix (numpy.ndarray): A square 2D similarity matrix.
        threshold (float): The similarity threshold for merging items.

    Returns:
        numpy.ndarray: The final reduced similarity matrix.
        list: A list of groups of merged items.
    """
    n = similarity_matrix.shape[0]
    groups = [{i} for i in range(n)]  # Track groups of merged items

    while True:
        # Find the pair (i, j) with max similarity above the threshold
        max_similarity = -1
        merge_pair = None
        for i in range(len(similarity_matrix)):
            for j in range(i - 1, -1, -1):
                if similarity_matrix[i, j] > max_similarity:
                    max_similarity = similarity_matrix[i, j]
                    merge_pair = (i, j)

        if max_similarity <= threshold:  # Stop if no similarity exceeds the threshold
            break

        i, j = merge_pair

        # Merge items i and j
        new_row = (similarity_matrix[i, :] + similarity_matrix[j, :]) / 2
        new_col = (similarity_matrix[:, i] + similarity_matrix[:, j]) / 2
        new_row = np.delete(new_row, [i, j])
        new_col = np.delete(new_col, [i, j])

        # Remove rows and columns i and j, then add the merged row/column
        similarity_matrix = np.delete(similarity_matrix, [i, j], axis=0)
        similarity_matrix = np.delete(similarity_matrix, [i, j], axis=1)
        similarity_matrix = np.vstack([similarity_matrix, new_row])
        new_col = np.append(new_col, 0)  # Prevent self-similarity
        similarity_matrix = np.hstack([similarity_matrix, new_col[:, None]])

        # Merge the groups
        merged_group = groups[i] | groups[j]
        groups = [group for k, group in enumerate(groups) if k not in (i, j)]
        groups.append(merged_group)

    return similarity_matrix, groups