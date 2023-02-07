import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple


import numpy as np
from PIL import Image

def dHash(gray,hash_size):
    #缩放8*8img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    dhash = np.zeros([hash_size, hash_size-1])
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(hash_size):
        for j in range(hash_size-1):
            if gray[i,j]>gray[i,j+1]:
                dhash[i,j]=int(1)
            else:
                dhash[i,j]=int(0)
    return dhash

def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """ 
    Calculate the dhash signature of a given file
    
    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2
    
    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    pil_image = Image.open(image_file).convert("L").resize(
                        (hash_size+1, hash_size),
                        Image.ANTIALIAS)
    pil_image = np.array(pil_image)
    dh = dHash(pil_image,hash_size)
    signature = dh.flatten()

    
    return signature

        
def find_near_duplicates(input_dir: str, threshold: float, hash_size: int, bands: int) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images
    
    Args:
        input_dir: Directory with images to check
        threshold: Images with a similarity ratio >= threshold will be considered near-duplicates
        hash_size: Hash size to use, signatures will be of length hash_size^2
        bands: The number of bands to use in the locality sensitve hashing process
        
    Returns:
        A list of near-duplicates found. Near duplicates are encoded as a triple: (filename_A, filename_B, similarity)
    """
    rows: int = int(hash_size**2/bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]
    
    # Build a list of candidate files in given input_dir
    file_list = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    # Iterate through all files in input directory
    for fh in file_list:
        try:
            signature = calculate_signature(fh, hash_size)
        except IOError:
            # Not a PIL image, skip this file
            continue

        # Keep track of each image's signature
        signatures[fh] = np.packbits(signature)
        
        # Locality Sensitive Hashing
        for i in range(bands):
            signature_band = signature[i*rows:(i+1)*rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(fh)

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i+1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i],hash_bucket[j]])
                        )

    # Check candidate pairs for similarity
    near_duplicates = list()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
                np.unpackbits(signatures[cpa]), 
                np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size**2 - hd) / hash_size**2
        if similarity > threshold:
            near_duplicates.append((cpa, cpb, similarity))
            
    # Sort near-duplicates by descending similarity and return
    near_duplicates.sort(key=lambda x:x[2], reverse=True)
    return near_duplicates





    
input_dir = "D:/prfinal5"
threshold = 0.8
hash_size = 16
bands = 32

try:
    near_duplicates = find_near_duplicates(input_dir, threshold, hash_size, bands)
    if near_duplicates:
        print(f"Found {len(near_duplicates)} near-duplicate images in {input_dir} (threshold {threshold:.2%})")
        for a,b,s in near_duplicates:
            print(f"{s:.2%} similarity: file 1: {a} - file 2: {b}")
    else:
        print(f"No near-duplicates found in {input_dir} (threshold {threshold:.2%})")
except OSError:
    print(f"Couldn't open input directory {input_dir}")
                    


