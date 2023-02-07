import os
import numpy as np
from mpi4py import MPI
import random as rd
import matplotlib.image as mpimg
def read_img(path):
    return np.round(mpimg.imread(path) / 255)

def convert_img(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if sum(img[i][j]) == 1 or sum(img[i][j]) == 2:
                img[i][j] = np.array([0., 0., 0.])

def to_2D_matrix(img):
    matrix = []
    for i in img:
        array = []
        for j in i:
            array.append(j[0])
        matrix.append(array)
    return np.asarray(matrix)

def to_matrix(path):
    img = read_img(path)
    convert_img(img)
    return to_2D_matrix(img)

def shingling(img):
    k = 5
    tokens = []
    for i in range(img.shape[0] - k):
        for j in range(img.shape[1] - k):
            img_pc = img[i:i+k, j:j+k].flatten()
            tokens.append(str(img_pc))
    return tokens

def min_hash(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1]
    num = 1000
    perm = list(range(row))
    sig = np.zeros([num, col])
    for n in range(num):
        rd.shuffle(perm)
        for i in range(col):
            index = 1
            for j in perm:
                if matrix[j, i] == 1:
                    sig[n, i] = index
                    break
                else:
                    index += 1
    return sig

def LSH(sig, band):
    given = np.split(sig[:,0], band)
    candidates = []


    for i in range( sig.shape[1]):
        col = np.split(sig[:, i], band)
        for j in range(band):
            if np.array_equal(given[j], col[j]):
                candidates.append(i)
                break
    return candidates

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()

    # Load Given Image
    img_given = to_matrix('1547.jpg')
    shingles_given = shingling(img_given)
    # Load Images
    pwd = '/gpfs/projects/AMS598/Projects2022/project5_final/group4/prfinal52/'
    img_names = np.array(os.listdir(pwd))
    # Find All Shingles
    imgs = [] + [('1547.jpg', shingles_given)]
    all_shingles = [] + shingles_given
    for i in np.array_split(img_names, p)[my_rank]:
        name = i
        mat = to_matrix(pwd + i)
        # Calculate Shingles
        shingles = shingling(mat)
        imgs.append((name, shingles))
        # Add Shingles
        all_shingles += shingles
        # Delete Duplicate Shingles
        all_shingles = list(set(all_shingles))
    # Calculates Boolean Matrices
    bool_mat = []
    for i in imgs:
        name = i
        shingles = i[1]
        img = []
        for j in all_shingles:
            if j in shingles:
                img.append(1)
            else:
                img.append(0)
        bool_mat.append(img)
    bool_mat = np.transpose(np.asarray(bool_mat))
    # Apply Min-Hashing
    signature = min_hash(bool_mat)
    # Apply Local-Sensitive-Hashing
    bands = 250
    sim_imgs = LSH(signature, bands)
    
    fn=[]
    for i in sim_imgs:
        fn.append(imgs[i][0])
    result = comm.gather(fn)

    if my_rank == 0:
        al = []
        for i in result:
            al += i
        al = list(set(al))
        print(al)

if __name__ == '__main__':
    main()
