import argparse
import h5py
import imageio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_num", required=True, type=int, 
        help="Which image to display?")
    parser.add_argument("--dataset_file", required=False, type=str, default="output", 
        help="The name of the output folder")


    args = parser.parse_args()
    if args.dataset_file[-5:] == '.hdf5':
        args.dataset_file = args.dataset_file[:-5]
    if not os.path.exists(args.dataset_file + ".hdf5") or not os.path.isfile(args.dataset_file + ".hdf5"):
        print("File: \'" + args.dataset_file + ".hdf5\' does not exist")
        return

    output = h5py.File(args.dataset_file + ".hdf5", "r")
    image_dataset = output['image_dataset']
    sketch_dataset = output['sketch_dataset']
    print("image data shape: ", image_dataset.shape)
    print("sketch data shape: ", sketch_dataset.shape)
    if args.image_num >= image_dataset.shape[0]:
        print("Image num out of range")
        return
    plt.imshow(image_dataset[args.image_num] / 255)
    plt.show()
    plt.imshow(sketch_dataset[args.image_num] / 255)
    plt.show()

if __name__ == "__main__":
    main()