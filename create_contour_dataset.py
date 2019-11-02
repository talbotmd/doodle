import argparse
import h5py
import imageio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random
import scipy.misc
from PIL import Image
import PIL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_type", required=False, type=str, default="*",
        help="The name of the object to be created (cat, dog, etc).")
    # parser.add_argument("--num_pairs", required=True, type=int, 
    #     help="How many image/sketch pairs do you want in the dataset?")
    parser.add_argument("--input_location", required=False, type=str, default='../contour' ,
        help="relative path to the Sketchy Dataset 256x256 folder")
    parser.add_argument("--output_file", required=False, type=str, default="output", 
        help="The name of the output folder")

    args = parser.parse_args()

    if args.output_file[-5:] == '.hdf5':
        args.output_file = args.output_file[:-5]

    if os.path.exists(args.output_file + ".hdf5") and os.path.isfile(args.output_file + ".hdf5"):
        user_input = input("Output file: \'" + args.output_file + ".hdf5\' already exists, would you like to overwrite it? (y/n)")
        if user_input != 'y' and user_input.lower() != 'yes':
            print("Ok, exiting now")
            return
        os.remove(args.output_file + ".hdf5")

    create_dataset(args)
    
def create_dataset(args):
    file_list = glob.glob(args.input_location + "/image/*.jpg")
    output_images = np.zeros((1000,256,256,3),dtype='i8')
    output_sketches = np.zeros((1000,256,256,3),dtype='i8')
    counter = 0
    max_image_size = 1000

    output = h5py.File(args.output_file + ".hdf5", "a")
    image_dataset = output.create_dataset("image_dataset", (1,256,256,3),dtype='i8',compression='gzip', maxshape=(None,None,None,None,))
    sketch_dataset = output.create_dataset("sketch_dataset", (1,256,256,3), dtype='i8',compression='gzip', maxshape=(None,None,None,None,))
    start = 0
    end = 0

    for file_name_and_loc in file_list:
        image = np.array(imageio.imread(file_name_and_loc))

        scale_factor = 256/min(image.shape[0], image.shape[1])
        larger_image = np.array(Image.fromarray(image).resize((int(scale_factor*image.shape[1])+1,int(scale_factor*image.shape[0])+1), PIL.Image.BICUBIC),dtype='i8')
        square_image = larger_image[0:256,0:256,:]

        file_name = file_name_and_loc.split('/')[-1][:-4] #This isolates the file name, and drops the file type
        for i in range(5):
            sketch_black = np.array(imageio.imread(args.input_location + "/sketch-rendered/width-1/"+file_name+"_0"+str(i+1)+".png"))
            sketch = np.zeros_like(image)
            sketch[:,:,0] = sketch_black
            sketch[:,:,1] = sketch_black
            sketch[:,:,2] = sketch_black

            larger_sketch = np.array(Image.fromarray(sketch).resize((int(scale_factor*sketch.shape[1])+1,int(scale_factor*sketch.shape[0])+1), PIL.Image.BICUBIC),dtype='i8')
            square_sketch = larger_sketch[0:256,0:256,:]
            
            output_images[counter] = square_image
            output_sketches[counter] = square_sketch
            counter += 1
            end += 1

            if counter >= max_image_size:
                print("saving: ", end)
                image_dataset.resize(end,axis=0)
                image_dataset[start:end] = output_images
                sketch_dataset.resize(end,axis=0)
                sketch_dataset[start:end] = output_sketches
                start = end
                output_images = np.zeros((max_image_size,256,256,3),dtype='i8')
                output_sketches = np.zeros((max_image_size,256,256,3),dtype='i8')
                counter = 0
        
        
        if image.shape[0] > image.shape[1]:
            square_image = larger_image[-257:-1,0:256,:]
        else:
            square_image = larger_image[0:256,-257:-1,:]
        
        file_name = file_name_and_loc.split('/')[-1][:-4] #This isolates the file name, and drops the file type
        for i in range(5):
            sketch_black = np.array(imageio.imread(args.input_location + "/sketch-rendered/width-1/"+file_name+"_0"+str(i+1)+".png"))
            sketch = np.zeros_like(image)
            sketch[:,:,0] = sketch_black
            sketch[:,:,1] = sketch_black
            sketch[:,:,2] = sketch_black

            larger_sketch = np.array(Image.fromarray(sketch).resize((int(scale_factor*sketch.shape[1])+1,int(scale_factor*sketch.shape[0])+1), PIL.Image.BICUBIC),dtype='i8')
            
            if image.shape[0] > image.shape[1]:
                square_sketch = larger_sketch[-257:-1,0:256,:]
            else:
                square_sketch = larger_sketch[0:256,-257:-1,:]
            
            output_images[counter] = square_image
            output_sketches[counter] = square_sketch
            counter += 1
            end += 1

            if counter >= max_image_size:
                print("saving: ", end)
                image_dataset.resize(end,axis=0)
                image_dataset[start:end] = output_images
                sketch_dataset.resize(end,axis=0)
                sketch_dataset[start:end] = output_sketches
                start = end
                output_images = np.zeros((max_image_size,256,256,3),dtype='i8')
                output_sketches = np.zeros((max_image_size,256,256,3),dtype='i8')
                counter = 0
                
        


if __name__ == "__main__":
    main()