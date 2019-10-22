import argparse
import h5py
import imageio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_type", required=False, type=str, default="*",
        help="The name of the object to be created (cat, dog, etc).")
    # parser.add_argument("--only_invalid_pose", required=False, type=bool, default=False,
    #     help="Will create a set of the invalid pose type images")
    # parser.add_argument("--include_ambiguous", required=False, type=bool, 
    #     help="Will include the ambiguous images in the dataset")
    # parser.add_argument("--include_context", required=False, type=bool, 
    #     help="Will includethe images with extra environmental context (water ripples, etc)")
    # parser.add_argument("--use_non_uniform_scaling", required=False, type=bool, default=False, 
    #     help="Use the non-uniform scaled images instead of the bounding box scaled images")
    # parser.add_argument("--add_augmentation", required=False, type=bool, default=False, 
    #     help="Add data augmentation")
    parser.add_argument("--flipping_augment", required=False, type=bool, default=False,
        help="Add flipping across y axis to all images for augmentation")
    parser.add_argument("--num_pairs", required=True, type=int, 
        help="How many image/sketch pairs do you want in the dataset?")
    parser.add_argument("--input_location", required=False, type=str,default='./' ,
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
        

    images_data, sketches_data = read_images_and_sketches(args)
    images_data, sketches_data = perform_augmentation(args, images_data, sketches_data)

    output = h5py.File(args.output_file + ".hdf5", "a")
    image_dataset = output.create_dataset("image_dataset", data=images_data,dtype='i8')
    sketch_dataset = output.create_dataset("sketch_dataset", data=sketches_data, dtype='i8')
    print("image data shape: ", image_dataset.shape)
    print("sketch data shape: ", sketch_dataset.shape)

def read_images_and_sketches(args):
    count = 0
    output_images = np.zeros((args.num_pairs,256,256,3))
    output_sketches = np.zeros((args.num_pairs, 256, 256, 3))
    folder_prefix = args.input_location + "/256x256/"
    invalid_ambiguous = set(line.strip() for line in open("./data/sketchy/info/invalid-ambiguous.txt"))
    invalid_context = set(line.strip() for line in open("./data/sketchy/info/invalid-context.txt"))
    invalid_error = set(line.strip() for line in open("./data/sketchy/info/invalid-error.txt"))
    invalid_pose = set(line.strip() for line in open("./data/sketchy/info/invalid-pose.txt"))
    sketch_index_start = 1
    while count < args.num_pairs:
        for file_name_and_loc in glob.glob(folder_prefix + "/photo/tx_000100000000/" + args.object_type + "/*.jpg"):
            output_images[count] = np.array(imageio.imread(file_name_and_loc))
            file_name = file_name_and_loc.split('/')[-1][:-4] #This isolates the file name, and drops the file type
            object_type = file_name_and_loc.split('/')[-2]
            #make sure we dont use an invalid sketch
            sketch_index = sketch_index_start
            sketch_name = file_name + "-" + str(sketch_index)
            while sketch_name in invalid_ambiguous or sketch_name in invalid_context or \
                    sketch_name in invalid_error or sketch_name in invalid_pose:
                sketch_index += 1
                sketch_name = file_name + "-" + str(sketch_index)
            
            output_sketches[count] = np.array(imageio.imread(folder_prefix + "sketch/tx_000100000000/" + 
                    object_type + "/" + sketch_name + ".png"))
            count += 1
            if count >= args.num_pairs:
                break
        sketch_index_start += 1

    return output_images, output_sketches

def perform_augmentation(args, images_data, sketches_data):
    images_copy = images_data
    sketches_copy = sketches_data
    if args.flipping_augment:
        images_copy = np.append(images_copy,np.flip(images_copy,axis=2),axis=0)
        sketches_copy = np.append(sketches_copy, np.flip(sketches_copy,axis=2),axis=0)
    return images_copy, sketches_copy

if __name__ == "__main__":
    main()