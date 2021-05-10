import numpy as np
import cv2
import struct
import os
import argparse

TYPE_IMAGE = 2051
TYPE_LABEL = 2049

filename = "train-images-idx3-ubyte"
output_folder = "train-images-idx3-ubyte"


def get_command_line_args():
    arg_parser = argparse.ArgumentParser(description="Extract MINST Handwritten Digit images, "
                                                     "more info at http://yann.lecun.com/exdb/mnist/")
    arg_parser.add_argument("filename", help="File name to extract")
    arg_parser.add_argument("output_folder", help="Folder to extract files in")
    return arg_parser.parse_args()


def extract_image_data(data, count, width, height):
    size = width * height
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(count):
        start = i * size
        end = start + size
        image = list(struct.unpack("B" * size, data[start:end]))
        image = np.array(image, dtype=np.uint8).reshape((width, height))
        output_filename = os.path.join(output_folder, str(i + 1).zfill(5))
        cv2.imwrite(f"{output_filename}.bmp", image)


def parse_image(data):
    count = struct.unpack(">i", data[:4])[0]
    width = struct.unpack(">i", data[4:8])[0]
    height = struct.unpack(">i", data[8:12])[0]
    extract_image_data(data[12:], count, width, height)


def extract_label_data(data, count):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_filename = os.path.join(output_folder, "00000-labels.txt")

    labels = []
    for i in range(count):
        labels.append(str(data[i]))
    with open(output_filename, "w") as f:
        f.write('\n'.join(labels))


def parse_labels(data):
    count = struct.unpack(">i", data[:4])[0]
    extract_label_data(data[4:], count)


def main():
    with open(filename, mode="rb") as f:
        data = f.read()
        magic_number = struct.unpack(">i", data[:4])[0]
        if magic_number == TYPE_IMAGE:
            print("File Type: Image Data Set")
            parse_image(data[4:])
        elif magic_number == TYPE_LABEL:
            print("File Type: Labels Data Set")
            parse_labels(data[4:])
        else:
            raise Exception("Wrong File Type")


if __name__ == '__main__':
    args = get_command_line_args()
    filename = args.filename
    output_folder = args.output_folder

    main()
