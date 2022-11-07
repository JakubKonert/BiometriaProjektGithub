import numpy as np
import os
import matplotlib.pyplot as plt
from boson_constants import *
import cv2

LUT = np.zeros(65536).astype(np.float)

LUT_R = np.zeros(256).astype(np.uint8)
LUT_G = np.zeros(256).astype(np.uint8)
LUT_B = np.zeros(256).astype(np.uint8)

def save_frame_to_file(data, file_name, compress=True):

    if compress:
        file_path = os.path.join(FOLDER_LABELS, file_name+'.npz')
        np.savez_compressed(file_path, data)
        print("Frame saved compressed")
    else:
        file_path = os.path.join(FOLDER_LABELS, file_name+'.npy')
        np.save(file_path, data)
        print("Frame saved uncompressed")



def read_frame_from_file(file_name, compressed=True, folder=FOLDER_LABELS):
    file_path = os.path.join(folder, file_name)
    if compressed:
        files = np.load(file_path)
        frame_data = []
        for key in files.keys():
            frame_data.append(files[key])
        frame_data = np.array(frame_data)
        # print(frame_data.shape)
        if len(frame_data.shape) > 3:
            frame_data = frame_data.reshape((frame_data.shape[1], frame_data.shape[2], frame_data.shape[3]))
        print("Read compressed frame, shape: ", frame_data.shape)

    else:
        frame_data = np.load(file_path)
        print("Read uncompressed frame, shape: ", frame_data.shape)

    if len(frame_data.shape) > 2:
            print("Sequance of images read...")

    return frame_data

def save_frame_sequence_to_file(data, file_name, compress=True):
    if compress:
        file_path = os.path.join(FOLDER_LABELS, file_name+'.npz')
        np.savez_compressed(file_path, np.array(data))
        print("Frame saved compressed")
    else:
        file_path = os.path.join(FOLDER_LABELS, file_name+'.npy')
        np.save(file_path, np.array(data))
        print("Frame saved uncompressed")


def gen_LUT():
    #LUT = np.zeros(65536).astype(np.float)
    global LUT
    for i in range(len(LUT)):
        # print(LUT[i])
        reminder = i & 3
        # print(reminder)
        LUT[i] = i >> 2
        LUT[i] = LUT[i]+reminder * 0.25

# Q14.2 format - last two bits for reminder
# 00 -> 0.0
# 01 -> 0.25
# 10 -> 0.5
# 11 -> 0.75
# 21717 -> 5429.25
# 22254 -> 5563.5
# 21369 -> 5342.25
# 22544 -> 5636.0
def convert_14_2_format(q14_2_value):
    value = 0.0
    reminder = q14_2_value & 3
    # print(reminder)
    value = q14_2_value >> 2
    # print(value)
    value = value+reminder*0.25
    # print(value)

    return value

"""
Convert original frame data2 from stored integer to Q14.2

"""
def convert_frame(data):
    global LUT
    if np.all(np.mod(data, 1) == 0):
        #indices = range(len(LUT))
        if LUT[int(len(LUT)/2)] == 0:
            print("Generating LUT ...")
            gen_LUT()

        # if not np.all(np.mod(data2, 1) == 0):
        #     data2 = (data2 * 4).astype(int)
        #     print("Data converted: ", data2)

        #lut_dict = dict(zip(indices, LUT))
        #print(lut_dict)
        return LUT[data]
    else:
        return data


def get_clut(name):

    global LUT_R, LUT_G, LUT_B

    clut = plt.get_cmap(name)
    indexes = []
    for i in range(256):
        indexes.append(i/255.0)
    lut = np.round(np.array(clut(indexes))*255, 0).astype(np.int)
    lut_t = (lut.transpose()[0:3])
    LUT_R = np.copy(lut_t[0])
    LUT_G = np.copy(lut_t[1])
    LUT_B = np.copy(lut_t[2])

    return LUT_R, LUT_G, LUT_B

def convert_clut(data):
    global LUT_R, LUT_G, LUT_B
    # gen CLUT first
    # get_clut('bone')
    return [LUT_R[data], LUT_G[data], LUT_B[data]]

def convert_clut_for_display(data, f_size, flatten=False):

    global LUT_R, LUT_G, LUT_B
    rgb_data = np.array([LUT_R[data], LUT_G[data], LUT_B[data]]).astype(np.uint8)

    rgb_data_t = rgb_data.transpose().astype(np.uint8)
    # print("SIZE: ", f_size)
    if flatten:
        rgb_data = rgb_data_t.flatten()
    else:
        rgb_data = rgb_data_t.reshape((int(f_size[0]), int(f_size[1]), 3)).astype(np.uint8)
    return rgb_data


def tests():
    data = read_frame_from_file('boson_sample_2')
    print(data.shape)
    print("Data (mean, std, min, max): ", np.mean(data), np.std(data), np.min(data), np.max(data))
    convert_14_2_format(21369)
    convert_14_2_format(22544)
    print(convert_frame(data))


if __name__ == '__main__':
    # tests()
    # convert_14_2_format(21369)
    # convert_14_2_format(22254)
    a = convert_14_2_format(22143)
    print(a)