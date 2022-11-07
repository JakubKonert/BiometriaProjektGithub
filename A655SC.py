import os
import glob

from PIL import Image
import PIL

from BosonFrame import *
import numpy as np


def getLong(bajt1, bajt2, bajt3, bajt4):
    lsb1 = bajt1
    lsb2 = bajt2
    lsb3 = bajt3
    msb = bajt4

    if lsb1 < 0:
        lsb1 = 256 + lsb1
    if lsb2 < 0:
        lsb2 = 256 + lsb2
    if lsb3 < 0:
        lsb3 = 256 + lsb3
    if msb < 0:
        msb = 256 + msb

    wynik = (msb << 24) + (lsb3 << 16) + (lsb2 << 8) + lsb1

    return wynik


def getInt(bajt1, bajt2):
    lsb = bajt1
    msb = bajt2

    if lsb < 0:
        lsb = lsb + 256
    if msb < 0:
        msb = msb + 256

    wynik = (msb << 8) + lsb
    return wynik


def pokaz_imgA655SC_nr(nazwapliku, nr_klatki):

    # wyswietla klatke z sekwencji termograficznej seq, z kamery FLIR A320G
    # przykład wywołania: pokaz_imgA320_nr('IBM0029.SEQ',10)

    fid = open(nazwapliku, 'rb')

    sekwencja3 = np.fromfile(fid, dtype=np.ubyte)

    l_bajtow = int(os.stat(nazwapliku).st_size)
    l_klatek = (l_bajtow - 617180) / 617052 + 1

    offset = 2781

    if nr_klatki > 0:
        # czy na pewno ten offset się zgadza??
        offset = (nr_klatki - 2) * 617052 + 617180 + 2653 - 1

    macierz_2D = np.zeros((480, 640))
    seq = []
    for nr_y in range(480):
        for nr_x in range(640):
            wartosc = sekwencja3[offset]

            wartosc = getInt(sekwencja3[offset], sekwencja3[offset + 1])
            macierz_2D[nr_y, nr_x] = wartosc
            offset = offset + 2
    seq.append(macierz_2D)

    return seq


def get_all_frames_a655sc(nazwapliku):
    # wczytuje wszystkie klatki z pliku

    fid = open(nazwapliku, 'rb')

    sekwencja3 = np.fromfile(fid, dtype=np.ubyte)

    l_bajtow = int(os.stat(nazwapliku).st_size)
    l_klatek = int((l_bajtow - 617180) / 617052 + 1)

    offset = 2781

    nr_klatki = 0

    seq = []
    for k in range(1, l_klatek):
        nr_klatki = k

        if nr_klatki > 0:  # nie zapisujemy pierwszej klatki
            offset = (nr_klatki - 2) * 617052 + 617180 + \
                2653 - 1  # -1 bo matlab numeruje od 1

            macierz_2D = np.zeros((480, 640))

            for nr_y in range(480):
                for nr_x in range(640):
                    wartosc = sekwencja3[offset]

                    wartosc = getInt(
                        sekwencja3[offset], sekwencja3[offset + 1])
                    macierz_2D[nr_y, nr_x] = wartosc
                    offset = offset + 2

            seq.append(macierz_2D)
            print(f"Nr klatki: {nr_klatki}")
    print(f"Nazwa pliku: {nazwapliku}")
    return seq


def get_one_frame_a655sc(nazwapliku, numer_klatki):
    # wczytuje wszystkie klatki z pliku

    fid = open(nazwapliku, 'rb')

    sekwencja3 = np.fromfile(fid, dtype=np.ubyte)

    l_bajtow = int(os.stat(nazwapliku).st_size)
    l_klatek = int((l_bajtow - 617180) / 617052 + 1)

    offset = 2781

    nr_klatki = numer_klatki

    seq = []

    if nr_klatki > 0:  # nie zapisujemy pierwszej klatki
        offset = (nr_klatki - 2) * 617052 + 617180 + \
            2653 - 1  # -1 bo matlab numeruje od 1

        macierz_2D = np.zeros((480, 640))

        for nr_y in range(480):
            for nr_x in range(640):
                wartosc = sekwencja3[offset]

                wartosc = getInt(sekwencja3[offset], sekwencja3[offset + 1])
                macierz_2D[nr_y, nr_x] = wartosc
                offset = offset + 2

        seq.append(macierz_2D)

    return seq


if __name__ == "__main__":
    # execute only if run as a script

    # filename = "DaneFlir/Rec-000241.seq"
    # dataset = pokaz_imgA655SC_nr(filename, 10)

    # dataset_array = np.stack(dataset[0])

    # to save npz files use save_frame_to_file() from BosonFrame
    # save_to_file = "A655SC_video_" + file
    # save_to_file = save_to_file[:-4]
    #
    # save_frame_sequence_to_file(dataset, save_to_file, 1)

    # to save npz frame as jpg file
    # image = np.interp(dataset_array, (dataset_array.min(), dataset_array.max()), (0, 255))
    # image = np.uint8(image)
    # im = Image.frombytes("L", image.T.shape, image, "raw", "L")
    # im = im.save("klatka_10.jpg")

    filesPath = glob.glob("DaneFlir/*.seq")
    counter = 1250
    for filePath in filesPath:
        frames = get_all_frames_a655sc(filePath)
        for id,frame in enumerate(frames):
            dataset = frame
            dataset_array = np.stack(dataset)
            image = np.interp(dataset_array, (dataset_array.min(), dataset_array.max()), (0, 255))
            image = np.uint8(image)
            im = Image.frombytes("L", image.T.shape, image, "raw", "L")
            im = im.save(f"KlatkaNR{counter}.jpg")
            counter += 1
