from PIL import Image


# Convert pgm file to vector of grayscale value
def read_img(pgm):
    img = Image.open(pgm)    # read the pgm file
    pixels = img.load()
    vector = []
    for j in range(0, img.size[1]):
        for i in range(0, img.size[0]):
            vector.append(pixels[i, j])  # get row first
    return vector   # output (1, 10304)


# Get train datasets
def load_training_datasets():
    file = "att_faces/s"
    # Get 200 pgm files to create training data (200, 10304).
    training_dataframe = []
    # Get the class label of 200 pgm files (200, 1).
    training_target = []
    for i in range(1, 41):
        for j in range(1, 6):
            vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
            training_target.append([i])
            training_dataframe.append(vector)
    return training_dataframe, training_target


# Get test datasets
def load_test_datasets():
    file = "att_faces/s"
    # Get 200 pgm files to create training data (200, 10304).
    test_dataframe = []
    # Get the class label of 200 pgm files (200, 1).
    test_target = []
    for i in range(1, 41):
        for j in range(6, 11):
            vector = read_img(file + str(i) + "/" + str(j) + ".pgm")
            test_target.append([i])
            test_dataframe.append(vector)
    return test_dataframe, test_target
