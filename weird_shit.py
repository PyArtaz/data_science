import csv
import math
import os


# Decode a given csv file and return three lists with the x, y and z coordinates
def decode_csv(test_file):
    x = []
    y = []
    z = []
    with open(test_file) as file:
        csv_reader_object = csv.reader(file)
        for row in csv_reader_object:
            for cell in row:
                platzhalter = str(cell)
                zahl = platzhalter.strip().split(' ')
                for letter in zahl:
                    if letter == "'":
                        del zahl["'"]
                x.append(float(zahl[0]))
                y.append(float(zahl[1]))
                z.append(float(zahl[2]))
    return x, y, z


# Measure the distance between Node 0 and 1 from a file as a normalizing
# parameter for all other csv_files to make them comparable to our KI
def measure_distance(file):
    x, y, z = decode_csv(file)
    distance = math.sqrt((x[0] - x[1]) * 2 + (y[0] - y[1]) * 2 + (z[0] - z[1]) ** 2)
    return distance


# Measure a factor which makes our coordinates comparable
def measure_scaling_factor(scaling_file, test_file):
    factor = measure_distance(scaling_file) / measure_distance(test_file)
    return factor


# Create a shifted coordinate system with the origin at node 0 and return the new
# coordinates x_new, y_new and z_new normalized by a factor
def new_coordinate_system(scaling_file, test_file):
    x_new = []
    y_new = []
    z_new = []
    x, y, z = decode_csv(test_file)
    factor = measure_scaling_factor(scaling_file, test_file)
    for x_coordinate in x:
        x_new.append((x[0] - x_coordinate) * factor)

    for y_coordinate in y:
        y_new.append((y[0] - y_coordinate) * factor)

    for z_coordinate in z:
        z_new.append((z[0] - z_coordinate) * factor)
    return x_new, y_new, z_new


# Measures the new csv_files datas and saves them in a new csv_file for all pictures
def write_existing_csv(scaling_file, test_file):
    x_new, y_new, z_new = new_coordinate_system(scaling_file, test_file)
    data = [x_new, y_new, z_new]
    with open(test_file, 'w') as Test_csv:
        writer = csv.writer(Test_csv)
        writer.writerows(data)


# Check if a file path exists
def check_path_exists(i, j, m):
    if not os.path.exists(
            r"/Users/maximilianruhl/PycharmProjects/xyz_plotter/CSV_files" + str(j) + "/" + str(i) + "" + str(
                    j) + "" + str(
                    m) + "_cam1.mpg"):
        return False
    else:
        return True


# Return the path of the next file
def get_path_name(i, j, m):
    path = r"/Users/maximilianruhl/PycharmProjects/xyz_plotter/CSV_files" + str(j) + "/" + str(i) + "" + str(
        j) + "" + str(
        m) + "_cam1.mpg"
    return path


# Change all CSV_files
def change_all_csv(scalling_file):
    i = 1
    j = 1
    # j muss die ordner bis 35 durchzählen
    while j <= 1:
        # i muss in den Ordnern bis 26 Zählen
        while i <= 26:
            m = 1
            if check_path_exists(i, j, m):
                path = get_path_name(i, j, m)
                write_existing_csv(scalling_file, path)
                print("Hallo")
            m = m + 1
            if check_path_exists(i, j, m):
                path = get_path_name(i, j, m)
                write_existing_csv(scalling_file, path)
                print("Hallo")
            i = i + 1
        i = 1
        j = j + 1
