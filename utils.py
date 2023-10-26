import os
import csv

def remove_file_or_dir(path_:str):
    try:
        os.remove(path_)
    except OSError:
        pass

def write_list_to_csv(data:list, file_path:str):
    with open(file_path, "w") as file:
        csv_writer = csv.writer(file, delimiter=",")
        csv_writer.writerows(data)
    return file_path