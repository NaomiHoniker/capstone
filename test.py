from capture import *
import sys

if __name__ == '__main__':
    file_path = ""
    if len(sys.argv) == 2:
        file_path = sys.argv[1]

    gather_data(10, file_path)
