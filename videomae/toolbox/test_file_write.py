import os
import sys
import shutil
import time
import threading


def  write_to_file():

    with open("/mnt/shuang/Data/ego4d/preprocessed_data/temp_write.log", "a+") as fp:

        fp.write("test\n")

def move_log_file():

    with open("./inst_write.log", "a+") as fp:

        fp.write("test\n")

    shutil.copy("./inst_write.log", "/mnt/shuang/Data/ego4d/preprocessed_data/inst_write.log")

def thread_worker():
    
    time.sleep(10)

def main():
    
    # fp_const = open("/mnt/shuang/Data/ego4d/preprocessed_data/const_write.log", "a+")

    pool = []
    for i in range(4):
        thread = threading.Thread(target=thread_worker)
        thread.start()
        pool.append(thread)

    try:
        print("waiting")
        while True:
            continue
    except KeyboardInterrupt as e:
        print("keyboard interruption!")
        sys.exit()

    # for i in range(1000):
    #     fp_const.write("test\n")
    #     write_to_file()
    #     move_log_file()
    #     time.sleep(0.1)



if __name__ == "__main__":
    main()