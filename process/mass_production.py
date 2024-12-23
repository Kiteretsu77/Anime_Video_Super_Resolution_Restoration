import os, time, sys, math

# import videos from local folder
from config import configuration


# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from process.single_video import parallel_process


def check_existence(dir, create=False):
    if not os.path.exists(dir):
        print("This " + dir + " file folder doesn't exist!")
        if not create:
            os._exit(0)
        else:
            os.mkdir(dir)



def mass_process(input_folder_dir, output_dir_parent):
    
    check_existence(input_folder_dir)
    check_existence(output_dir_parent, create=True)


    # If use_rename, we will rename the video easily
    if configuration.use_rename:
        video_lists = os.listdir(input_folder_dir)
        video_num = len(video_lists)
        total_idx_length = len(str(video_num))
        for idx, filename in enumerate(sorted(video_lists)):
            idx = idx + 1
            format = filename.split('.')[-1]
            # The new name is usally 000d
            new_name = "0"*(total_idx_length-len(str(idx))) + str(idx) + "." + format

            input_dir = os.path.join(input_folder_dir, filename)
            new_dir = os.path.join(input_folder_dir, new_name)
            os.rename(input_dir, new_dir)


    print("All files begin")
    for _, filename in enumerate(sorted(os.listdir(input_folder_dir))):
        lists = filename.split('.')
        target_name = ''.join(lists[:-1])


        # Find the name of input and ouput
        input_dir = os.path.join(input_folder_dir, filename)
        output_name = os.path.join(output_dir_parent, target_name + "_processed.mp4")
        print("We are super resolving {} and we will save it at {}".format(input_dir, output_name))


        # Process the video
        parallel_process(input_dir, output_name, parallel_num=configuration.process_num)


        # After Processing
        print("After finish one thing, sleep for a moment!")
        time.sleep(5)
    
        

