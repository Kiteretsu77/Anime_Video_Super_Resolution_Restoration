import collections
import threading, sys, cv2
from random import uniform
from multiprocessing import Queue
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
from time import time as ttime, sleep
import os, time
import numpy as np
from threading import Lock
s_print_lock = Lock()


# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from process.crop_utils import crop4partition, crop4partition_SR, combine_partitions_SR


class UpScalerMT(threading.Thread):
    
    def __init__(self, id, inp_q, res_q, model, p_sleep, nt):
        '''
            Multi-Thread Processing
        '''
        # threading.Thread.__init__(self)
        super(UpScalerMT, self).__init__()
        self.id = id
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.nt = nt
        self.p_sleep = p_sleep

        #下面几个是用于时间统计的
        self.cost_list = []
        self.total_cost_time = 0
        self.total_counter_ = 0


    def __del__(self):
        '''
            Report time 
        '''
        if self.total_counter_ != 0:
            with s_print_lock:
                print("UpScalerMT #" + str(self.id) + " report:")
                # print("\tFull Exe Cost: " + str(round(sum(self.cost_list), 2)) + "s on " + str(len(self.cost_list)) + " frames and in average " + str(round(sum(self.cost_list) / len(self.cost_list), 4 )) + "s")
                print("\tFull Exe Cost: " + str(round(self.total_cost_time, 2)) + "s on " + str(
                    self.total_counter_) + " frames and in average " + str(
                    round(self.total_cost_time / self.total_counter_, 4)) + "s \n")

    def execute(self, np_frame, position):
        ''' Send data(frames) to the model
        '''
        return self.model(np_frame, position)


    def inference(self, tmp):
        '''
        Args:
            tmp (set): frame_idx (int), position (int, 0|1|2|3), np_frame (numpy)
        '''
        frame_idx, position, np_frame = tmp

        
        ####################### Neural Network Model Execuation ###########################
        full_exe_start_time = ttime()

        try:
            res = self.execute(np_frame, position)

        except Exception:
            print("Exception! Execute Exception problem, try to execute again")
            res = self.execute(np_frame, position)

        full_exe_end_time = ttime()
        full_exe_time_spent = full_exe_end_time - full_exe_start_time
        ####################################################################################

        self.total_cost_time += full_exe_time_spent
        self.total_counter_ += 1
        if self.nt > 1:
            # sleep adjustment, Must have here else there may be bug
            sleep(uniform(self.p_sleep[0], self.p_sleep[1]))

        return (frame_idx, position, res)


    def run(self):
        while True:
            tmp = self.inp_q.get()      # frame_idx (int), position (int, 0|1|2|3), np_frame (numpy)
            if tmp == None:
                # We can only break (end) the loop when all inputs are processed; else, we need to keep the loop
                break

            if self.res_q.full():
                print("Exception!!! res_q is full, Needs more optimization! Will pause some time!")
                # sleep(1)
            output = self.inference(tmp)
            self.res_q.put(output)



class VideoUpScaler(object):
    def __init__(self, configuration, model_full_name, model_partition_name, process_id = 0):
        '''
            Args:
            configuration (class): all config info we need
            model_full_name (str): the full model name
            model_partition_name (str): the parition model name
            process_id (int): the processid this class belongs to
        '''

        ###################################### Important Params ##################################################################################
        self.writer = None    # Used in Call function
        self.rescale_factor = configuration.rescale_factor
        self.scale = configuration.scale
        self.model_name = configuration.model_name
        self.encode_params = configuration.encode_params
        self.decode_sleep = configuration.decode_sleep
        self.full_model_num = configuration.full_model_num
        self.nt = configuration.nt
        self.now_idx = 0            # Ever frame before this index is encoded by ffmpeg writer
        self.loop_counter = 0
        self.total_frame_number = 0
        self.decode_fps = configuration.decode_fps
        self.height = None
        self.process_id = process_id
        print("This process id is ", self.process_id)


        self.max_cache_loop = (self.nt + self.full_model_num) * 200     # The is for bug report purpose (should be less thantotal frame num)
        #################################################################################################################################

        ################################### Load model ##################################################################################
        # TODO: we should support more than float16 cases
        # Model full and partition weight path setup
        weight_path_full_path, weight_path_partition_path = "", ""

        if configuration.use_tensorrt:
            weight_path_partition_path = os.path.join(configuration.weights_dir, configuration.model_name, 'trt_' + model_partition_name + '_float16_weight.pth')
            assert(os.path.exists(weight_path_partition_path))
            if self.full_model_num != 0:
                weight_path_full_path = os.path.join(configuration.weights_dir, configuration.model_name, 'trt_' + model_full_name + '_float16_weight.pth')
                assert(os.path.exists(weight_path_full_path))

        #################################################################################################################################

        ######################## Similar frame optimization #######################
        self.skip_counter_ = 0
        self.reference_frame = [None, None, None]
        self.reference_idx = [-1, -1, -1]
        self.parition_processed_num = 0
        self.MSE_range = configuration.MSE_range
        self.Max_Same_Frame = configuration.Max_Same_Frame
        ############################################################################

        ########################## Image Crop & Momentum ##################################################################################
        self.pixel_padding = configuration.pixel_padding
        self.full_frame_cal_num = 0
        self.momentum_skip_crop_frame_num = configuration.momentum_skip_crop_frame_num
        self.time2switchFULL = 0                # This counter presents number of frames we need to use momentum
        self.momentum_used_times = 0            # This counter record how many processes are processed in full due to the momentum mechanism
        self.momentum_reference_size = 3        # Momentum Queue size (Use momentum when this number of value in the queue is True)
        self.momentum_reference = collections.deque([False]*self.momentum_reference_size, maxlen=self.momentum_reference_size)
        ###################################################################################################################################

        ############################### MultiThread And MultiProcess ######################################################
        # Full Frame
        self.inp_q_full = None
        if self.full_model_num != 0:
            Full_Frame_Queue_size = int( (self.full_model_num + 1) * configuration.Queue_hyper_param//2)
            self.inp_q_full = Queue(Full_Frame_Queue_size) # queue of full frame
            print("Total FUll Queue size is ", Full_Frame_Queue_size)

        # Sub Frame 
        self.inp_q = None
        if self.nt != 0:
            Divided_Block_Queue_size = int( (self.nt) * configuration.Queue_hyper_param)
            self.inp_q = Queue(Divided_Block_Queue_size)  # queue of partition frame
            print("Total Divided_Block_Queue_size is ", Divided_Block_Queue_size)
        
        # In total
        res_q_size = int( (self.nt + self.full_model_num) * configuration.Queue_hyper_param)
        self.res_q = Queue(res_q_size)  # Super-Resolved Frames Cache
        print("res_q size is ", res_q_size)
        self.idx2res = collections.defaultdict(dict)
        ####################################################################################################################


        ############################# Model Preparation ####################################################################
        if configuration.model_name == "Real-ESRGAN":
            from Real_ESRGAN.uprrdb_main import RealESRGAN_Scalar
            NN_model = RealESRGAN_Scalar
        elif configuration.model_name == "Real-CUGAN":
            from Real_CuGAN.upcunet_main import RealCuGAN_Scalar
            NN_model = RealCuGAN_Scalar
        elif configuration.model_name == "VCISR":
            from Real_ESRGAN.uprrdb_main import RealESRGAN_Scalar
            NN_model = RealESRGAN_Scalar
        else:
            raise NotImplementedError()

        # Full Frame Model
        print("Full Model Preparation")
        for idx in range(self.full_model_num):
            print(configuration.model_name + " full : " + str(idx))
            model = NN_model(weight_path_full_path, self.pixel_padding)
            upscaler_full = UpScalerMT("FULL", self.inp_q_full, self.res_q, model, configuration.p_sleep, 1)
            upscaler_full.start()

        # Partition Frame Model
        print("Partition Model Preparation")
        for id in range(self.nt):
            print(configuration.model_name + " partition : " + str(id))
            model = NN_model(weight_path_partition_path, self.pixel_padding)
            upscaler = UpScalerMT(id, self.inp_q, self.res_q, model, configuration.p_sleep, self.nt)
            upscaler.start()
        ######################################################################################################################


    def __call__(self, input_path, output_path):
        ''' Main Calling function
        
        '''
        ############################### Build PATH && INIT ############################################
        # Basic Path Preparation
        # video_format = input_path.split(".")[-1]
        # os.makedirs(os.path.join(root_path_, "tmp"), exist_ok=True)
        # tmp_path = os.path.join(root_path_, "tmp", "%s.%s" % (int(ttime()*1000000), video_format))
        # os.link(input_path, tmp_path)
        # objVideoreader = VideoFileClip(filename=tmp_path)
        objVideoreader = VideoFileClip(filename=input_path)

        # Obtain basic video information
        total_duration = objVideoreader.duration
        self.width, self.height = objVideoreader.reader.size
        original_fps = objVideoreader.reader.fps
        nframes = objVideoreader.reader.nframes
        has_audio = objVideoreader.audio


        if self.decode_fps == -1:
            # Use original fps as decode fps
            self.decode_fps = original_fps
        self.total_frame_number = int(self.decode_fps * (nframes/original_fps))

        # Build Video Writer 
        self.writer = FFMPEG_VideoWriter(output_path, (self.width * self.scale * self.rescale_factor, self.height * self.scale * self.rescale_factor), self.decode_fps, ffmpeg_params=self.encode_params)
        #############################################################################################

        
        
        video_decode_loop_start = ttime()
        ######################################### video decode loop #######################################################
        for frame_idx, frame in enumerate(objVideoreader.iter_frames(fps=self.decode_fps)): 
            
            # Rescale the video frame at the beginning if we want a different output resolution
            if self.rescale_factor != 1:
                frame = cv2.resize(frame, (int(self.width*self.rescale_factor), int(self.height*self.rescale_factor))) # interpolation=cv2.INTER_LANCZOS4


            if frame_idx % 50 == 0 or int(self.total_frame_number) == frame_idx:
                # 以后print这边用config统一管理
                print("Total frame:%s\t video decoded frames:%s"%(int(self.total_frame_number), frame_idx))
                sleep(self.decode_sleep)  # This is very needed to avoid conflict in the process that is too overwhelming


            # Check if NN process too slow to catch up the frames decoded
            decode_processed_diff = frame_idx - self.now_idx
            if decode_processed_diff >= 650:        # This is an empirical value
                self.frame_write()
                if decode_processed_diff >= 1000:
                    # Have to do this else it's possible to raise bugs
                    print("The program decode speed is much faster than the GPU processing speed, needs to sleep 0.4s!")
                    sleep(0.4)


            ######################### Use MSE to judge whether it's highly overlapped #########################################

            queue_put_idx = []      # 0, 1, 2 are the partition frame, 3 is the full frame 
            (crop0, crop1, crop2) = crop4partition(frame)  # We use "cropX" to access these variables

            if frame_idx == 0: 
                # For the first frame, we just put into a whole frame into the sequence
                self.full_frame_cal_num += 1
                queue_put_idx = [3]

                # Init the reference_frame and reference_idx
                for i in range(3):
                    cropX = eval("crop%s"%i)
                    self.reference_frame[i] = cropX[:, :, 0]            # We only store Single Red Channel to compare to accelerate
                    self.reference_idx[i] = frame_idx
            
            elif self.time2switchFULL > 0:  # Use Momentum 
                self.time2switchFULL -= 1       # Update the counter
                self.momentum_used_times += 1   # Update the number of frames we process with momentum

                # Abjust based on full_model_num and nt
                if self.full_model_num > 0:
                    queue_put_idx = [3]
                else:       # In this case, we can only use partition queue
                    assert(self.nt != 0)
                    queue_put_idx = [0,1,2]
            
            else:  # No momentum considered
                # Calculate MSE
                mse_differences = [float("inf"), float("inf"), float("inf")]
                for i in range(3):
                    # Calculate MSE for each crop partition
                    cropX = eval("crop%s"%i)

                    if self.reference_frame[i] is None or (frame_idx - self.reference_idx[i]) >= self.Max_Same_Frame:     
                        # Add Reference if it is none or more than the designated maximum same frame we allow
                        self.reference_frame[i] = cropX[:, :, 0]            # We only store Single Red Channel to compare to accelerate
                        self.reference_idx[i] = frame_idx

                    else:   # Calculate MAE error
                        # Record the MSE between two frames
                        frame_err = np.sum((self.reference_frame[i] - cropX[:, :, 0])**2) / float(cropX.shape[0] * cropX.shape[1])     # We must use float32 here for precision (else, it's int8); Also, MAE speed has no distinct difference
                        mse_differences[i] = frame_err
    

                # Decide if we use PARTITION / FULL frame mode (for full frame, decide if we need to use MOMENTUM)
                if self.full_model_num != 0 and all(mse > self.MSE_range for mse in mse_differences):
                    # Use FULL frame mode When we use the full_model mode, and When ALL the MSE difference is larger than the threshold
                    self.full_frame_cal_num += 1
                    queue_put_idx = [3]

                    # Check if we need to activate the MOMENTUM mechanism when the change of motion is too much
                    if all(mse_value > 5 and mse_value != float("inf") for mse_value in mse_differences):  # 5 is an empirical value (mse_value cannot be inf, because this is frame after momentum or more than Max Same Frame)
                        self.momentum_reference.append(True)
                        if sum(self.momentum_reference) == self.momentum_reference_size:
                            # If we have momentum_reference_size amount of frames that have big MSE difference between consequent frames, we activate MOMENTUM mechanism
                            # print("We need to use momentum at frame {}".format(frame_idx))
                            self.time2switchFULL = self.momentum_skip_crop_frame_num                    # Set how many frames we will skip
                            self.reference_frame = [None, None, None]                                   # Reset reference
                            self.reference_idx = [-1, -1, -1]                                           # Reset reference
                            self.momentum_reference.extend([False] * self.momentum_reference_size)      # Fill in all elements of momentum_reference with False
                        else:
                            # Update the Parition and idx
                            for partition_idx in range(3):
                                cropX = eval("crop%s"%partition_idx)
                                self.reference_frame[partition_idx] = cropX[:, :, 0]
                                self.reference_idx[partition_idx] = frame_idx
                    else:
                        # Reset the reference_frame and reference_idx
                        self.momentum_reference.append(False)

                        # Update the Parition and idx
                        for partition_idx in range(3):
                            cropX = eval("crop%s"%partition_idx)
                            self.reference_frame[partition_idx] = cropX[:, :, 0]
                            self.reference_idx[partition_idx] = frame_idx

                else:
                    # Put PARTITION frame instead of the full frame into the queue
                    self.momentum_reference.append(False)       # Update momentum record
                    for partition_idx in range(3):
                        cropX = eval("crop%s"%partition_idx)
                        if mse_differences[partition_idx] > self.MSE_range:
                            # Two frames have limited similarity, Reset reference_frame and reference_idx;
                            self.reference_frame[partition_idx] = cropX[:, :, 0]
                            self.reference_idx[partition_idx] = frame_idx
                            queue_put_idx.append(partition_idx)
                        else:
                            # Two frames has very high similarity, Put reference to idx2res from reference_idx
                            self.idx2res[frame_idx][partition_idx] = self.reference_idx[partition_idx]
                            self.skip_counter_ += 1
                            # print("We skip frame_idx {} and partition_idx {}!".format(frame_idx, partition_idx))


            # Put partition/full frames into the queue 
            if 3 in queue_put_idx:
                # Full frame put into the queue
                assert(len(queue_put_idx) == 1)     # We cannot have partition idx here
                self.queue_put(frame_idx, 3, frame, full = True)
            else:
                # Partition frame put into the queue (the queue may be empty)
                assert(3 not in queue_put_idx)      # We cannot have full idx here
                for partition_idx in sorted(queue_put_idx):
                    cropped_frame = eval("crop%s"%partition_idx)
                    self.queue_put(frame_idx, partition_idx, cropped_frame, full = False)

            ####################################################################################################################

            # Write frames per 4 frame. We don't write for each frame because we want to save time that is 
            if frame_idx % 4 == 0:
                # sleep(self.decode_sleep)      # If you need, put this in front of the frame_write
                self.frame_write()

        print("All Frames are decoded from the input video!")
        #########################################################################################################


        ################################################ 后面残留的计算 ##################################################
        frame_idx += 1 # Fit the total number of frames
        while True:
            self.frame_write()

            # if self.inp_q is not None and self.inp_q.qsize() == 0 and self.res_q.qsize() == 0 and idx == self.now_idx:
            #     break
            # elif self.res_q.qsize() == 0 and idx == self.now_idx:
            #     break

            if frame_idx == self.now_idx:
                # If we process till the last index, we can end the loop
                if self.inp_q is not None:
                    assert(self.inp_q.qsize() == 0)
                assert(self.res_q.qsize() == 0)
                break
            
            # TODO: optimize this code if needed
            sleep(self.decode_sleep)

        print("Final image index is ", self.now_idx)

        # Closs all queues
        for _ in range(self.nt):  
            self.inp_q.put(None)
        for _ in range(self.full_model_num):
            self.inp_q_full.put(None)

        # close writer to save all stuff
        self.writer.close()
        

        video_decode_loop_end = ttime()
        ################################################################################################################################################

        ############################################################ Analysis ##########################################################################
        # Calculation
        full_time_spent = video_decode_loop_end - video_decode_loop_start
        total_exe_fps = self.total_frame_number / full_time_spent
        full_frame_portion = self.full_frame_cal_num / self.total_frame_number
        partition_saved_portion = self.skip_counter_ / (self.total_frame_number*3)      # This 3 means that there are three partitions. This is a hard-coded design

        # The most import report        
        print("Input path is %s and the report is the following:"%input_path)
        if full_time_spent < 60:
            print("Done! Total time cost:", full_time_spent)
        else:
            print("Done! Total time cost: %d min %d s" %(full_time_spent//60, full_time_spent%60))


        # Details report
        print("The following is the detailed report:")
        print("\t Saved frames number: %d partitions which is %.2f %% of the full area" %(self.skip_counter_, 100*partition_saved_portion))
        print("\t The Number of partitions put into small Upscaler (1in3) is %d which is %.2f %%" % (
                self.parition_processed_num, 100 * self.parition_processed_num / (self.total_frame_number * 3)))
        print("\t Total full_frame_cal_num is %d which is %.2f %%" %(self.full_frame_cal_num, 100*full_frame_portion))
        print("\t Total momentum used num is %d which is %.2f %%" %(self.momentum_used_times, 100*self.momentum_used_times//self.total_frame_number))
        #############################################################################################################################################
        

        ##################################### Generate Final Report #################################################################################
        report = {}
        report["input_path"] = input_path
        report["full_time_spent"] = full_time_spent
        report["total_exe_fps"] = total_exe_fps
        report["performance_scale"] = total_exe_fps/self.decode_fps
        report["parition_processed_num"] = self.parition_processed_num
        report["skip_counter"] = self.skip_counter_
        report["full_frame_cal_num"] = self.full_frame_cal_num
        report["momentum_used_times"] = self.momentum_used_times

        return report


    def frame_write(self):
        ''' Extract parition/full frame from res_q and write to ffmpeg writer (moviepy)
        '''

        #Step1：Write to tmp places because multi-process and multi-threading is unblanced
        while True:  # Get all the processed results
            if self.res_q.empty():
                break
            iidx, position, res = self.res_q.get()
            self.idx2res[iidx][position] = res


        #Step2: put the tmp result to video writer 
        while True:  # 按照idx排序写帧
            if not self.res_q.empty():
                iidx, position, res = self.res_q.get()
                self.idx2res[iidx][position] = res

            # For sanity check purpose
            if self.loop_counter == self.max_cache_loop: 
                self.writer.close()
                print("Ends at frame ", self.now_idx)
                print("\t Continuously not found, end the program and store what's stored")
                os._exit(0)

            # Neither all parition nor single whole frame inside the idx2res, break this loop
            if not all(i in self.idx2res[self.now_idx] for i in [0, 1, 2]) and not 3 in self.idx2res[self.now_idx]:
                self.loop_counter += 1
                break
            self.loop_counter = 0


            ######################################## The following is safe to execute with all frames needed #######################################
            if self.now_idx % 50 == 0:
                print("Process {} had written frames: {}".format(self.process_id, self.now_idx))

            # 3 types of cropping
            if 3 not in self.idx2res[self.now_idx]:
                # Partition Frame cases
                crops = []
                for idx in [0, 1, 2]:
                    if isinstance(self.idx2res[self.now_idx][idx], int):
                        # This one is an index based (with a reference that has high similarity that we can skip its inference)
                        target_idx = self.idx2res[self.now_idx][idx]
                        if 3 in self.idx2res[target_idx]:
                            # This means that the reference is a whole frame, we need to crop it to extract.
                            crops.append(crop4partition_SR(self.idx2res[target_idx][3], idx))
                        else: 
                            crops.append(self.idx2res[target_idx][idx])   # Extract directly
                    else:
                        # This one is NN inferenced result
                        crops.append(self.idx2res[self.now_idx][idx])

                combined_frame = combine_partitions_SR(*crops)  # adjust is a fixed value
            else:
                # Full frame cases
                combined_frame = self.idx2res[self.now_idx][3]

            # Write the frame
            # cv2.imwrite(str(self.now_idx)+".png", cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))  # For Debug purpose (Please set the process_num = 1)
            self.writer.write_frame(combined_frame)


            # Delete frame or index in idx2res to save the memory
            if self.now_idx > self.Max_Same_Frame:
                del self.idx2res[self.now_idx - self.Max_Same_Frame]

            self.now_idx += 1   # Update the index of frames we have already inferenced and encoded.



    def queue_put(self, frame_idx, position, frame, full):
        ''' Put into the queue
        Args:
            frame_idx (int):    Global frame index
            position (int):     Position of frame 0|1|2|3
            frame (numpy):      The numpy format of the image
            full (bool):        If we use the full queue (when it's False, we use partition queue)
        '''
        # print("put info is ", frame_idx, position, frame.shape)

        # For either queue, we need to send corresponding frame index and its position, such that the program can correspond each frame in after-process
        if full:
            # Full queue put
            self.inp_q_full.put((frame_idx, 3, frame))          # Full queue
        else:
            # Partition queue put
            self.inp_q.put((frame_idx, position, frame))        # Partition frame
            self.parition_processed_num += 1



