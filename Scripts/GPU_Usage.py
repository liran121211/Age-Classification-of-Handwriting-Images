import psutil
import humanize
import os
import GPUtil as GPU
from time import sleep
import torch

def GPU_Usage():
    while(True):
        GPUs = GPU.getGPUs()
        
        def printm():
            process = psutil.Process(os.getpid())
            print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
                  " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
            print("GPU Name: {0} | GPU RAM Free: {1:.0f}MB | Used: {2:.0f}MB | Util {3:3.0f}% | Total {4:.0f}MB".format(
                                                                                                        torch.cuda.get_device_name(0),
                                                                                                        GPUs[0].memoryFree,
                                                                                                        GPUs[0].memoryUsed,
                                                                                                        GPUs[0].memoryUtil * 100,
                                                                                                        GPUs[0].memoryTotal))

            print("GPU Name: {0} | GPU RAM Free: {1:.0f}MB | Used: {2:.0f}MB | Util {3:3.0f}% | Total {4:.0f}MB".format(
                                                                                                        torch.cuda.get_device_name(1),
                                                                                                        GPUs[1].memoryFree,
                                                                                                        GPUs[1].memoryUsed,
                                                                                                        GPUs[1].memoryUtil * 100,
                                                                                                        GPUs[1].memoryTotal))
        sleep(3.0) # sleep 1 second
        print("\n\n")
        printm()

# run loop
GPU_Usage()