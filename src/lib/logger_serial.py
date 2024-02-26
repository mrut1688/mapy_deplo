import serial
import time
import threading
import datetime
import os
import analyse_csv
from tkinter import *
from tkinter.filedialog import askdirectory

try:
    ser = serial.Serial('COM7', 115200)
except  serial.serialutil.SerialException:
    print("connect to comport7")

def timer_callback():
    global closefile_flag
    closefile_flag=1
    timer_count=0

 
def timer1_callback():
    global timer1_flag
    global timer_count
    global movements
    global timmer2
    
    timer1_flag=1
    timer_count=timer_count+1
    timmer[timer_count].start()
    print(movements[timer_count]+"\n")  
    

def timer2_callback():
    global timer2_flag
    global timer_count
    global movements
    global closefile_flag
    global timmer2
    
    timer2_flag=1
    timer_count=timer_count+1
    if closefile_flag!=1:
        timmer[timer_count].start()
    print(movements[timer_count]+"\n")

def timer3_callback():
    global timer2_flag
    global timer_count
    global movements
    global closefile_flag
    global timmer2
    
    timer2_flag=1
    if closefile_flag!=1:
        
        print(movements[6]+"\n")

closefile_flag=0
timer1_flag=0
timer2_flag=0
timmer = [None] * 6 

thread=threading.Timer(120,timer_callback)

timmer[0]=threading.Timer(10,timer1_callback)
timmer[1]=threading.Timer(20,timer2_callback)
timmer[2]=threading.Timer(20,timer2_callback)
timmer[3]=threading.Timer(20,timer2_callback)
timmer[4]=threading.Timer(20,timer2_callback)
timmer[5]=threading.Timer(10,timer3_callback)

timer_count=0
movements=["rest for 10 s","right hand slow mostion for next 20 s",
           "right hand fast motion for next 20 s","rest for next 20 s ",
           "left hand slow mostion for next 20 s",
           "left hand fast motion for next 20 s",
           "random motion for 10 s"]


def logger_data_img():
    global closefile_flag
    global thread
    global ser
    global timer1_flag
    global timer2_flag
    global timmer1
    global timmer2
    global timer_count
    global movements
      
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    datestamp = datetime.datetime.now().strftime("%Y%m%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"data_{timestamp}.csv"
    par_dir=askdirectory()
    
    path= os.path.join(par_dir+datestamp)
    
    if not os.path.exists(path): 
        os.mkdir(path)
    file_path_name=path+"/"+"img_"+file_name
    file= open(file_path_name, "w") 
    print("File created:", file_name)
    
    closefile_flag=0
    
    
    thread.start()
    timmer[timer_count].start()
    print(movements[timer_count]+'\n')
    o=1
    
    while o==1:
#         print(".")
#         print(str(ser.readline().decode('utf-8').strip().split(',')) + '\n')
#         file.write(str(ser.readline().decode('utf-8').strip().split(',')) + '\n')
#         print(".")
        
        try:
             
             file.write(str(ser.readline().decode('utf-8').strip().
                            split(',')).replace("[","").replace("]","").
                        replace("'","").replace(" ","") + '\n')
             
        except KeyboardInterrupt:
             file.close()
             print(f"file{file_name} cloced \n")
             break
            
        if(closefile_flag==1):
            file.close()
            print(f"file{file_name} cloced after 120s \n")
            break 
    sampling_freq=analyse_csv.get_sampling_freq(file_name,path)    
    print('data was sampled at rate of'+sampling_freq+r'$Hz$')
    analyse_csv.muscle_state(file_name,path)
    
    nexts=input('do you want to analyse the data?[Y/N]')
    if next==Y :
        analyse_csv.get_fft(file_name,path) 
    return file_name,path
 
def logger_data():
    global closefile_flag
    global thread
    global ser
    global timer1_flag
    global timer2_flag
    global timmer1
    global timmer2
    global timer_count
    global movements
    
   
    datestamp = datetime.datetime.now().strftime("%Y%m%d")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"data_{timestamp}.csv"
    par_dir=askdirectory()
    
    path= os.path.join(par_dir+datestamp)
    
    if not os.path.exists(path): 
        os.mkdir(path)
    file_path_name=path+"/"+file_name
    file= open(file_path_name, "w") 
    print("File created:", file_name)
    
    closefile_flag=0
    
    
    thread.start()
    
    timmer[timer_count].start()
    print (movements[timer_count]+'\n')
    o=1
    
    while o==1:
        try:
            
             file.write(str(ser.readline().decode('utf-8').strip().
                            split(',')).replace("[","").replace("]","")
                        .replace("'","").replace(" ","") + '\n')
             
        except KeyboardInterrupt:
             file.close()
             print(f"file{file_name} cloced \n")
             break
            
        if(closefile_flag==1):
            file.close()
            print(f"file{file_name} cloced after 120s \n")
            break 
    sampling_freq=analyse_csv.get_sampling_freq(file_name,path)    
    print('data was sampled at rate of'+sampling_freq+r'$Hz$')
    analyse_csv.muscle_state(file_name,path)
    
    nexts=input('do you want to analyse the data?[Y/N]')
    if next==Y :
        analyse_csv.get_fft(file_name,path) 
    return file_name,path
def main(a):
    global ser
    while True:
        try:
            ser = serial.Serial('COM7', 115200)
            break
        except  serial.serialutil.SerialException:
            print("connect to comport7")

    print(str(ser.readline().decode('utf-8').strip().split(','))
          .replace("[", "").replace("]", "").replace("'", "")
          .replace(" ", "") + '\n')


    while True:
        s=str(input("press s for actual data or press i  for imaginaty data "))
        if(a=='s'):
            file_name,path=logger_data()
            break
            s=''
        if(s=='i'):
            file_name,path=logger_data_img()
            break
            s=''
    return file_name,path
        
if __name__=="__main__":
    main('s')