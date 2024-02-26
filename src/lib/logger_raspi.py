import serial
import time
import threading
import datetime
import RPi.GPIO as GPIO
    
try:
    ser = serial.Serial('/dev/ttyUSB0', 74880)
except :
    print("connect to ttyusb0")

button_pin = 18  
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
check_start_flag=0

closefile_flag=0



GPIO.add_event_detect(button_pin, GPIO.FALLING, callback=handle_file_operations, bouncetime=300)

def handle_file_operations(i):
    global check_start_flag
    check_start_flag=1

def timer_callback():
    global closefile_flag
    closefile_flag=1
 
    
def logger_data():
    global closefile_flag
    global thread
    global ser
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{timestamp}.txt"
    file= open("D:/matlab/ymaps_code/data"+file_name, "w") 
    print("File created:", file_name)
    
    closefile_flag=0
    thread=threading.Timer(120,timer_callback)
    thread.start()
    
    
    while True:
        try:
            file.write(str(ser.readline().decode('utf-8').strip().split(',').replace("'", "").replace("[", "").replace("]", "")) + '\n')
        except KeyboardInterrupt:
            file.close()
            print(f"file{file_name} cloced \n")
        
            break
        if(closefile_flag==1):
            file.close()
            print(f"file{file_name} cloced after 120s \n")
            break

def main(i):
    global check_start_flag
    while True:
        
        if check_start_flag==1:
            check_start_flag=0
            logger_data()
        
        
if __name__=="__main__":
    main(0)
    
