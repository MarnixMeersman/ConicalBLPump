import serial
import time
import csv
import threading

# Set up the serial port for SimpleFOC Arduino
arduino2_port = '/dev/cu.usbmodem2101'
baud_rate = 115200

# Connect to Arduino
arduino2 = serial.Serial(arduino2_port, baud_rate, timeout=1)

# Cone angle defined within the script
cone_angle = 30  # Example value, change as needed

# Global variable to control the experiment flow
experiment_running = True


def increase_speed():
    global experiment_running
    current_speed = 0
    while experiment_running:
        # Send speed command to Arduino 2
        speed_command = f'A{current_speed}\n'.encode()
        arduino2.write(speed_command)
        print("Current Speed Command: ", current_speed)

        # Increment speed
        time.sleep(2)
        current_speed += 2.5


def start_experiment():
    global experiment_running

    # Start the speed increase in a separate thread
    threading.Thread(target=increase_speed).start()

    # Wait for Enter key to stop the experiment
    input("Press Enter to detect flow and stop experiment.")
    experiment_running = False
    return arduino2.current_speed  # Assuming the current speed is somehow tracked or returned


# Main loop
print("Press Enter to start the experiment.")
input()  # Wait for Enter to start
final_speed = start_experiment()

# Record the experimental_data_old to CSV
file_path = 'experimental_data/flow-occurance.csv'
with open(file_path, mode='a', newline='') as file:
    data_writer = csv.writer(file)
    data_writer.writerow([cone_angle, final_speed])

# Cleanup
arduino2.write(b'A0\n')  # Stop the motor
arduino2.close()
print("\nExperiment completed and experimental_data_old saved.")
print(f"Data saved to {file_path}")
