import serial
import time
import csv
import sys
import select

# Set up the serial port for SimpleFOC Arduino
arduino2_port = '/dev/cu.usbmodem2101'
baud_rate = 115200

# Connect to Arduino
arduino2 = serial.Serial(arduino2_port, baud_rate, timeout=1)

# Cone angle defined within the script
cone_angle = 30  # Example value, change as needed

# CSV file setup
file_path = 'experimental_data/flow-occurrence.csv'
with open(file_path, mode='a', newline='') as file:
    data_writer = csv.writer(file)
    # Uncomment the next line if the file might be empty and needs a header
    # data_writer.writerow(['Cone Pitch (deg)', 'Angular Velocity at flow occurrence (rad/s)'])


def start_experiment():
    current_speed = 0
    try:
        while True:
            # Send speed command to Arduino 2
            speed_command = f'A{current_speed}\n'.encode()
            arduino2.write(speed_command)
            print("Current Speed Command: ", current_speed)

            # Non-blocking input check
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input()  # Clear the input buffer
                print("Flow occurs at: ", current_speed, "rad/s")
                return current_speed

            # Determine speed increment and time interval based on current speed
            if current_speed < 100:
                increment = 1
                sleep_time = 0.5

            elif current_speed < 140:
                increment = 1


                sleep_time = 0.75
            elif current_speed < 155:
                increment = 0.5
                sleep_time = 1.25
            else:
                increment = 0.25
                sleep_time = 1.5

            current_speed += increment  # Increment speed

            time.sleep(sleep_time)  # Wait before the next increment
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("Experiment interrupted.")


# Main loop
print("Experiment will start shortly.")
time.sleep(5)  # Give time for preparation before starting

for experiment_count in range(10):
    print(f"Starting experiment {experiment_count + 1}")
    final_speed = start_experiment()

    # Record the experimental_data_old to CSV
    with open(file_path, mode='a', newline='') as file:
        data_writer = csv.writer(file)
        data_writer.writerow([cone_angle, final_speed])

    print(f"Experiment {experiment_count + 1} completed and experimental_data_old saved.")
    time.sleep(2)  # Short break between experiments

# Cleanup
arduino2.write(b'A0\n')  # Stop the motor
arduino2.close()
print("\nAll experiments completed and experimental_data_old saved.")
print(f"All experimental_data_old saved to {file_path}")
