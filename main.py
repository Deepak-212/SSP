import numpy as np
import matplotlib.pyplot as plt
from control import tf, feedback, forced_response

#This is the transfer function G(s)
numerator = [1]
denominator = [1, 3, 5, 1]
plant = tf(numerator, denominator)

# PID Controller Parameters
Kp = 4.5
Ki = 1.174
Kd = 0.9

# Here I have implemented a pseudo derivative that is a derivative
#passed through a low pass filter. This will reduce the amplification
#of the high frequency noise by filtering.
a = 2.0
D_filtered = Kd * tf([a, 0], [1, a])  # (a * s) / (s + a)

pid_controller = Kp + Ki / tf([1, 0], [1]) + D_filtered

# Since the pid controller is not able to account for the noise I
#have used another low pass filter
b = 0.5
sensor_lpf = tf([1], [b, 1])  # First-order LPF: 1 / (τs + 1)

# Closed-loop system (before adding sensor noise)
closed_loop_system = feedback(pid_controller * plant, 1)

time = np.linspace(0, 20, 1000)

# Step Input starting from 2 seconds
step_input = np.where(time >= 2, 1.0, 0.0)

# This simulates the output response
_, true_output = forced_response(closed_loop_system, time, step_input)

# Adding sensor noise (Gaussian noise)
add_sensor_noise = True#if I set this to false no noise will be added
if add_sensor_noise:
    noise_amplitude = 0.1  # Adjust based on realistic noise levels
    noise = np.random.normal(0, noise_amplitude, size=len(true_output))  # Random noise
    noisy_measurement = true_output + noise  # Noisy sensor measurement
else:
    noisy_measurement = true_output

# using the sensor low pass filter in the feedback loop
_, filtered_measurement = forced_response(sensor_lpf, time, noisy_measurement)


plt.figure(figsize=(12, 8))

# Plot of the Input Signal
plt.subplot(3, 1, 1)
plt.plot(time, step_input, label="Input Signal", color="black", linestyle="dashed")
plt.title("Input vs. Output of PID Controlled System with Realistic Sensor Noise")
plt.xlabel("Time (s)")
plt.ylabel("Input")
plt.legend()
plt.grid(True)

# Plot of the Noisy Signal
plt.subplot(3, 1, 2)
plt.plot(time, noisy_measurement, label="Noisy Measurement", color="red", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

# Plot of the Filtered Measurement Signal
plt.subplot(3, 1, 3)
plt.plot(time, filtered_measurement, label="Filtered Signal", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
