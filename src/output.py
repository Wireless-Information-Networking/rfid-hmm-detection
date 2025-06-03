import numpy as np

# Constants
P_TX_dBm = 10*np.log10(2000) # 2W
NOISE_POWER_dBm = -90 # UNKNOWN
SHAD_dB = 3 # UNKNOWN
FREQ = 915e6 # tag frequency
C = 3e8 # Speed of light

# Emulates a tag appearing and disappearing from a given place
# Returns a 2D numpy tensor. First row is presence/absence. Second row is received power.
# Columns indicate time.
def one_tag_in_and_out(cycles, mean_cycle_samples, distance_to_reader):
    rng = np.random.default_rng()
    samples_per_cycle = np.round(rng.uniform(0, 2*mean_cycle_samples, cycles)).astype(int)
    tag_presence_per_cycle = (np.round(rng.uniform(0, 1, cycles))).astype(int)
    output = np.zeros((2, np.sum(samples_per_cycle)))
    column_position = 0
    for cycle in range(cycles):
        output[0, column_position: column_position+samples_per_cycle[cycle]] = tag_presence_per_cycle[cycle]
        noise_linear = 10**(NOISE_POWER_dBm/10)*rng.chisquare(2,samples_per_cycle[cycle])
        rssi_dbm = P_TX_dBm + 40*np.log(C/(4*np.pi*distance_to_reader*FREQ)) + rng.normal(0,SHAD_dB,samples_per_cycle[cycle])
        rssi_and_noise_linear = tag_presence_per_cycle[cycle]*10**(rssi_dbm/10) + noise_linear
        rssi_noise_dbm = 10*np.log10(rssi_and_noise_linear)
        output[1, column_position: column_position + samples_per_cycle[cycle]] = rssi_noise_dbm
        column_position += samples_per_cycle[cycle]
    return output

# Emulates a tag appearing and disappearing from a given place
# Returns a 2D numpy tensor. First row is received power. Second row is binary (presence/absence).
# Columns indicate time.
if __name__ == '__main__':
    example = one_tag_in_and_out(10, 100, 0.7)
    
    # Plot the samples
    import matplotlib.pyplot as plt
    
    # set two subplots
    plt.subplot(2,1,1)
    plt.plot(example[1,:])
    plt.title('Received power')
    plt.subplot(2,1,2)
    plt.plot(example[0,:])
    plt.title('Tag presence')
    plt.show()
