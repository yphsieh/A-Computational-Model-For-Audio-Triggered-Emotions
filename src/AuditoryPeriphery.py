import nengo 
from nengo.dists import Choice, Uniform
import numpy as np
import math
import argparse
# import matplotlib.pyplot as plt
import brian.hears as bh
from functions import *
from nengo.utils.ensemble import tuning_curves

def AudPeri(n_neurons, audio, freqs):
    Net = nengo.Network(label = "Auditory Periphery")
    with Net:
        # for axis-x limit in figure
        Net.duration = audio.duration
        # pass through filterbanks: MiddleEar and Gammatone
        fb = AuditoryFilterBank(freqs, audio, samplerate=audio.samplerate)
        # signals passing filterbanks
        spfb = nengo.Node(output=fb, size_out=freqs.size)
        # set EnsembleArray with LIF neurons
        an = nengo.networks.EnsembleArray(n_neurons, freqs.size, intercepts=Uniform(-0.1, 0.5), 
                                              encoders=Choice([[1]]), neuron_type=nengo.LIF())
        # store output in oder to connect with the ceptral layer
        Net.output = an.output
        # connect signals passing filterbanks to the input of AP 
        nengo.Connection(spfb, an.input)

        # observe signals passing filterbanks
        probe1 = nengo.Probe(spfb) 
        # observe signals passing AP with synapse=0.01
        probe2 = nengo.Probe(an.output, synapse=.01) 

    # plt.figure(figsize=(15,12))
    # with nengo.Simulator(Net) as sim:
    #     sim.run(Net.duration)
        
    

        # for i in range(0,freqs.size):
        #     eval_points, activities = tuning_curves(an.ea_ensembles[i], sim)
        #     plt.subplot(4,freqs.size/4,i+1)
        #     plt.plot(eval_points, activities)
        #     plt.ylabel("Firing rate (Hz)")
        #     plt.xlabel("Input scalar, x");

    # plt.show()

    # plt.figure()
    # plt.plot(sim.trange(), sim.data[probe1], 'r',label="signal pass filterbank")
    # plt.plot(sim.trange(), sim.data[probe2], 'b',label="decoded output")

    # plt.xlim(0, math.ceil(Net.duration))
    # plt.ylim(0, 1.75)
    # plt.show()

    return Net


