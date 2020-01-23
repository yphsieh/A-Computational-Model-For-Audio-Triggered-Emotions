import nengo 
import numpy as np
import argparse
import math
# import matplotlib.pyplot as plt
from nengo.utils.ensemble import tuning_curves

from functions import *


def Cep(n_neurons, freqs, n_cepstra, Net):
    assert Net != None
    with Net:
        # set EnsembleArray 
        cep = nengo.networks.EnsembleArray(n_cepstra, freqs.size)
        # connect AP output to Ceptral input after iDCT transform
        nengo.Connection(Net.output, cep.input, transform=idct(freqs.size, n_cepstra))

        # observe signal from AP output
        probe0 = nengo.Probe(Net.output, synapse=0.01)
        # observe signal after iDCT transform
        probe1 = nengo.Probe(cep.input)
        #observe signals passing Cep with synapse=0.01
        probe2 = nengo.Probe(cep.output, synapse=0.01)

    # plt.figure(figsize=(15,12))
    with nengo.Simulator(Net) as sim:
        sim.run(Net.duration)

    #     for i in range(0,freqs.size):
    #         eval_points, activities = tuning_curves(cep.ea_ensembles[i], sim)
    #         plt.subplot(4,freqs.size/4,i+1)
    #         plt.plot(eval_points, activities)
    #         plt.ylabel("Firing rate (Hz)")
    #         plt.xlabel("Input scalar, x");

    # plt.show()

    # plt.figure()
    # plt.plot(sim.trange(), sim.data[probe0], 'b', label="signal pass AP")
    # plt.plot(sim.trange(), sim.data[probe2], 'r', label="signal pass Ceptral")
    # plt.plot(sim.trange(), sim.data[probe1], 'k', label="signal pass iDCT")

    np.save("./x.npy", sim.trange())
    np.save("./ap.npy", sim.data[probe0])
    np.save("./cep.npy", sim.data[probe2])
    # np.save("./np2.npy", sim.data[probe2])

    # plt.xlim(0, math.ceil(Net.duration))
    # plt.ylim(0, 1.75)
    # plt.show()

    Net.output=cep.output
    return Net
