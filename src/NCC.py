import numpy as np
# import nengo_dl
import argparse
# import matplotlib.pyplot as plt
import brian.hears as bh
import librosa
import soundfile as sf

# import BrianHears as bh
from functions import *
from AuditoryPeriphery import *
from Ceptral import *
# from Derivatives import *
# from nengo.cache import NoDecoderCache
import nengo

def ncc(audio):
	# load .wav file through brian.hears.loadsound()

	x,_ = librosa.load(audio, sr=16000)
	sf.write('tmp.wav', x, 16000)

	audio = bh.loadsound('tmp.wav')
	t_audio = audio.size / 16000
	# set the observing frequences
	# human hearing frequency range 20 ~ 20000 Hz
	freqs = mel2hz(np.linspace(hz2mel(20), hz2mel(8000), 4))
	print freqs

	# pass audio to AuditoryPeriphery net
	APnet = AudPeri(20, audio, freqs)
	# pass AuditoryPeriphery net to Ceptral net
	Cepnet = Cep(20, freqs, 4, APnet)
	# the whole NCC network is the network returned by Ceptral net
	NCC = Cepnet

	# dNCC = IntermediateDeriv(20, freqs, Cepnet)

	# # pass Ceptral net to Derivative net
	# # choose between the two models to simulate the derivative network 
	# if args.deriv == 'interm' : NCC = IntermediateDeriv(20, freqs, Cepnet)
	# # elif args.deriv == 'feedforward' : 
	# NCC = FeedforwardDeriv(20, freqs, Cepnet)

	
	# the returned net is the NCCs network
	with NCC:
		# observe the output of NCCs network with a probe
		probe = nengo.Probe(NCC.output, synapse=0.01)

 # #    Disable decoder cache for this model
	# # _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
	# # Simulation using nengo_dl
	# with nengo.Simulator(NCC) as sim:
	# 	sim.run(t_audio, progress_bar=False)
	# # the feature extraction is done by returning the data through simulation
	# feat = sim.data[probe]

	# with dNCC:
	# 	probe = nengo.Probe(dNCC.output, synapse=0.01)
	# # Disable decoder cache for this model
	# _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
	# # Simulation using nengo_dl
	# with nengo_dl.Simulator.(dNCC, model=_model) as sim:
	# 	sim.run(t_audio, progress_bar=False)
	# # the feature extraction is done by returning the data through simulation
	# dfeat = sim.data[probe]

	# feat = [feat, dfeat]
	# plt.figure()
 #    plt.plot(sim.trange(), sim.data[probe2], 'b',label="decoded output")

 #    plt.xlim(0, math.ceil(Net.duration))
 #    plt.ylim(0, 1.75)
 #    plt.show()
	# return feat

if __name__ == '__main__':
	# run code : python NCC.py --audio [.wav file] --n_freqs [integer]
	parser = argparse.ArgumentParser(prog='sound2brain.py')
	parser.add_argument('--audio', type=str, default='./data/footstep_trim.wav')
	# number of observing frequences 
	parser.add_argument('--deriv' , type=str, default='interm')
	parser.add_argument('--n_freqs', type=int, default=20)
	args = parser.parse_args()
	audio = args.audio
	ncc(audio)
