import numpy as np
import nengo
import brian.hears as bh
import nengo.utils.numpy as npext

# transform Hz to Mel
def hz2mel(hz):
    return 2595. * np.log10(1 + hz / 700.0)
# transform Mel to Hz
def mel2hz(mel):
    return 700. * (10. ** (mel / 2595.0) - 1)

# Gammatone filterbank
def gammatone(sound, freqs, dt, b=1.019):
    duration = int(dt * sound.samplerate)
    fb = bh.Gammatone(sound, freqs, b=b)
    fb.buffersize = duration
    ihc = bh.FunctionFilterbank(fb, (lambda x: 3*np.clip(x, 0, np.inf) ** (1. / 3.)))
    ihc.cached_buffer_end = 0  
    return ihc


class AuditoryFilterBank(nengo.processes.Process):
    def __init__(self, freqs, audio, samplerate):
        self.freqs = freqs
        self.audio = audio
        self.samplerate = samplerate
        super(AuditoryFilterBank, self).__init__()

    def make_step(self, size_in, size_out, dt, rng):
        audio = self.audio

        # audio pass through filterBanks: MiddleEar, Gammatone
        sound = bh.MiddleEar(audio, gain=1)
        ihc = gammatone(sound, self.freqs, dt)
        # bh.sound.buffer_fetch()
        duration = int(dt * self.samplerate)
        def step_filterbank(t, startend=np.array([0, duration], dtype=int)):
            result = ihc.buffer_fetch(startend[0], startend[1])
            startend += duration
            return result[-1]

        return step_filterbank

# Inverse discrete cosine transform
# reference : wikipedia
def idct(n, size_out):
    k = np.arange(n)
    s = np.ones(n)
    s[0] = np.sqrt(0.5)
    idct_matrix = (np.sqrt(2. / n) * s
                   * np.cos(np.pi * np.outer(k + 0.5, k) / n))
    return idct_matrix[:size_out]



