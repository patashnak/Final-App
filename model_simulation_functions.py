
import numpy as np
from scipy.integrate import odeint, simps
from functools import lru_cache

@lru_cache(maxsize=10)
def simulate(params, mode='acute', t_max=50):
    t = np.linspace(0, t_max, 500)
    tnf_profile = lambda x: TNF_input(x, mode)
    y0 = [0.0, 0.0]
    sol = odeint(model_odes, y0, t, args=(params, tnf_profile))
    Ca_astro = sol[:, 0]
    F_neuron = sol[:, 1]
    return t, Ca_astro, F_neuron

def TNF_input(t, mode='acute'):
    if mode == 'acute':
        return 1.0 * (np.exp(-t/5.0) - np.exp(-t/0.5))
    elif mode == 'chronic':
        return 0.5 / (1 + np.exp(-0.2*(t-20)))
    else:
        raise ValueError("Mode must be 'acute' or 'chronic'")

def model_odes(y, t, params, tnf_profile):
    Ca, F_neuron = y
    TNF_t = tnf_profile(t)
    dCa_dt = params['alpha'] * TNF_t + params['eta'] * F_neuron - params['beta'] * Ca
    Glu = params['gamma'] * Ca
    dF_dt = (params['delta'] * Glu - params['epsilon'] - F_neuron) / 0.5
    return [dCa_dt, dF_dt]

def extract_metrics(t, F_neuron):
    peak = np.max(F_neuron)
    auc = simps(F_neuron, t)
    time_to_peak = t[np.argmax(F_neuron)]
    duration = np.sum(F_neuron > 1.0) * (t[1] - t[0])
    return {'peak_firing': peak, 'auc_firing': auc, 'time_to_peak': time_to_peak, 'firing_duration': duration}
