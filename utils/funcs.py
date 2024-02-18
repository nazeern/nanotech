import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt


# **********************
# Simulation Functions
# **********************

@st.cache_data
def I_out_sin(t, freq, I_amp, phase):
    return I_amp * np.sin(freq*(2*np.pi*t) + phase)

@st.cache_data
def get_V_out(V_in, I_out, Rf):
    rail = 1.65
    V_out = V_in - I_out * Rf
    V_out[V_out > rail] = rail
    V_out[V_out < -rail] = -rail
    return V_out

@st.cache_data
def generate_wave(freq, amp, form="sin", cycle_num=None, duration=None, samp_rate=None):
    """
    Return the time domain t and values  for a wave of given
    FREQUENCY, AMPLITUDE, and STYLE={"sin", "triangle"}
    
    Returns:
    t: Time domain of the generated wave
    vals: Output of the wave function
    
    Example:
    >>> t, V_in = generate_wave(FREQ, AMP, CYCLE_NUM, "sin")
    """
    if duration:
        cycle_num = freq * duration
    elif cycle_num:
        duration = cycle_num / freq
    elif duration and cycle_num and (cycle_num != freq * duration):
        raise ValueError(
            f"{cycle_num} cycles at {freq} cycles per second is incompatible with \
            duration {duration}"
        )
    else:
        raise ValueError('Wave generator requires either "cycle_num" or "duration"')
    
    # Sampling rate of device (samples / sec)
    if not samp_rate:
        warnings.warn("A sampling rate has not been specified; defaulting to 100 samples per cycle")
        samp_rate = freq * 100
    
    # Total number of samples
    samp_num = np.rint(duration * samp_rate)
    
    if form == "sin":
        t = np.arange(0, duration, duration / samp_num)
        vals = amp * np.sin(freq*(2*np.pi*t))
        
    elif form == "triangle":
        t = np.arange(0, 1, 1 / samp_num) * duration
        vals = amp * signal.sawtooth(2*np.pi*cycle_num*np.arange(0, 1, 1 / samp_num), 0.5)
        
    else:
        raise ValueError('Pass in a valid wave form ("sin", "triangle")')
    
    return t, vals

@st.cache_data
def get_Vr(V0, t, freq, amp, sgn, R, C):
    if sgn == 1:
        a = 4*freq*amp
    else:
        a = -4*freq*amp
    
    B = a*R*C - V0
    return a*R*C - B * np.exp(-t / (R * C))

@st.cache_data
def get_Vr_out(t, cycle_num, freq, amp, R, C):
    # Values of voltage across resistor
    Vr = np.zeros_like(t)

    # Number of total samples
    n = t.size

    # Number of samples in each rising/falling window
    k = n // cycle_num // 2

    currV = 0
    sgn = 1
    t_slice = t[:k]
    for w in range(cycle_num * 2):
        Vr[w*k:w*k+k] = get_Vr(currV, t_slice, freq, amp, sgn, R, C)
        currV = get_Vr(currV, t[k], freq, amp, sgn, R, C)
        sgn = -sgn
    return Vr

@st.cache_data
def digitize(wave, num_levels):
    V_amp = 1.65
    
    levels = (2 * V_amp * np.arange(num_levels + 1) - V_amp) / num_levels
    return (np.digitize(wave, levels, right=False) - 1) * (2 * V_amp / num_levels)

def generate_readings(R, C, start_freq, end_freq, cycle_num, form="sin", ex_cyc=0, save_csv=False):
    V_amp = 3.3 / 2
    
    freqs = []
    res = []
    curr_freq = start_freq
    csv_cols = []
    while curr_freq <= end_freq:
        if form == "sin":
            freqs.append(curr_freq)
            w = 2 * np.pi * curr_freq
            Z = complex(R, -1/(w*C))

            t, V_in = generate_wave(curr_freq, V_amp, form, cycle_num=cycle_num)
            phase = np.angle(- (V_amp / Z))
            I_amp = V_amp / abs(Z)

            I_out = I_out_sin(t, curr_freq, I_amp, phase)
            
            if save_csv:
                csv_cols.extend((t, V_in, I_out))

            I_out_fft = 2 * np.fft.fft(I_out) / len(I_out)
            I_out_fft = I_out_fft[:len(I_out_fft)//2]
            I_crit = np.argmax(abs(I_out_fft))
            
            
            Z_calc = V_amp / I_out_fft[I_crit]
            Z_calc = complex(-Z_calc.imag, Z_calc.real)
            res.append(Z_calc)
            
        
        if form == "triangle":
            freqs.append(curr_freq)
            
            t, V_in = generate_wave(curr_freq, V_amp, form, cycle_num=cycle_num)
            Vr = get_Vr_out(t, cycle_num, freq, V_amp, R, C)
            I_out = Vr / R
            
            n = V_in.size
            if not ex_cyc: ex_cyc = cycle_num - 10
            start_idx = int(np.rint(ex_cyc * n / cycle_num))
            
            I_out_dft = ftt(t[:-start_idx], I_out[start_idx:], 1000) * 3 / I_out[start_idx:].size
            Z_tri = V_amp / I_out_dft
            
            res.append(Z_tri)
        
        curr_freq *= 2

    if save_csv:
        assert 3 * len(freqs) == len(csv_cols), "We should have 3 columns per frequency"
        header = ",".join([f"t_{freq},V_in_{freq},I_out_{freq}" for freq in freqs])
        np.savetxt("single_curve.csv", np.array(csv_cols).T, delimiter=",", header=header, comments="")
    
    return freqs, res

# **********************
# Machine Learning Model Helpers
# **********************

def fit_log(X, y, noise=None):
    
    assert X.shape[0] == y.shape[0], "X dim and y dim must match"
    assert (noise is None) or (noise.shape[0] == X.shape[0]), "Must input noise weight for each data point"
    
    n = X.shape[0]
    
    if isinstance(noise, np.ndarray):
        D = np.diag(1 / noise ** 2)
    else:
        D = np.identity(n)
    
    X = np.log(X)
    X = np.vstack((X, np.ones(X.shape[0]))).T
    w_opt = np.linalg.solve(X.T @ D @ X, X.T @ D @ y)
    return w_opt

def predict_log(x, w):
    """
    General purpose logarithmic evaluator given weights w = [a, b]
    For our purposes, this function maps (concentration) ==> (impedance)
    
    Input: variable x, weights w = (a, b)
    Output: a * log(x) + b
    """
    a, b = w
    return a * np.log(x) + b

def predict_exp(x, w):
    """
    General purpose exponential evaluator given weights w = [a, b]
    This is the inverse of function y = a * log(x) + b
    For our purposes, this function maps (impedance) ==> (concentration)
    
    Input: variable x, weights w = (a, b)
    Output: exp( (y - b) / a )
    """
    a, b = w
    return np.exp( x / a - b / a , dtype=np.float128)

def fit(X, y, model='nitro'):
    """
    Fits a linear model to the data. If X is wide, attempts to fit via an RV decomposition.
    """
    m, n = X.shape
    if model == 'linear':
        if m < n:
            U, d, Vh = np.linalg.svd(X)
            R = U @ np.diag(d)
            V = Vh.T
            w_opt = V[:,:m] @ np.linalg.inv(R.T @ R) @ R @ y

        else:
            w_opt = np.linalg.inv(X.T @ X) @ X.T @ y
        
    elif model == 'nitro':
        assert y is not None and len(y) == m, "Must input valid concentrations"
        
        w = np.empty((n, 2))
        for i in range(n):
            w[i] = fit_log(np.array(y), X[:,i])
            
        return w
        
    return w_opt

def predict(X, w, model='nitro', agg=gmean, exclude_fns=0, return_preds=False):
    """
    Predict concentration given a list of impedances.
    """
    if model == 'linear':
        return X @ w
    
    elif model == 'nitro':
        
        assert agg is not None, "Must input valid aggregation function"
        
        n = len(X) # Number of elements in input vector
        
        preds = np.empty(n - exclude_fns)
        for j in range(w.shape[0] - exclude_fns):
            preds[j] = predict_exp(X[j], w[j])
    
        if return_preds:
            return agg(preds), preds
        else: 
            return agg(preds)


# **********************
# Sample Data Generation
# **********************

# @st.cache_data
def c_to_z(c, freq_idx, weights):
    """
    Wrapper utility function mapping (concentration) ==> (impedance) at a given frequency.
    Simply calls the fitted logarithmic function.
    """
    return predict_log(c, weights[freq_idx])

# @st.cache_data
def freq_sweep_at_c(c, weights, freqs):
    """
    Mimics a biosensing experiment at a single concentration c > 0. Note that the sweep doesn't impart any error.
    Any impedance values are exactly generated from the model.
    
    Input: Concentration C
           Set of weights describing the function (concentration) ==> (impedance) at each frequency.
           
    Output: A list of impedance values, one for each frequency.
    """
    assert c > 0
    return np.array([c_to_z(c, i, weights) for i in range(len(freqs))])

# @st.cache_data
def generate_experiment(concs, weights, freqs):
    """
    Generates an (m x n) data matrix with impedance values.
    Each row represents a single concentration.
    Each col represents a single frequency.
    """
    m = len(concs)
    n = freqs.shape[0]
    Z = np.empty((m, n))
    for i, c in enumerate(concs):
        Z[i] = freq_sweep_at_c(c, weights, freqs)
    return Z

def generate_noise(m, n, noise_at_freq):
    """
    Return matrix of noise at each concentration (rows) and frequency (cols).
    """
    noises = np.empty((m, n))
    zeros = np.zeros(n)
    for i in range(m):
        noises[i] = np.random.normal(zeros, noise_at_freq)
        
    return noises

def avg_noise(n_samples, noise_at_freq, m, n):
    """
    Repeatedly generate noise for n_samples and return the average.
    By the central limit theorem, as n_samples increases, average_noise should return to zero.
    
    n_samples is limited to 1000 to avoid overflow issues, as the noise scale is quite large at around 10^6
    """
    assert n_samples <= 1000
    
    agg = np.zeros((m, n))
    for i in range(n_samples):
        agg += generate_noise(m, n, noise_at_freq) / n_samples
    
    return agg

# **********************
# Data Plotting
# **********************

def dual_axis_fig(x, y, title, xtitle, yname, ylabel):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    n = len(y)

    for i in range(n-1):
        # Add traces
        fig.add_trace(
            go.Line(x=x, y=y[i], name=yname[i]),
            secondary_y=False,
        )

        # Set y-axes titles
        fig.update_yaxes(title_text=ylabel[i], secondary_y=False)
    
    # Add traces
    fig.add_trace(
        go.Line(x=x, y=y[n-1], name=yname[n-1]),
        secondary_y=True,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=ylabel[n-1], secondary_y=True)

    # Add figure title
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=28)
        )
    )

    # Set x-axis title
    fig.update_xaxes(title_text=xtitle, showgrid=True)

    return fig

def plot_rows(M, x, y, label="", xlabel="", ylabel="", title=""):

    fig = make_subplots()
    m = len(M)

    for i in range(m):
        fig.add_trace(
            go.Scatter(x=x, y=M[i], name=label + "=" + str(y[i]), opacity=0.75)
        )
    
    # Add figure title
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=28)
        ),
        autosize=False, width=800, height=600
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text=xlabel, type="log")

    # Set y-axes titles
    fig.update_yaxes(title_text=ylabel, type="log")

    return fig

def avg_scatter(vals, true_val, pred_val, figsize):
    fig, ax = plt.subplots(figsize=figsize)

    plt.scatter([i for i in range(len(vals))], vals, s=75)
    ax.axhline(y=true_val, color='r', linestyle='-', label="True")
    ax.axhline(y=pred_val, color='r', linestyle='--', label="Predicted")
    ax.set_title(f"Predictions at each frequency, true={true_val}", size=22)
    ax.set_xlabel("Index of frequency", size=16)
    ax.set_ylabel("Prediction (Concentration)", size=16)
    ax.legend()

    return fig

def nyquist(freqs, Z):
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=np.rint(Z.real), y=np.rint(-Z.imag), mode="markers")
    )
    fig.update_xaxes(title_text="Z (Real)")
    fig.update_yaxes(title_text="Negative Z (Imaginary)")
    fig.update_layout(title_text="Nyquist Plot")
    return fig

def bode(freqs, Z):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=freqs, y=abs(Z), name="Magnitude")
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", title_text="Magnitude", secondary_y=False)
    fig.add_trace(
        go.Scatter(x=freqs, y=np.angle(Z) / (2 * np.pi) * 360, name="Phase"),
        secondary_y=True
    )
    fig.update_xaxes(title_text="Freq (Hz)")
    fig.update_yaxes(title_text="Phase (deg)", secondary_y=True)
    fig.update_layout(title_text="Bode Plot")
    return fig