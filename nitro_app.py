import streamlit as st
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from streamlit_option_menu import option_menu
import warnings

st.set_page_config(page_title="Impedance Analyzer",
                   page_icon="images/logo.png",
                   layout="wide")


st.title("Nano-integrated Technology \
                 Research Operations (NiTRO)")

with st.sidebar:

    menu_sel = option_menu(
        "Main Menu",
        ["Simulate", "Preprocess", "Train", "Predict"],
        menu_icon="hexagon",
        default_index=0,
    )

    st.info("This software enables accurate simulation \
            of a dynamic RC circuit when stimulated \
            by sine or triangle waves.")

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
    samp_num = int(duration * samp_rate)
    
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
def I_out_sin(t, freq, I_amp, phase):
    return I_amp * np.sin(freq*(2*np.pi*t) + phase)

@st.cache_data
def get_V_out(V_in, I_out, Rf):
    V_out = V_in - I_out * Rf
    V_out[V_out > 3.3] = 3.3
    V_out[V_out < -3.3] = -3.3
    return V_out

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
        title_text=title
    )

    # Set x-axis title
    fig.update_xaxes(title_text=xtitle, showgrid=True)

    return fig

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

if menu_sel == "Simulate":

    with st.sidebar:
        freq = st.number_input(label="Frequency (Hz)", min_value=100,
                        max_value=100_000_000, value=1000, step=100)
        cycle_num = st.number_input(label="Number of Cycles", min_value=1,
                        max_value=10, value=3, step=1)
        R = st.number_input(label="Resistance (Ohms)", min_value=1,
                        max_value=80_000, value=100, step=100)
        nC = st.number_input(label="Capacitance (nF)", min_value=1,
                        max_value=1000, value=1000, step=100)
        C = nC * 1e-9
        Rf = st.number_input(label="Feedback Resistor (Ohms)", min_value=0,
                        max_value=80_000, value=1000, step=100)

    V_amp = 3.3 / 2
    w = 2 * np.pi * freq
    Z = complex(R, -1/(w*C))

    wave_sel = option_menu(
        "", 
        ["Sine Wave", "Triangle Wave"],
        orientation="horizontal"
    )
    if wave_sel == "Sine Wave":
        with st.expander("Important Sine Wave Details"):
            """
            All calculations were performed based on the circuit
            schematic. Note that the current is in the opposite
            direction compared to a typical RC System, which causes
            I_out to appear flipped from its expected value.

            We can observe that, in a capacitive circuit, the output
            current leads the input voltage. This means that the
            current peak appears to the left of the voltage
            peak. Again, remember that the current I_out is flipped.

            V_out is implemented to rail at 3.3V. Use the feedback
            resistor Rf to influence V_out.
            """

        with st.expander("Mathematical Justifications"):
            r'''
            From first principles, we accept the following propositions:
            $$\\V_{in} - V_{out} = iR_{f}\\
                Z = R + \frac{1}{j \omega C} = R - \frac{1}{\omega C}j\\
                V_g-V_{in}=iZ \implies -\frac{V_{in}}{Z}=i\\
                \text{Current Phase} = \texttt{angle}(-\frac{V_{in}}{Z})\\
                \text{Current Amplitude} = \texttt{mag}(-\frac{V_{in}}{Z})
            $$
            Notice that the phase angle is negative because the current
            leads voltage in a capacitive circuit.

            Decreasing frequency and capacitance increases the phase shift,
            while increasing resistance has the opposite effect.
            '''

        with st.spinner("Calculating..."):
            t, V_in = generate_wave(freq, V_amp, form="sin", 
                                    cycle_num=cycle_num)
            
            phase = np.angle(- (V_amp / Z))
            I_amp = V_amp / abs(Z)
            
            I_out = I_out_sin(t, freq, I_amp, phase)
            V_out = get_V_out(V_in, I_out, Rf)
            fig = dual_axis_fig(t, [V_in, V_out, I_out], 
                                "V_in and I_out: Sine Wave", "Time", 
                                ["V_in", "V_out", "I_out"], 
                                ["Volts", "Volts", "Amps"])
            st.plotly_chart(fig, use_container_width=True)


    if wave_sel == "Triangle Wave":
        with st.expander("Important Triangle Wave Details"):
            """
            All calculations were performed based on the circuit
            schematic. Note that the current is in the opposite
            direction compared to a typical RC System, which causes
            I_out to appear flipped from its expected value.

            This implementation uses the fact that a triangle wave
            is simply a sequence of rising and falling ramp inputs. 
            Therefore, we can determine the voltage across the resistor
            by solving a differential equation with various initial states.
            Note that the output current is directly related to this resistor 
            voltage via Ohm's Law. 

            Because there is no identifiable pattern, each half cycle is
            generated using a switching for loop. This reveals interesting 
            patterns, such as a buildup charging effect on V_out. I also provide
            Vc, the capacitor voltage, as a sanity check.

            V_out is implemented to rail at 3.3V. Use the feedback
            resistor Rf to influence V_out.
            """

        with st.expander("Mathematical Justifications"):
            st.warning("Under construction")

        with st.spinner("Calculating..."):
            t, V_in = generate_wave(freq, V_amp, form="triangle",
                                    cycle_num=cycle_num)
            
            Vr = get_Vr_out(t, cycle_num, freq, V_amp, R, C)
            I_out = Vr / R
            Vc = V_in - Vr
            V_out = get_V_out(V_in, I_out, Rf)

            fig1 = dual_axis_fig(t, [V_in, V_out, I_out], 
                                "Triangle Wave Circuit Readings", "Time",
                                ["V_in", "V_out", "I_out"],
                                ["Volts", "Volts", "Amps"])
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = dual_axis_fig(t, [Vc],
                                "Voltage Across Capacitor", "Time",
                                ["Vc"],
                                ["Volts"])
            st.plotly_chart(fig2, use_container_width=True)
    
elif menu_sel == "Preprocess":

    st.warning("Under construction...")

elif menu_sel == "Train":

    st.warning("Under construction...")

elif menu_sel == "Predict":

    st.warning("Under construction...")
