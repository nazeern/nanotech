import streamlit as st
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
import warnings
from utils.funcs import *


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

if menu_sel == "Simulate":

    with st.sidebar:
        digit = st.checkbox("Digitize V_out")

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

            Selecting "Digitize V_out" will alter the output of V_out 
            between 0.0 and 3.3V.
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
            if digit:
                V_out = digitize(V_out + V_amp, 4096)

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

            Selecting "Digitize V_out" will alter the output of V_out 
            between 0.0 and 3.3V.
            """

        with st.expander("Mathematical Justifications"):
            r'''
            Notice that $V_{in}$ is a sequence of rising and falling
            ramp inputs. Therefore, for a ramp input, $V_{in}(t) = 
            \alpha t - A$, where $\alpha = 4fA$ and $A = 
            \frac{V_{max}-V_{min}}{2}$. This can be derived using 
            basic slope and intercept techniques. Now, we derive the
            voltage across the resistor, $V_R$:
            $$
            V_{in} = V_R + V_C \\
            V_{in} = \frac{1}{C}\int_0^ti dt + iR \\
            \frac{d}{dt}V_{in} = \frac{i}{C} + \frac{d}{dt}V_R \\
            = \frac{V_R}{RC}+\frac{d}{dt}V_R=\pm\alpha \\ 
            \frac{dV_R}{dt} = \frac{\alpha RC - V_R}{RC} \\
            \int\frac{1}{\alpha RC - V_R}dV_R = \int\frac{1}{RC}dt
            $$
            Now, integrate, letting $B$ be the integration constant
            as $C$ is reserved for capacitance:
            $$
            ln(\alpha RC - V_R) + B = -\frac{t}{RC} \\
            V_R = \alpha RC - Be^{-\frac{t}{RC}}
            $$
            Now, we find B using the initial condition of $V_R = V_0$ at
            time $t = 0$. Substituting, we find:
            $$
            V_0 = \alpha RC - B \implies B = \alpha RC + V_0
            $$
            Now, just plug B into the original equation:
            $$
            V_R = \alpha RC - (\alpha RC - V_0)e^{-\frac{t}{RC}}
            $$
            After this, calculating $I_{out}$, $V_{out}$ and $V_C$ are 
            simple:
            $$
            V_C = V_{in} - V_R \\
            I_{out} = -\frac{V_R}{R} \\
            V_{out} = V_{in} - I_{out}R_f
            $$
            '''

        with st.spinner("Calculating..."):
            t, V_in = generate_wave(freq, V_amp, form="triangle",
                                    cycle_num=cycle_num)
            
            Vr = get_Vr_out(t, cycle_num, freq, V_amp, R, C)
            I_out = Vr / R
            Vc = V_in - Vr
            V_out = get_V_out(V_in, I_out, Rf)
            if digit:
                V_out = digitize(V_out + V_amp, 4096)

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

    source_sel = option_menu(
        "Choose Input Source", 
        ["Sample Data", "Simulated Data", "Uploaded Data"],
        orientation="horizontal"
    )

    if source_sel == "Sample Data":

        # Sample data is loaded from .npy files
        concs = [c / 100 for c in range(1, 100, 10)] + [1]
        freqs = np.load("data/sample_model_freqs.npy", allow_pickle=True)
        m = len(concs)
        n = len(freqs)
        
        noise = np.load("data/sample_data_noise.npy", allow_pickle=True)
        noise_at_freq = np.mean(noise, axis=0)

        weights = np.load("data/sample_model_weights.npy", allow_pickle=True)

        with st.sidebar:
            train_noise_scale = st.slider("Training Noise Scale",
                                          0.0, 1.0, value=0.4, step=0.01)
            n_samples = st.slider("Number of Samples",
                                  1, 64, value=9, step=1)
            test_noise_scale = st.slider("Test Noise Scale",
                                         0.0, 1.0, value=0.01, step=0.01)
            true_conc = st.slider("True Concentration",
                                  float(min(concs)), float(max(concs)), 
                                  value=0.18, step=0.01)

        # Average N samples of noise
        noises = avg_noise(n_samples, noise_at_freq, m, n)
        X_true = generate_experiment(concs, freqs=freqs, weights=weights)
        y_true = np.array(concs)
        X = X_true + train_noise_scale * noises

        # Plotly graphs
        fig_true = plot_rows(X_true, freqs, y_true, label="conc=", xlabel="Freq", 
                  ylabel="Impedance", title="Actual Impedance vs. Concentration")
        fig_noisy = plot_rows(X, freqs, y_true, label="conc=", xlabel="Freq", 
                  ylabel="Impedance", title="Noisy Impedance vs. Concentration")
        
        # Fit algorithm weights to noisy data
        fit_w = fit(X, concs)
        noises = test_noise_scale * np.random.normal(0, noise_at_freq)

        # Generate impedance at test concentration and add noise
        X_test = freq_sweep_at_c(true_conc, weights=weights, freqs=freqs)
        fig_true.add_trace(go.Scatter(x=freqs, y=X_test, name=f"True Conc={true_conc}",
                                      line=dict(color="black", dash="dash")))
        X_test = X_test + noises
        fig_noisy.add_trace(go.Scatter(x=freqs, y=X_test, name=f"True Conc={true_conc}",
                                      line=dict(color="black", dash="dash")))


        # Predict noisy test concentration
        agg_pred, preds = predict(X_test, fit_w, agg=gmean, exclude_fns=0, return_preds=True)

        # *******************
        # Front-end Rendering
        # *******************

        st.info("""
        The data generation below is based on real-world data gathered from a strain 
        of yeast known as Saccharomyces cerevisiae. A state-of-the-art nano-device
        collects impedance curves of the yeast at various concentrations, and our 
        novel algorithm is able to accurately predict the unknown concentration of a
        separate yeast sample.
        """)

        f"""
        ### True Concentration: {true_conc}\n
        ### Algorithm Output: {agg_pred}
        ***

        The dotted black line is the concentration to be predicted.
        """

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_noisy, use_container_width=True)

        with col2:
            st.plotly_chart(fig_true, use_container_width=False)

    








elif menu_sel == "Predict":

    st.warning("Under construction...")
