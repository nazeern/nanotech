import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import warnings
from utils.funcs import *

def calculate_d_edl():
    return np.sqrt((k_b * T * e_0 * e_r) / (2 * (z * e)**2 * N_a * C_0))

def calculate_edl_capacitance(d_edl):
    return e_0 * e_r / d_edl

st.set_page_config(page_title="Impedance Analyzer",
                   page_icon="images/logo.png",
                   layout="wide")


st.title("Nano-integrated Technology \
                 Research Operations (NiTRO)")

with st.sidebar:

    "[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/nazeern/nanotech/blob/main/nitro_app.py) \
    View Source Code"

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
                V_out = digitize(V_out + V_amp, 4096) - V_amp

            fig = dual_axis_fig(t, [V_in, I_out], 
                                "V_in and I_out: Sine Wave", "Time", 
                                ["V_in", "I_out"], 
                                ["Volts", "Amps"])
            mode = "markers" if digit else "lines"
            fig.add_trace(
                go.Scatter(x=t, y=V_out, name="V_out", mode=mode)
            )
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
                V_out = digitize(V_out + V_amp, 4096) - V_amp

            fig1 = dual_axis_fig(t, [V_in, I_out], 
                                "Triangle Wave Circuit Readings", "Time",
                                ["V_in", "I_out"],
                                ["Volts", "Amps"])
            mode = "markers" if digit else "lines"
            fig1.add_trace(
                go.Scatter(x=t, y=V_out, name="V_out", mode=mode)
            )
            
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
        with st.spinner("Simulating full experiment..."):
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

        fig_preds = avg_scatter(preds, true_conc, agg_pred, figsize=(6, 4))

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

        st.info("""
        Goal: Predict the sample concentration that generated the 
        new impedance curve identified by the dashed black line.
        """)

        f"""
        ### True Concentration: {true_conc}\n
        ### Algorithm Output: {agg_pred}
        """

        # with st.expander("Important Details"):

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_noisy, use_container_width=True)
            st.expander("View Algorithm Internals") \
              .pyplot(fig_preds, use_container_width=True)

        with col2:
            st.plotly_chart(fig_true, use_container_width=True)
        

    elif source_sel == "Simulated Data":
        
        curve_sel = option_menu(
            "",
            ["Single Curve", "Multi-curve"],
            orientation="horizontal"
        )

        if curve_sel == "Single Curve":
            with st.sidebar:
                advanced_settings = st.checkbox("Advanced Settings")
                R = st.number_input(label="Resistance (Ohms)", min_value=1,
                            max_value=80_000, value=80000, step=100)
                if advanced_settings:
                    with st.form("calc_cap_form"):
                        submitted = st.form_submit_button("Update")
                        #Nanotube geometry
                        CNT_radius = st.number_input("CNT Radius (m)", value=5.0E-09, format="%f") #m
                        CNT_height = st.number_input("CNT Height (m)", value=2.5E-04, format="%f") #m

                        #Nanotube forest geometry
                        nanostructure_width = st.number_input("Nanostructure Width (m)", value=3.0E-06, format="%f") #m
                        nanostructure_length = st.number_input("Nanostructure Length (m)", value=2.54E-02, format="%f") #m
                        gap_between_nanostructures = st.number_input("Gap Between Nanostructures (m)", value=2.0E-06, format="%f") #m

                        #Chip geometry
                        chip_length = st.number_input("Chip Length (m)", value=2.54E-02, format="%f") #1 inch
                        chip_width = st.number_input("Chip Width (m)", value=2.54E-02, format="%f") #1 inch

                        #Dielectric properties
                        epsilon_r = st.number_input("Epsilon R", 1000) #average dielectric constant between two forests
                        E_breakdown = st.number_input("E Breakdown (V/m)", value=1.2E+09, format="%f") # dielectric breakdown E-field [V/m]

                        #Substrate geometry
                        Si_thickness = st.number_input("Si Thickness (m)", value=3.0E-04, format="%f") #m
                        SiO2_thickness = st.number_input("SiO2 Thickness (m)", value=2.1E-06, format="%f") #m
                        Metal_1_thickness = st.number_input("Metal 1 Thickness (m)", value=1.0E-07, format="%f") #m
                        Metal_2_thickness = st.number_input("Metal 2 Thickness (m)", value=1.0E-08, format="%f") #m
                        Catalyst_thickness = st.number_input("Catalyst Thickness (m)", value=1.0E-08, format="%f") #m

                        #Physical constants (SI units)
                        e = st.number_input("Electron Charge", value=1.602e-19, format="%f")  #electron charge
                        z = st.number_input("Electrons / Surface Particle", value=1) #electrons/surface particle
                        C = st.number_input("C", value=1.0E-15, format="%f") # nmols/liter
                        C_0 = st.number_input("C_0", value=1.0E-12, format="%f")
                        e_r = st.number_input("Dielectric Constant", value=78.49, format="%f") #dielctric constant
                        e_0 = st.number_input("Vacuum Permittivity", value=8.854E-12, format="%f") #vacuum permittivity
                        k_b= st.number_input("Boltzmann Constant", value=1.38E-23, format="%f", disabled=True) #Boltzmann const
                        T = st.number_input("Temperature", value=298.1, format="%f") #room temperature
                        V_zeta = st.number_input("V_zeta", value=5.0E-02, format="%f")
                        N_a = st.number_input("Avogadro's Number", value=6E+23, format="%f", disabled=True)
                    C = calculate_edl_capacitance(calculate_d_edl())
                    st.write("Capacitance (nF): ", C * 1e9)
                else:
                    nC = st.number_input(label="Capacitance (nF)", min_value=1,
                                    max_value=1000, value=1, step=100)
                    C = nC * 1e-9

            with st.spinner("Spooling Virtual Device..."):
                freqs, Z = generate_readings(R=R, C=C, start_freq=100, end_freq=10_000_000, 
                                             cycle_num=5, form="sin", ex_cyc=5-1, save_csv=False)

            Z = np.array(Z)
            bode_fig = bode(freqs, Z)
            nyquist_fig = nyquist(freqs, Z)

            # Front-end Rendering
            st.info("""
                Run an entire impedance curve collection cycle. A virtual device
                generates an input voltage wave and stimulates an output current
                response. Our impedance analysis algorithm processes these raw
                waves into an array of impedance values, drawing an impedance curve.
                The processing algorithm also extracts phase information, which allows
                accurate Nyquist and Bode Plots.

                Important: This software is not simply implementing equations, it 
                generates impedance curves from **raw voltage and current waveforms**
                generated by our virtual device.
            """)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(nyquist_fig, use_container_width=True)
            with col2:
                st.plotly_chart(bode_fig, use_container_width=True)

        
        elif curve_sel == "Multi-curve":
            NUM_CAPS = 8
            with st.sidebar:
                R = st.number_input(label="Resistance (Ohms)", min_value=1,
                            max_value=80_000, value=80000, step=100)
                with st.form("cap_form"):
                    submitted = st.form_submit_button("Update")
                    nC_vals = [ st.number_input(label=f"Capacitance {i+1} (nF)", min_value=1,
                                    max_value=1000, value=i+1, step=100, key=i)
                    for i in range(NUM_CAPS)]
                C_vals = [nC * 1e-9 for nC in nC_vals]

            Z_vals = []
            bar = st.progress(0)
            with st.spinner("Spooling Virtual Device..."):
                for i, C in enumerate(C_vals):
                    freqs, Z = generate_readings(R=R, C=C, start_freq=100, end_freq=10_000_000, cycle_num=5, 
                                                form="sin", ex_cyc=5-1)
                    Z_vals.append(Z)
                    bar.progress((i + 1) / NUM_CAPS)

            Z = np.array(Z_vals)
            fig = plot_rows(abs(Z), freqs, C_vals, label="Capacitance=", xlabel="Freq", 
                  ylabel="Impedance", title="Actual Impedance vs. Concentration")

            # Front-end Rendering
            st.info("""
                Run an entire impedance curve collection cycle. A virtual device
                generates an input voltage wave and stimulates an output current
                response. Our impedance analysis algorithm processes these raw
                waves into an array of impedance values, drawing an impedance curve.

                Important: This software is not simply implementing equations, it 
                generates impedance curves from **raw voltage and current waveforms**
                generated by our virtual device.
            """)
            st.plotly_chart(fig)


    elif source_sel == "Uploaded Data":

        with st.expander("Please Read: CSV Format Details"):
            """
            The CSV input should contain experiment values taken at different frequencies.
            For each frequency, we require three inputs: time, input voltage, and output current.
            These inputs should be in the columns of the CSV file, and should obviously be the same length.

            For example, if you run three experiments at frequencies 100, 1000, and 10000, you will have 9 columns.
            The first three columns are t_100, V_in_100, I_out_100. 
            The next three columns are t_1000, V_in_1000, I_out_1000, and so on.

            After uploading a correctly formatted CSV, the software will provide the entire electrochemical impedance
            spectroscopy, calculating impedance and displaying a Bode and Nyquist plot. Don't worry, both of these are just different
            ways to graph impedance.
            """

        uploaded_file = st.file_uploader("Import CSV")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            column_names = df.columns.to_list()
            num_cols = len(column_names)
            assert num_cols % 3 == 0, "Expected columns in groups of 3 (t, V_in, I_out)"

            freqs = []
            Z = []

            for i in range(0, len(column_names), 3):
                freq = int(column_names[i].split("_")[-1])
                t = df.iloc[:,i].to_numpy()
                V_in = df.iloc[:,i+1].to_numpy()
                I_out = df.iloc[:,i+2].to_numpy()
                V_amp = np.max(V_in)

                I_out_fft = 2 * np.fft.fft(I_out) / len(I_out)
                I_out_fft = I_out_fft[:len(I_out_fft)//2]
                I_crit = np.argmax(abs(I_out_fft))
            
            
                Z_calc = V_amp / I_out_fft[I_crit]
                Z_calc = complex(-Z_calc.imag, Z_calc.real)
                Z.append(Z_calc)
                freqs.append(freq)

            Z = np.array(Z)
            bode_fig = bode(freqs, Z)
            nyquist_fig = nyquist(freqs, Z)
        
            # Render data
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(nyquist_fig, use_container_width=True)
            with col2:
                st.plotly_chart(bode_fig, use_container_width=True)



elif menu_sel == "Predict":

    st.warning("Under construction...")
