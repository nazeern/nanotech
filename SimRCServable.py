import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import param
pn.extension(sizing_mode="stretch_width", throttled=True)

class SimRC(param.Parameterized):
    
    # Amplitude of Input Voltage Wave
    amp = 3.3 / 2
    
    # Frequency & Amplitude of sin wave
    freq = param.Integer(default=1, bounds=(1, 10000), label="Frequency (Hz)")

    # Desired number of wave oscillations
    cycle_num = param.Integer(default=1, bounds=(1, 10), label="Number of Cycles")

    # Resistance in Ohms
    R = param.Integer(default=1, bounds=(1, 10000), label="Resistance (Ohms)")

    # Capacitance in Nanofarads
    nC = param.Integer(default=1, bounds=(1, 1000), label="Capacitance (nF)")
    
    # Expected impedance
    Z = 0
    
    # Input Voltage Wave
    V_in = np.array([0])
    
    # Time domain values
    t = np.array([0])
    
    # Output Current Wave
    I_out = np.array([0])
    
    V_dft = I_dft = np.array([0])
    V_amp = I_amp = 0
    
    # Calculated Impedance
    Z_calc = 0
    
    @param.depends('freq', 'R', 'nC', watch=True)
    def set_Z(self):
        C = self.nC * 10e-9
        self.Z = abs(complex(self.R, -1/(self.freq*C)))
      
    @param.depends('cycle_num', 'freq', watch=True)
    def set_V_in(self):
        # Duration of sin wave in seconds
        duration = self.cycle_num / self.freq

        # Sampling rate of device
        samp_rate = self.freq * 100
        
        samp_num = int(duration * samp_rate)
        self.t = np.linspace(0, duration, num=samp_num)
        self.V_in = self.amp * np.sin(self.freq*(2*np.pi*self.t))
        
    @param.depends('V_in', watch=True)
    def set_I_out(self):
        self.I_out = (- self.V_in) / self.Z
        
    @param.depends('V_in', 'I_out', watch=True)
    def set_dfts(self):
        V_dft = 2 * np.abs(np.fft.fft(self.V_in)) / len(self.V_in)
        self.V_dft = V_dft[:len(V_dft)//2]
        self.V_amp = np.max(self.V_dft)
        
        I_dft = 2 * np.abs(np.fft.fft(self.I_out)) / len(self.I_out)
        self.I_dft = I_dft[:len(I_dft)//2]
        self.I_amp = np.max(self.I_dft)
        
        self.Z_calc = self.V_amp / self.I_amp
        

    @param.depends('Z')
    def view_Z(self):
        return pn.pane.Markdown(f"#### Expected Impedance: {self.Z}")
    
    @param.depends('V_in', 'I_out')
    def view_Z_calc(self):
        return pn.pane.Markdown(f"#### Calculated Impedance: {self.Z_calc}")
    
    @param.depends('V_in', 'I_out')
    def view_plots(self):
        fig, ax = plt.subplots(2, 1, constrained_layout=True)

        fig.suptitle(f"V_in and I_out at freq = {self.freq}")
        fig.supxlabel("Time (Seconds)")
        ax[0].set_title(f"V_in")
        ax[0].plot(self.t, self.V_in)

        ax[1].set_title(f"I_out")
        ax[1].plot(self.t, self.I_out)
        plt.close()
        return fig
    
    @param.depends('V_in', 'I_out')
    def view_dfts(self):
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        

        fig.suptitle(f"Fourier Transform of V_in and I_out")
        fig.supxlabel("Frequency (Hz)")
        fig.supylabel("Amplitude")

        ax[0].set_title(f"V_in")
        ax[0].stem(self.V_dft)

        ax[1].set_title(f"I_out")
        ax[1].stem(self.I_dft)
        plt.close()
        return fig
    
    
    

simRC = SimRC()

# Serve the web application
bootstrap = pn.template.BootstrapTemplate(title="SimRC")

bootstrap.sidebar.append(simRC.param)

message = pn.pane.Markdown("""
    # RC Circuit Impedance Software v1.0
    
    <font size=4>A Fourier Transform implementation that enables reliable impedance calculations on input voltage waveforms.</font>

    #### This software will be helpful if:
    * <font size=4>You want to simulate input voltage and output current waveform readings on a simulated RC circuit</font>
    * <font size=4>You require high-accuracy impedance readings given raw input voltage and output current data</font>

    #### Instructions
    * <font size=4>Use the sliders on the sidebar to change input voltage frequency, number of analyzed cycles,
    the resistance, and the capacitance of the RC circuit. By default, the voltage amplitude is 1.65 volts.</font>
    * <font size=4>Don't forget the capacitance is in nanofarads (nF)! You can collapse this cell using the arrow
    in the upper left corner.</font>

    #### Contact

    <font size=4>If you would like to contact me, please reach out at <nitin.nazeer@gmail.com>. I will try to best to respond to all emails.</font>

    #### License

    <font size=4>This project holds the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html). All modifications must attain this license
    and remain open-source.</font>
""")

m_card = pn.Card(message)

Z_card = pn.Card(simRC.view_Z, simRC.view_Z_calc)

plots = pn.Row(
          simRC.view_plots,
          simRC.view_dfts)

bootstrap.main.append(m_card)
bootstrap.main.append(Z_card)
bootstrap.main.append(plots)

bootstrap.servable()