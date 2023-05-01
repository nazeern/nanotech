import numpy as np
import hvplot.pandas
import pandas as pd
import panel as pn
import param
pn.extension(sizing_mode="stretch_width", throttled=True)

class SimRC(param.Parameterized):
    
    # Amplitude of Input Voltage Wave
    amp = 3.3 / 2
    
    # Frequency & Amplitude of sin wave
    freq = param.Integer(default=1000, bounds=(1, 100000), label="Frequency (Hz)")

    # Desired number of wave oscillations
    cycle_num = param.Integer(default=2, bounds=(1, 10), label="Number of Cycles")

    # Resistance in Ohms
    R = param.Integer(default=100, bounds=(1, 80000), label="Resistance (Ohms)")

    # Capacitance in Nanofarads
    nC = param.Integer(default=1000, bounds=(1, 1000), label="Capacitance (nF)")
    
    # Variable Resistor Rf
    Rf = param.Integer(default=1000, bounds=(1, 80000), label="Feedback Resistor (Ohms)")
    
    # Expected impedance
    Z = -1
    
    # Input Voltage Wave
    V_in = np.array([0])
    
    # Time domain values
    t = np.array([0])
    
    # Output Current Wave
    I_out = np.array([0])
    
    # Output Voltage Wave
    V_out = np.array([0])
    
    V_dft = I_dft = np.array([0])
    V_amp = I_amp = 0
    
    # Calculated Impedance
    Z_calc = -1
    
    @param.depends('freq', 'R', 'nC', watch=True)
    def set_Z(self):
        C = self.nC * 1e-9
        w = 2*np.pi*self.freq
        self.Z = complex(self.R, -1/(w*C))
      
    @param.depends('cycle_num', 'freq', 'R', 'nC', 'Rf', watch=True)
    def set_V_in(self):
        # Duration of sin wave in seconds
        duration = self.cycle_num / self.freq

        # Sampling rate of device
        samp_rate = self.freq * 100
        
        samp_num = int(duration * samp_rate)
        self.t = np.arange(0, duration, duration / samp_num)
        self.V_in = self.amp * np.sin(self.freq*(2*np.pi*self.t))
        
    @param.depends('V_in', 'Z', watch=True)
    def set_I_out(self):
        
        phase = np.angle(- (self.amp / self.Z)) + np.pi
        I_amp = self.amp / abs(self.Z)
        self.I_out = I_amp * np.sin(self.freq*(2*np.pi*self.t) - phase)
        
    @param.depends('V_in', 'I_out', watch=True)
    def set_dfts(self):
        
        V_dft = 2 * np.fft.fft(self.V_in) / len(self.V_in)
        self.V_dft = V_dft[:len(V_dft)//2]
        self.V_amp = np.max(abs(self.V_dft))
        
        I_dft = 2 * np.fft.fft(self.I_out) / len(self.I_out)
        self.I_dft = I_dft[:len(I_dft)//2]
        I_crit = np.argmax(abs(self.I_dft))
        self.I_amp = self.I_dft[I_crit]
        
        self.Z_calc = self.V_amp / self.I_amp
        self.Z_calc = complex(self.Z_calc.imag, self.Z_calc.real)
        
    @param.depends('Rf', 'V_in', 'I_out', watch=True)
    def set_V_out(self):
        self.V_out = self.V_in - self.I_out * self.Rf
        self.V_out[self.V_out > 12] = 12
        self.V_out[self.V_out < -12] = -12
            

    @param.depends('Z')
    def view_Z(self):
        return pn.pane.Markdown(f"#### Expected Impedance: {self.Z}")
    
    @param.depends('V_in', 'I_out')
    def view_Z_calc(self):
        return pn.pane.Markdown(f"#### Calculated Impedance: {self.Z_calc}")
    
    @param.depends('V_in', 'I_out', 'V_out')
    def view_plots(self):
        df = pd.DataFrame()
        df["Time (Sec)"] = pd.Series(self.t)
        df["Voltage In (V)"] = pd.Series(self.V_in)
        df["Current (A)"] = pd.Series(self.I_out)
        df["Voltage Out (V)"] = pd.Series(self.V_out)
        
        voltages = df.hvplot(x='Time (Sec)', 
                         y=['Voltage In (V)', 'Voltage Out (V)'],
                        title='Input and Output Voltage',
                        shared_axes=False,
                         height=300, responsive=True)
        current = df.hvplot(x='Time (Sec)', 
                         y='Current (A)', 
                         title='Output Current',
                         subplots=True,
                         shared_axes=False,
                         color="green",
                         height=300, responsive=True)
        return (voltages + current).cols(1)
    
    @param.depends('V_dft', 'I_dft')
    def view_dfts(self):
        
        df = pd.DataFrame()
        df["Voltage Amplitude (V)"] = pd.Series(abs(self.V_dft))
        df["Current Amplitude (A)"] = pd.Series(abs(self.I_dft))
        
        return df.hvplot(y=['Voltage Amplitude (V)', 'Current Amplitude (A)'],
                                 subplots=True,
                                 shared_axes=False,
                                height=300, responsive=True).cols(1)
    
    
    

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

tabs = pn.Tabs(
    ('Waveforms', simRC.view_plots),
    ('Fourier Transform', simRC.view_dfts)
)
# plots = pn.Row(
#           simRC.view_plots,
#           simRC.view_dfts)

bootstrap.main.append(m_card)
bootstrap.main.append(Z_card)
bootstrap.main.append(tabs)

bootstrap.servable()
