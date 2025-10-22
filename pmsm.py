import numpy as np
from deductor import DeductorBaseNamed, BaseAttribute, DerivedAttribute, AliasAttribute, ScaledAliasAttribute, DerivateRule, Validator
from foc_base import PointDQ

def flux_from_nominal_motor_mode(R, L, N, vn, Pn, In, Un, sign = 1, J = np.nan):
        # quadratic equation a x^2 + b x + c = 0
        a = 4/9 * (Pn/In)**2
        b = 4/3* N*vn*L * Pn
        c = (R*In)**2 + (N*vn*L*In)**2 + 4/3*Pn*R + 4/9*(Pn/In)**2 - Un**2
        D = b**2 - 4*a*c
        # check if solution exists
        if D < 0:
            return np.nan, None, None, None
            #raise ValueError('nominal mode: no solution')
        ctgI = (-b + sign*np.sqrt(D)) / (2*a)
        # calculate currents
        sinI = np.sqrt(1 / (1 + ctgI**2)) 
        Iq = In * sinI
        Id = np.sqrt(In**2 - Iq**2) * (-1 if ctgI < 0 else 1)
        # calculate rotor flux
        Fm = 2/3 * Pn / (N*vn*Iq)
        # calculate voltages
        Ud = R*Id - N*vn*L*Iq
        Uq = R*Iq + N*vn*L*Id + N*vn*Fm
        # stability analisys
        stable = None
        if not np.isnan(J):
            # Id Iq v
            A = np.array([[  -R/L,     N*vn,         N*Iq ],
                          [ -N*vn,     -R/L,   -N*Id-N*Fm ],
                          [     0, 3/2*N*Fm/J,        0 ]]) 
            if not np.any(np.isnan(A)):
                eigvals, _ = np.linalg.eig(A)
                stable = np.all(np.real(eigvals) < 0)
        # return result
        return Fm, PointDQ(Id, Iq), PointDQ(Ud, Uq), stable

def flux_from_nominal_generator_mode(R, L, N, vn, Pn, In, Un, sign = 1, J = np.nan):
        # quadratic equation a x^2 + b x + c = 0
        a = 4/9 * (Pn/In)**2
        b = 4/3* N*vn*L * Pn
        c = (R*In)**2 + (N*vn*L*In)**2 - 4/3*Pn*R + 4/9*(Pn/In)**2 - Un**2
        D = b**2 - 4*a*c
        # check if solution exists
        if D < 0:
            return np.nan, None, None, None
            #raise ValueError('nominal mode: no solution')
        ctgI = (-b + sign*np.sqrt(D)) / (2*a)
        # calculate currents
        sinI = np.sqrt(1 / (1 + ctgI**2)) 
        Iq = In * sinI
        Id = np.sqrt(In**2 - Iq**2) * (-1 if ctgI < 0 else 1)
        # calculate rotor flux
        Fm = 2/3 * Pn / (N*vn*Iq)
        # calculate voltages
        Ud = R*Id + N*vn*L*Iq
        Uq = R*Iq - N*vn*L*Id - N*vn*Fm
        # stability analisys 
        stable = None
        if not np.isnan(J):
            # Id Iq v
            A = np.array([[  -R/L,     -N*vn,     -N*Iq ],
                          [ N*vn,     -R/L,   N*Id+N*Fm ],
                          [     0, 3/2*N*Fm/J,        0 ]])
            if not np.any(np.isnan(A)):
                eigvals, _ = np.linalg.eig(A)
                stable = np.all(np.real(eigvals) < 0)
        # return result
        return Fm, PointDQ(Id, Iq), PointDQ(Ud, Uq), stable

class _ModelBase(DeductorBaseNamed):
    ''' Incomplete model common for Rotary and Linear motors. '''
    _ATTRIBUTES = [
        BaseAttribute('L', 'H', 'Motor iductance as it present in model.', groups=['inductance']),
        BaseAttribute('L2', 'H', '2-nd motor iductance harmonic.', groups=['inductance']),
        BaseAttribute('R', 'Ohm', 'Resistance, phase resistance (resistance between zero point and phase).', groups=['resistanse']),
        BaseAttribute('Fm', 'Wb', 'Rotor flux linkage', groups=['flux']),
        AliasAttribute('Rph', 'R', groups=['resistanse']),
        ScaledAliasAttribute('Lph', 2.0/3.0, 'L', 'H', 'Measured phase inductnce (between zero point and phase).', groups=['inductance']),
        ScaledAliasAttribute('Lll', 2.0, 'L', 'H', 'Measured line to line inductnce (between two phases).', groups=['inductance']),
        ScaledAliasAttribute('Rll', 2.0, 'R', 'Ohm', 'Measured line to line resistance  (between two phases)', groups=['resistanse']),
        DerivedAttribute('Ld', 'H', 'Inductance along direct axis.', 
                         lambda L, L2: L + L2, groups=['inductance']),
        DerivedAttribute('Lq', 'H', 'Inductance along quadrature axis.', 
                         lambda L, L2: L - L2, groups=['inductance']),
    ]
    _DERIVATE_RULES = [
        DerivateRule('L', lambda Ld, Lq: (Ld + Lq)/2),
        DerivateRule('L2', lambda Ld, Lq: (Ld - Lq)/2),
        DerivateRule('L2', lambda L, Ld: Ld - L),
        DerivateRule('L2', lambda L, Lq: L - Lq),
    ]
    
    def __init__(self, *args, **kwargs):
        super(_ModelBase, self).__init__(*args, **kwargs)
        # set default L2
        if np.isnan(self.L2):
            self.L2 = 0.0

class ModelRotary(_ModelBase):
    _ATTRIBUTES = [
        BaseAttribute('N', None, 'Number of poles pairs.', groups=['poles']),
        BaseAttribute('J', 'kg m^2', 'Rotor inertia.', groups=['inertia']),
        AliasAttribute('n_pole_pairs', 'N', groups=['poles']),
        ScaledAliasAttribute('n_poles', 2, 'N', None, 'Number of poles', groups=['poles']),
        DerivedAttribute('Kemf', 'Vs', 'Meashured back EMF constant: speed in rad/s to phase voltage amplitude.', 
                         lambda Fm, N: N*Fm, groups=['flux']),
        DerivedAttribute('Kt', 'Hm/A', 'Torque constant: current amplitude to force.', 
                         lambda Fm, N: 3/2*N*Fm, groups=['flux']),
        ScaledAliasAttribute('Kemf_llrms_rpm', 2*np.pi/60 * np.sqrt(3/2), 'Kemf', 'V/rpm', 
                             'Meashured back EMF constant: speed in rpms to rms line-to-line voltage.', groups=['flux']),
        ScaledAliasAttribute('Kemf_rpm', 2*np.pi/60, 'Kemf', 'V/rpm', 
                             'Meashured back EMF constant: speed in rpms to phase voltage amplitud.', groups=['flux']),
        ScaledAliasAttribute('Kemf_ll', np.sqrt(3), 'Kemf', 'Vs', 
                             'Meashured back EMF constant: speed in rad to line-to-linr voltage amplitude.', groups=['flux']),
    ]
    _DERIVATE_RULES = [
        DerivateRule('Fm', lambda N, Kt: Kt/(3/2*N)),
        DerivateRule('Fm', lambda N, Kemf: Kemf/N),
    ]

class ModelLinear(_ModelBase):
    _ATTRIBUTES = [
        BaseAttribute('tau', 'm', 'Pole pitch, distance between two poles.', groups=['poles']),
        BaseAttribute('m', 'kg', 'Moving part mass.', groups=['inertia']),
        AliasAttribute('pole_pitch', 'tau', groups=['poles']),
        ScaledAliasAttribute('pole_pair_pitch', 2, 'pole_pitch', 'm', 'Number of poles', groups=['poles']),
        DerivedAttribute('Kemf', 'Vs/m', 'Meashured back EMF constant: speed in m/s to phase voltage amplitude.', 
                         lambda Fm, tau: np.pi/tau*Fm, groups=['flux']),
        DerivedAttribute('Kt', 'H/A', 'Torque constant: current amplitude to force.', 
                         lambda Fm, tau: 3/2*np.pi/tau*Fm, groups=['flux']),
        ScaledAliasAttribute('Kemf_llrms', np.sqrt(3/2), 'Kemf', 'Vs/m', 
                             'Meashured back EMF constant: speed in m/s to rms line-to-line voltage.', groups=['flux']),
        ScaledAliasAttribute('Kemf_ll', np.sqrt(3), 'Kemf', 'Vs/m', 
                             'Meashured back EMF constant: speed in m/s to line-to-linr voltage amplitude.', groups=['flux']),
        # compatibility with rotary model
        DerivedAttribute('N', '1/m', 'Compatibility: pi/tau ', 
                         lambda tau: np.pi/tau, groups=['compatibility']),
        DerivedAttribute('J', 'kg', 'Compatibility: m ', 
                         lambda m: m, groups=['compatibility']),
    ]
    _DERIVATE_RULES = [
        DerivateRule('Fm', lambda tau, Kt: Kt/(3/2*np.pi/tau)),
        DerivateRule('Fm', lambda tau, Kemf: Kemf/(np.pi/tau)),
    ]

class _NominalModeBase(DeductorBaseNamed):
    ''' Incomplete nominal mode specification common for Rotary and Linear motors. '''
    _ATTRIBUTES = [
        BaseAttribute('Un', 'V', 'Rated phase voltage amplitude (beeween phase and zero point).', groups=['rated_voltage','rated']),
        BaseAttribute('In', 'A', 'Rated current amplitude (on phase line).', groups=['rated_current','rated']),
        ScaledAliasAttribute('rated_ac_voltage', np.sqrt(3/2), 'Un', 'V', 'Rated 3-phase rms voltage', groups=['rated_voltage', 'rated']),
        ScaledAliasAttribute('Un_rms', np.sqrt(1/2), 'Un', 'V', 'Rated phase rms voltage'),
        ScaledAliasAttribute('rated_dc_voltage', np.sqrt(3), 'Un', 'V', 'DC bus volatge', groups=['rated_voltage','rated']),
        ScaledAliasAttribute('In_rms', 1.0/np.sqrt(2), 'In', 'A', 'Rated rms phase current (on phase line)'),
        AliasAttribute('rated_current', 'In_rms', groups=['rated_current','rated']),
        AliasAttribute('rated_ac_phase_voltage', 'Un_rms', groups=['rated_voltage','rated']),
    ]

class ModelRotaryNominalMode(ModelRotary, _NominalModeBase): 
    _ATTRIBUTES = [
        BaseAttribute('vn', 'rad/s', 'Rated speed.'),
        BaseAttribute('Tn', 'Nm', 'Rated torque.'),
        AliasAttribute('rated_speed', 'vn', groups=['rated_speed','rated']),
        AliasAttribute('rated_torque', 'Tn', groups=['rated_effort','rated']),
        ScaledAliasAttribute('rated_speed_rpm', 30/np.pi, 'vn', 'rpm', 'Rated speed.', groups=['rated_speed','rated']),
        DerivedAttribute('Pn', 'W', 'Rated power', lambda vn, Tn: vn*Tn),
        DerivedAttribute('fn', 'Hz', 'Rated frequency (electrical).', 
                         lambda N, vn: N*vn/(2*np.pi)),
        AliasAttribute('rated_power', 'Pn', groups=['rated_power', 'rated']),
        AliasAttribute('rated_frequency', 'fn', groups=['rated_speed','rated']),
    ]
    _DERIVATE_RULES = [
        DerivateRule('vn', lambda Pn, Tn: Pn/Tn),
        DerivateRule('vn', lambda N, fn: 2*np.pi*fn/N),
        DerivateRule('Tn', lambda vn, Pn: Pn/vn),
        DerivateRule('Fm', lambda R, L, N, vn, Pn, In, Un: flux_from_nominal_motor_mode(R, L, N, vn, Pn, In, Un, sign = 1, J = np.nan)[0]),
    ]
    _VALIDATORS = [
        Validator(lambda Un_rms, In_rms, Pn: 3*Un_rms*In_rms > Pn, 'Mechanical power output must not be greater then electrical power input')
     ]

class ModelLinearNominalMode(ModelLinear, _NominalModeBase): 
    _ATTRIBUTES = [
        BaseAttribute('vn', 'm/s', 'Rated speed.'),
        BaseAttribute('Fn', 'N', 'Rated force.'),
        AliasAttribute('rated_speed', 'vn', groups=['rated_speed','rated']),
        AliasAttribute('rated_force', 'Fn', groups=['rated_effort','rated']),
        DerivedAttribute('Pn', 'W', 'Rated power', lambda vn, Fn: vn*Fn),
        DerivedAttribute('fn', 'Hz', 'Rated frequency (electrical).', 
                         lambda tau, vn: vn/(2*tau)),
        AliasAttribute('rated_power', 'Pn', groups=['rated_power', 'rated']),
        AliasAttribute('rated_frequency', 'fn', groups=['rated_speed','rated']),
        # compatibility with rotary model
        AliasAttribute('Tn', 'Fn', groups=['compatibility']),
    ]
    _DERIVATE_RULES = [
        DerivateRule('vn', lambda Pn, Fn: Pn/Fn),
        DerivateRule('vn', lambda tau, fn: 2*tau*fn),
        DerivateRule('Fn', lambda vn, Pn: Pn/vn),
        DerivateRule('Fm', lambda R, L, tau, vn, Pn, In, Un: flux_from_nominal_motor_mode(R, L, np.pi/tau, vn, Pn, In, Un, sign = 1, J = np.nan)[0]),
    ]
    _VALIDATORS = [
        Validator(lambda Un_rms, In_rms, Pn: 3*Un_rms*In_rms > Pn, 'Mechanical power output must not be greater then electrical power input')
     ]
