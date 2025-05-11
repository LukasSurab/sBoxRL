"""
Helper functions for S-box generation, differential and linear spectra,
and metric evaluation.
"""
from collections import Counter
import operator
from sympy import fwht

# =============================================================================
# Helper Functions for S-box Generation and Analysis
# =============================================================================

def reverse_bits(x):
    """Reverse the order of bits in list x."""
    return x[::-1]

def apply_operations(x, operations):
    """
    Apply a series of gate operations to the bit list x.
    Allowed operations:
      - "XOR": target = target XOR control1.
      - "TOFFOLI": if (control1 AND control2) are both 1, flip target.
      - "NOT": flip the target bit.
      - "FREDKIN": if the control bit is 1, swap bits at target1 and target2.
    """
    for op in operations:
        if op['type'] == 'XOR':
            x[op['target']] ^= x[op['control1']]
        elif op['type'] == 'TOFFOLI':
            if x[op['control1']] & x[op['control2']]:
                x[op['target']] ^= 1
        elif op['type'] == 'NOT':
            x[op['target']] ^= 1
        elif op['type'] == 'FREDKIN':
            if x[op['control']] == 1:
                t1, t2 = op['target1'], op['target2']
                x[t1], x[t2] = x[t2], x[t1]
    return x

def sbox_logic(x, logic_json):
    """Process input bits x through the S-box logic."""
    if logic_json.get("reverse_bits", False):
        x = reverse_bits(x)
    x = apply_operations(x, logic_json.get('operations', []))
    if logic_json.get("reverse_bits", False):
        x = reverse_bits(x)
    return x

def generate_sbox(sbox_logic_func, logic_json):
    """Generate the complete S-box mapping for an 8-bit input (0-255)."""
    sbox_table = []
    for i in range(256):
        input_bits = [(i >> bit) & 1 for bit in range(8)]
        output_bits = sbox_logic_func(input_bits.copy(), logic_json)
        output_value = sum([output_bits[bit] << bit for bit in range(8)])
        sbox_table.append(output_value)
    return sbox_table

def log_sbox_hex(sbox):
    """Format the S-box as a hexadecimal string."""
    hex_sbox = [f'{value:02X}' for value in sbox]
    return "S = [" + ", ".join(f"'{hex_val}'" for hex_val in hex_sbox) + "]"

def xprofile(sb, dx):
    """Compute the XOR difference distribution (DDT) for a given difference dx."""
    N = [0] * len(sb)
    for x in range(len(sb)):
        N[sb[(x ^ dx)] ^ sb[x]] += 1
    return N

def fullxprofile(sb):
    """Compute the full DDT for nonzero differences."""
    return [xprofile(sb, dx) for dx in range(1, len(sb))]

def dspectrum(sb):
    """Compute the differential spectrum from the DDT."""
    ctr = Counter()
    for ddt in fullxprofile(sb):
        ctr += Counter(ddt[1:])  # Skip the zero difference.
    return sorted(ctr.items(), key=operator.itemgetter(0))

def WHTspectrum(S):
    """Compute the Walsh-Hadamard Transform (WHT) spectrum of the S-box."""
    CM = []
    for v in range(1, len(S)):
        transform = fwht([(F & v).bit_count() % 2 for F in S])
        CM.append(transform)
    return CM

def lspectrum(sb):
    """Compute the linear spectrum from the WHT results."""
    ctr = Counter()
    for wht in WHTspectrum(sb):
        ctr += Counter([abs(i) for i in wht[1:]])
    return sorted(ctr.items(), key=operator.itemgetter(0))

def dif_uniformity(ds):
    """Extract the differential uniformity (maximum frequency) from the spectrum."""
    return ds[-1][0]

def nonlinerity(ls):
    """Compute the nonlinearity for an 8-bit S-box."""
    n = 2 ** 7
    return n - ls[-1][0]

def get_sb_props(S):
    """
    Return the cryptographic properties of S-box S.
    Also return 'linearity' (the maximum Walsh coefficient) directly.
    """
    ds_val = dspectrum(S)
    ls_val = lspectrum(S)
    du = dif_uniformity(ds_val)
    nonl = nonlinerity(ls_val)
    lin = ls_val[-1][0]
    return du, ds_val, ls_val, nonl, lin

def evaluate_dspectrum_metrics(ds):
    """
    Given the differential spectrum ds (list of (diff, frequency)),
    return additional metrics:
      - worst_diff: the worst-case difference (du)
      - worst_freq: how many times that worst-case difference appears
      - avg_freq: average frequency over all nonzero differences.
    """
    worst_diff, worst_freq = ds[-1]
    total_freq = sum(freq for diff, freq in ds)
    avg_freq = total_freq / len(ds)
    return worst_diff, worst_freq, avg_freq

def evaluate_lspectrum_metrics(ls):
    """
    Given the linear spectrum ls (list of (abs(W), frequency)),
    return additional metrics:
      - worst_walsh: the highest absolute Walsh coefficient
      - worst_walsh_freq: how many times that worst-case coefficient appears
      - avg_walsh: average frequency over all nonzero Walsh coefficients.
    """
    worst_walsh, worst_walsh_freq = ls[-1]
    total_freq = sum(freq for walsh, freq in ls)
    avg_walsh = total_freq / len(ls)
    return worst_walsh, worst_walsh_freq, avg_walsh