#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyteomics.mzml as mz
import pyteomics.mass as mass
import matplotlib.pyplot as plt
import spectrum_utils.iplot as sup
import spectrum_utils.spectrum as sus
import urllib.parse


from bisect import *
from array import *

mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
mzScan = 32688
psmPeptide = "IANVQSQLEK"
precursorCharge = 2

print(f"Trying to recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}") 
fragment_tol_mass = 10
fragment_tol_mode = 'ppm'

single_aa_mass = {aa[0]:comp.mass() for aa, comp in mass.std_aa_comp.items() if len(aa)==1 and aa != 'I'}
double_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in single_aa_mass.items() for aa2, aam2 in single_aa_mass.items()}
tripple_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in double_aa_mass.items() for aa2, aam2 in single_aa_mass.items()}
quad_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in double_aa_mass.items() for aa2, aam2 in double_aa_mass.items()}
std_aa_mass = dict(sorted({ **single_aa_mass, **double_aa_mass, **tripple_aa_mass, **quad_aa_mass}.items(), key=lambda item: item[1]))
#std_aa_mass = dict(sorted(single_aa_mass.items(), key=lambda item: item[1]))
aa_mass = array('d',std_aa_mass.values())
aa_ix = list(std_aa_mass.keys())
#print(aa_mass)
#print([chr(aa) for aa in aa_ix])
#print(mass.std_aa_comp)
#n_term_mass = mass.fast_mass("G", ion_type='b', charge=1)-aa_mass[0]
#c_term_mass = mass.fast_mass("G", ion_type='y', charge=1)-aa_mass[0]
#print(n_term_mass, c_term_mass)
frag_charge = 1 # We look at the b1+ and y1+ series
n_term_mass = mass.Composition({'H+': frag_charge}).mass()
c_term_mass = mass.Composition({'O': 1, 'H': 2, 'H+': frag_charge}).mass()

def find_diff(difference, tolerance, aa_masses = aa_mass, aa_index = aa_ix):
    if difference <= aa_mass[0] - tolerance:
        return []
    if difference >= aa_mass[-1] + tolerance:
        return None
    if difference <= aa_mass[0]:
        return aa_index[0]
    if difference >= aa_mass[-1]:
        return aa_index[-1]
    ixlo = bisect_right(aa_mass, difference-tolerance)
    ixhigh = bisect_left(aa_mass, difference+tolerance)
    #print(ix, len(aa_mass), difference, aa_mass[-1])
    #print(aa_mass[ix], aa_mass[ix+1])
    return aa_index[ixlo:ixhigh]
    

def generate_graph(mzarray, precursor_mass, n_mass=n_term_mass, c_mass=c_term_mass):
    #print(mzarray)
    mzarray = np.insert(mzarray, 0, c_mass)
    mzarray = np.insert(mzarray, 0, n_mass)
    mzarray = np.append(mzarray, [precursor_mass + n_mass])
    mzarray = np.append(mzarray, [precursor_mass + c_mass])
    _num_peaks = mzarray.shape[0]
    _from, _to, _edge_aa = [], [], []
    for i in range(_num_peaks-1):
        for j in range(i+1,_num_peaks):
            _diff = mzarray[j] - mzarray[i]
            _aas = find_diff(_diff, fragment_tol_mass*mzarray[j]/1.E6)
            if _aas == None:
                break
            for _aa in _aas:
                _from.append(i)
                _to.append(j)
                _edge_aa.append(_aa)
                #print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(mzarray[i], mzarray[j], _diff, chr(_aa), std_aa_mass[aa] ))
    return _from, _to, _edge_aa, mzarray


def brute_force(f,t,c,goto=0,peptide=""):
    _largest = max(t)
    for _from, _to, _char  in zip(f,t,c):
        if _from == goto:
            _current_peptide = peptide + _char
            if _to >= _largest-1:
                # print (_current_peptide)
                print (_current_peptide, mass.calculate_mass(_current_peptide))
                return
            brute_force(f,t,c,_to,_current_peptide)

def right_path(peptide, f, t, c, peaks):
    peptide = peptide.replace("I","L")
    _b_max = list(set(t))[-2] # last b-ion
    _b_loc, _y_loc = 0, list(set(f))[1]  # initiating to first b- and y-ion
    # matching forward
    _pos = 0
    while _pos < len(peptide):
        residues = peptide[_pos:min(_pos+4,len(peptide))]
        print(f"From pos {_pos} and peak at {peaks[_b_loc]:1.3f} (peak index {_b_loc}) searching string {residues} i.e. the b{_pos+1}-ion")
        _b_indeces = [i for i, x in enumerate(f) if x == _b_loc]
        # print(_b_indeces, [c[_ix] for _ix in _b_indeces])
        _b_to = [t[_ix] for _ix in _b_indeces if residues.startswith(c[_ix])]
        _b_char = [c[_ix] for _ix in _b_indeces if residues.startswith(c[_ix])]
        if len(_b_to)==0:
            print(f"No match for b pos={_pos}, residues={residues}")
            break
        print("b matches ", _b_char,_b_to, [peaks[ix] for ix in _b_to ])
        _b_loc = _b_to[0] # Take the first (lightest)
        _pos += len(_b_char[0]) # move forward according to lentht of match
    # matching backward
    _pos = len(peptide)
    while _pos > 0:
        residues = peptide[max(0,_pos-4):_pos]
        print(f"From pos {_pos} and peak at {peaks[_y_loc]:1.3f} (peak index {_y_loc}) searching string {residues} i.e. the y{len(peptide)-_pos+1}-ion")
        _y_indeces = [i for i, x in enumerate(f) if x == _y_loc]
        if len(_y_indeces)==0:
            print(f"No start of a match for y pos={_pos}, residues={residues}")
            break
        # print([c[_ix] for _ix in _y_indeces], [peaks[t[ix]] for ix in _y_indeces ])
        _y_to = [t[_ix] for _ix in _y_indeces if c[_ix] and residues.endswith(c[_ix])]
        _y_char = [c[_ix] for _ix in _y_indeces if c[_ix] and residues.endswith(c[_ix])]
        if len(_y_to)==0:
            print(f"No match for y pos={_pos}, residues={residues}")
            break
        print("y matches ", _y_char, [peaks[ix] for ix in _y_to ])
        _y_loc = _y_to[0] # Take the first (lightest)
        _pos -= len(_y_char[0]) # move backward according to length of match

    

#print(mass.std_aa_comp)
#print(mass.std_aa_mass)
with mz.read(mzFile) as spectra:
    for spectrum in spectra:
        # print (spectrum)
        # print (spectrum["scanList"]["scan"][0])
        if spectrum["ms level"]==2:
            if spectrum["index"]==mzScan-1:
                precursor = spectrum["precursorList"]['precursor'][0]['selectedIonList']['selectedIon'][0]
                p_mz = float(precursor['selected ion m/z'])
                p_z = int(precursor['charge state'])
                p_m = (p_mz-mass.Composition({'H+': 1}).mass())*p_z
                print("Spectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}, z={z}, precursor m/z={mz:1.2f} mass={mass:1.2f}".format(
                    spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"], 
                    z=p_z, mz=p_mz, mass=p_m ))
                mzarray = spectrum['m/z array']
                peaks = mzarray.copy()
                # Plot spectrum with annotation
                annotated_spectrum = sus.MsmsSpectrum("my_spectrum", p_mz, p_z,
                            mzarray, spectrum['intensity array'])
                annotated_spectrum = (annotated_spectrum.set_mz_range(min_mz=100, max_mz=1400)
                    .remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
                    .filter_intensity(min_intensity=0.05, max_num_peaks=50)
                    .scale_intensity('root')
                    .annotate_proforma(proforma_str=psmPeptide,fragment_tol_mass=fragment_tol_mass, fragment_tol_mode=fragment_tol_mode, ion_types='by'))
                (sup.spectrum(annotated_spectrum).properties(width=640, height=400)
                       .save('spectrum_iplot.html'))
                _from, _to, _edge_aa, peaks = generate_graph(peaks, p_m, n_term_mass, c_term_mass)
                right_path(psmPeptide, _from, _to, _edge_aa, peaks)
                # brute_force(b_from, b_to, b_edge_aa)
                # print(p_m)
