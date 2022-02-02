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

mzFile = "/hd2/lukask/ms/graphtest/PXD028735/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
mzScan = 99217
psmPeptide = "LPNGLEYEQPTGLFINNK"
precursorCharge = 2

print(f"Trying to recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}") 
fragment_tol_mass = 50
fragment_tol_mode = 'ppm'

single_aa_mass = {aa[0]:comp.mass() for aa, comp in mass.std_aa_comp.items() if len(aa)==1 and aa != 'I'}
double_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in single_aa_mass.items() for aa2, aam2 in single_aa_mass.items()}
std_aa_mass = dict(sorted({ **single_aa_mass, **double_aa_mass}.items(), key=lambda item: item[1]))
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
        return None
    if difference >= aa_mass[-1] + tolerance:
        return False
    if difference <= aa_mass[0]:
        return aa_index[0]
    if difference >= aa_mass[-1]:
        return aa_index[-1]
    ix = bisect_left(aa_mass, difference)-1
    #print(ix, len(aa_mass), difference, aa_mass[-1])
    #print(aa_mass[ix], aa_mass[ix+1])
    if abs(difference-aa_mass[ix])<=tolerance:
        return aa_index[ix]
    if abs(difference-aa_mass[ix+1])<=tolerance:
        return aa_index[ix+1]
    return None
    

def generate_graph(mzarray, precursor_mass, n_term_mass=n_term_mass, c_term_mass=c_term_mass):
    #print(mzarray)
    mzarray = np.insert(mzarray, 0, c_term_mass)
    mzarray = np.insert(mzarray, 0, n_term_mass)
    mzarray = np.append(mzarray, [precursor_mass + n_term_mass])
    mzarray = np.append(mzarray, [precursor_mass + c_term_mass])
    _num_peaks = mzarray.shape[0]
    _from, _to, _edge_aa = [], [], []
    for i in range(_num_peaks-1):
        for j in range(i+1,_num_peaks):
            _diff = mzarray[j] - mzarray[i]
            _aa = find_diff(_diff, 0.02)
            if _aa == False:
                break
            if _aa:
                _from.append(i)
                _to.append(j)
                _edge_aa.append(_aa)
                #print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(mzarray[i], mzarray[j], _diff, chr(_aa), std_aa_mass[aa] ))
    _from.append(i)
    _to.append(j)
    _edge_aa.append(_aa)
    return _from, _to, _edge_aa


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
                #print(spectrum)
                print("Spectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}, z={z}, precursor m/z={mz:1.2f} mass={mass:1.2f}".format(
                    spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"], 
                    z=p_z, mz=p_mz, mass=p_m ))
                mzarray = spectrum['m/z array']
                peaks = mzarray.copy()
                _from, _to, _edge_aa = generate_graph(mzarray, p_m)
                brute_force(_from, _to, _edge_aa)
                # print(p_m)
                annotated_spectrum = sus.MsmsSpectrum("my_spectrum", p_mz, p_z,
                            peaks, spectrum['intensity array'])
                            # peptide=psmPeptide)
                # Process the MS/MS spectrum.
                annotated_spectrum = (annotated_spectrum.set_mz_range(min_mz=100, max_mz=1400)
                    .remove_precursor_peak(fragment_tol_mass, fragment_tol_mode)
                    .filter_intensity(min_intensity=0.05, max_num_peaks=50)
                    .scale_intensity('root')
                    .annotate_proforma(proforma_str=psmPeptide,fragment_tol_mass=fragment_tol_mass, fragment_tol_mode=fragment_tol_mode, ion_types='by'))
                # Plot the MS/MS spectrum.
                #fig, ax = plt.subplots(figsize=(12, 6))
                #sup.spectrum(annotated_spectrum, ax=ax)
                #plt.show()
                #plt.close()            
                (sup.spectrum(annotated_spectrum).properties(width=640, height=400)
                       .save('spectrum_iplot.html'))

print(mass.fast_mass(psmPeptide[:1], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:2], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:3], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:4], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:5], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:6], ion_type='b', charge=1))
print(mass.fast_mass(psmPeptide[:7], ion_type='b', charge=1))
