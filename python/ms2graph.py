#!/usr/bin/env python
import numpy as np
import pyteomics.mzml as mz
import pyteomics.mass as mass
from bisect import *
from array import *

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
n_term_mass = mass.Composition({'H+': 1}).mass()
c_term_mass = mass.Composition({'O': 1, 'H': 2, 'H+': 1}).mass()

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
    

def generate_graph(mzarray, precursor_mass):
    #print(mzarray)
    _from, _to, _edge_aa = [], [], []
    for i in range(mzarray.shape[0]-1):
        for j in range(i+1,mzarray.shape[0]):
            _diff = mzarray[j] - mzarray[i]
            _aa = find_diff(_diff, 0.05)
            if _aa == False:
                break
            if _aa:
                _from.append(i)
                _to.append(j)
                _edge_aa.append(_aa)
                #print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(mzarray[i], mzarray[j], _diff, chr(_aa), std_aa_mass[aa] ))
    return _from, _to, _edge_aa


def brute_force(f,t,c,goto=0,peptide=""):
    _largest = max(t)
    for _from, _to, _char  in zip(f,t,c):
        if _from == goto:
            _current_peptide = peptide + _char
            if _to == _largest:
                print (_current_peptide)
                # print (_current_peptide, mass.calculate_mass(_current_peptide))
                return
            brute_force(f,t,c,_to,_current_peptide)

    

#print(mass.std_aa_comp)
#print(mass.std_aa_mass)
with mz.read('/hd2/lukask/ms/marcom_lf/converted/20210528_HF2_01_MM_1_P01.mzML') as spectra:
    for spectrum in spectra:
        # print (spectrum)
        # print (spectrum["scanList"]["scan"][0])
        if spectrum["ms level"]==2:
            if spectrum["index"]==3531:
                precursor = spectrum["precursorList"]['precursor'][0]['selectedIonList']['selectedIon'][0]
                p_mz = float(precursor['selected ion m/z'])
                p_z = float(precursor['charge state'])
                p_m = p_mz*p_z
                #print(spectrum)
                #print("Spectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}".format(
                #    spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"] ))
                mzarray = spectrum['m/z array']
                mzarray = np.insert(mzarray, 0, n_term_mass)
                mzarray = np.append(mzarray, [p_m - c_term_mass])
                _from, _to, _edge_aa = generate_graph(mzarray, p_m)
                brute_force(_from, _to, _edge_aa)
                # print(p_m)
