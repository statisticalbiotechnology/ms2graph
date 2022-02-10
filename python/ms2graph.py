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

# Global Variables
percolatorFile = "../data/crux-output/percolator.target.psms.txt"
fragment_tol_mass = 10
fragment_tol_mode = 'ppm'
single_aa_mass = {aa[0]:comp.mass() for aa, comp in mass.std_aa_comp.items() if len(aa)==1 and aa != 'I'}
double_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in single_aa_mass.items() for aa2, aam2 in single_aa_mass.items()}
tripple_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in double_aa_mass.items() for aa2, aam2 in single_aa_mass.items()}

# Quad aa masses makes tha graph probably too dense TODO to be discussed
# quad_aa_mass = {aa1+aa2:aam1+aam2 for aa1, aam1 in double_aa_mass.items() for aa2, aam2 in double_aa_mass.items()}
# std_aa_mass = dict(sorted({ **single_aa_mass, **double_aa_mass, **tripple_aa_mass, **quad_aa_mass}.items(), key=lambda item: item[1]))

std_aa_mass = dict(sorted({ **single_aa_mass, **double_aa_mass, **tripple_aa_mass}.items(), key=lambda item: item[1]))
#std_aa_mass = dict(sorted(single_aa_mass.items(), key=lambda item: item[1]))
aa_mass = array('d',std_aa_mass.values())
aa_ix = list(std_aa_mass.keys())

def readPSMs(fileName, specFileIndex="0", fdrTreshold=0.01):
    scan2charge = {}
    scan2peptide = {}
    with open(fileName,"r") as reader:
        for line in reader:
            words = line.split('\t')
            if words[0] == specFileIndex:
                if float(words[7])<=fdrTreshold:
                    scan = int(words[1])
                    scan2charge[scan] = int(words[2])
                    scan2peptide[scan] = words[10]
                else:
                    break
    return scan2charge, scan2peptide

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
    

def generate_graph(mzarray, precursor_mass, series_charge=1,
                   fragment_tol_mass=fragment_tol_mass, verbose=False):
    #print(mzarray)
    n_term_mass = mass.Composition({'H+': series_charge}).mass()
    c_term_mass = mass.Composition({'O': 1, 'H': 2, 'H+': series_charge}).mass()
    mzarray = np.insert(mzarray, 0, c_term_mass)
    mzarray = np.insert(mzarray, 0, n_term_mass)
    mzarray = np.append(mzarray, [precursor_mass + n_term_mass])
    mzarray = np.append(mzarray, [precursor_mass + c_term_mass])
    _num_peaks = mzarray.shape[0]
    _from, _to, _edge_aa = [], [], []
    for i in range(_num_peaks-1):
        _claimed = []
        for j in range(i+1,_num_peaks):
            _diff = mzarray[j] - mzarray[i]
            _aas = find_diff(_diff, fragment_tol_mass*mzarray[j]/1.E6)
            if _aas == None:
                break
            for _aa in _aas:
                _previous = [_cl for _cl in _claimed if _aa.startswith(_cl)]
                if len(_previous) == 0:
                    _from.append(i)
                    _to.append(j)
                    _edge_aa.append(_aa)
                    _claimed.append(_aa)  # TODO to be checked
                    if verbose:
                        print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(mzarray[i], mzarray[j], _diff, _aa, std_aa_mass[_aa] ))
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

def oriented_match(long_str, short_str, orientation):
    if orientation == 1:
        return long_str.startswith(short_str)
    else:
        return long_str.endswith(short_str)

def right_path(peptide, f, t, c, orientation, peaks):
    peptide = peptide.replace("I","L")
    _ion_type = "b" if orientation == 1 else "y"
    _ion_nr = 1
    _peak_max = list(set(t))[-2 if orientation ==1 else -1 ] # peak for last b/y-ion
    _loc = 0 if orientation == 1 else 1
    # matching forward
    while _ion_nr <= len(peptide):
        if orientation == 1:
            residues = peptide[_ion_nr-1:min(_ion_nr+3,len(peptide))]
        else:
            residues = peptide[max(0,len(peptide) - _ion_nr - 3):len(peptide) - _ion_nr + 1]
        print(f"From peak at {peaks[_loc]:1.3f} (peak index {_loc}) searching string {residues} i.e. the {_ion_type}{_ion_nr}-ion")
        _indeces = [i for i, x in enumerate(f) if x == _loc]
        # print(_b_indeces, [c[_ix] for _ix in _b_indeces])
        _to = [t[_ix] for _ix in _indeces if oriented_match(residues,c[_ix], orientation)]
        _char = [c[_ix] for _ix in _indeces if oriented_match(residues,c[_ix], orientation)]
        if len(_to)==0:
            print(f"No match for {_ion_type} ion nr={_ion_nr}, residues={residues}")
            remainder = peptide[_ion_nr-1:len(peptide)] if orientation == 1 else peptide[0:len(peptide) - _ion_nr + 1]
            print(f"Matching remainding {remainder}")
            break
        print("{_ion_type} matches ", _char, _to, [peaks[ix] for ix in _to ])
        _loc = _to[0] # Take the first (lightest)
        _ion_nr += len(_char[0]) # move forward according to length of match


def processSpectra(mzFile, scan2charge, scan2peptide):
    with mz.read(mzFile) as spectra:
        for spectrum in spectra:
            if spectrum["ms level"]==2:
                scan = spectrum["index"] + 1
                if scan in scan2charge:
                    psmPeptide = scan2peptide[scan]
                    precursor = spectrum["precursorList"]['precursor'][0]['selectedIonList']['selectedIon'][0]
                    p_mz = float(precursor['selected ion m/z'])
                    p_z = int(precursor['charge state'])
                    p_m = (p_mz-mass.Composition({'H+': 1}).mass())*p_z
                    print("Spectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}, z={z}, precursor m/z={mz:1.2f} mass={mass:1.2f}".format(
                        spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"], 
                        z=p_z, mz=p_mz, mass=p_m ))
                    print(f"Mattched to {psmPeptide}")
                    mzarray = spectrum['m/z array']
                    peaks = mzarray.copy()
                    _from, _to, _edge_aa, peaks = generate_graph(peaks, p_m, series_charge=1)
                    right_path(psmPeptide, _from, _to, _edge_aa, 1, peaks)
                    right_path(psmPeptide, _from, _to, _edge_aa, -1, peaks)


def process_sample_spectrum(mzScan, mzFile, plot_spectrum=True,
                            fragment_tol_mass=fragment_tol_mass,
                            fragment_tol_mode=fragment_tol_mode):
    # Well working example spectrum
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

                    # Filter peaks [TODO to be checked]
                    annotated_spectrum = sus.MsmsSpectrum("my_spectrum", p_mz, p_z,
                                                          mzarray, spectrum['intensity array'])
                    annotated_spectrum = (annotated_spectrum.set_mz_range(min_mz=100, max_mz=1400)
                                          .filter_intensity(min_intensity=0.05, max_num_peaks=50)
                                          .scale_intensity('root'))

                    _from, _to, _edge_aa, peaks = generate_graph(annotated_spectrum.mz,
                                                                 p_m, series_charge=1,
                                                                 fragment_tol_mass=fragment_tol_mass,
                                                                 verbose=False)
                    if plot_spectrum:
                        # Plot spectrum with annotation
                        annotated_spectrum = (annotated_spectrum
                                              .remove_precursor_peak(fragment_tol_mass,
                                                                     fragment_tol_mode)
                                              .annotate_proforma(
                                                  proforma_str=psmPeptide,
                                                  fragment_tol_mass=fragment_tol_mass,
                                                  fragment_tol_mode=fragment_tol_mode,
                                                  ion_types='by'))
                        (sup.spectrum(annotated_spectrum).properties(width=640, height=400)
                         .save('spectrum_iplot.html'))
                        right_path(psmPeptide, _from, _to, _edge_aa, 1, peaks)
                        right_path(psmPeptide, _from, _to, _edge_aa, -1, peaks)
                        # right_path(psmPeptide, _from, _to, _edge_aa, peaks)
                        # brute_force(b_from, b_to, b_edge_aa)
                        # print(p_m)
                    data = {'idx': spectrum['index'],
                            'mz_array': peaks,
                            'intensities': annotated_spectrum.intensity,
                            'p_m': p_m,
                            'p_z': p_z,
                            'p_mz': p_mz,
                            'from': _from,
                            'to': _to,
                            'edges_aa': _edge_aa}
                    return data


if __name__ == '__main__':
    # Process sample spectrum
    print("Processing sample spectrum")

    mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
    mzScan = 32688
    psmPeptide = "IANVQSQLEK"
    precursorCharge = 2

    print(f"Recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}")
    process_sample_spectrum(mzScan, mzFile)
    quit()

    # Process multiple spectra
    scan2charge, scan2peptide = readPSMs(percolatorFile, specFileIndex="0", fdrTreshold=0.01)
    processSpectra(mzFile, scan2charge, scan2peptide)
    quit()
