#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyteomics.mzml as mz
import pyteomics.mass as mass
import matplotlib.pyplot as plt
import spectrum_utils.iplot as sup
import spectrum_utils.spectrum as sus
from os import path
import urllib.parse
import pandas

# PSI peak interpretation specification:
# https://docs.google.com/document/d/1yEUNG4Ump6vnbMDs4iV4s3XISflmOkRAyqUuutcCG2w/edit?usp=sharing

from bisect import *
from array import *

# Global Variables
percolatorFile = "../data/crux/crux-output/percolator.target.psms.txt"
fragment_tol_mass = 10
fragment_tol_mode = 'ppm'
max_number_of_aa_considered = 3
single_aa_mass = {aa[0]:comp.mass() for aa, comp in mass.std_aa_comp.items() if len(aa)==1 \
                  and aa != 'I' and aa != 'O' and aa != 'U' and aa != 'J'}
composite_masses = single_aa_mass.copy()
all_masses = single_aa_mass.copy()
for _ in range(1,max_number_of_aa_considered):
    composite_masses = {aa1+aa2:aam1+aam2 for aa1, aam1 in composite_masses.items() for aa2, aam2 in single_aa_mass.items()}
    all_masses = dict(sorted({ **all_masses, **composite_masses}.items(), key=lambda item: item[1]))
aa_mass = array('d',all_masses.values())
aa_ix = list(all_masses.keys())

def readPSMs(fileName, specFileIndex="0", fdrThreshold=0.01):
    scan2charge = {}
    scan2peptide = {}
    with open(fileName,"r") as reader:
        for line in reader:
            words = line.split('\t')
            if words[0] == specFileIndex:
                if float(words[7])<=fdrThreshold:
                    scan = int(words[1])
                    scan2charge[scan] = int(words[2])
                    scan2peptide[scan] = words[10]
                else:
                    break
    return scan2charge, scan2peptide

def find_diff(difference, tolerance, aa_masses = aa_mass, aa_index = aa_ix):
    #print(tolerance)
    #print(difference)
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

def find_corresponding_ion(mzarray, intensities, target_mz, tolerance):
    ixlo = bisect_right(mzarray, target_mz - tolerance)
    ixhigh = bisect_left(mzarray, target_mz + tolerance)
    if ixlo == ixhigh:
        if abs(mzarray[ixlo] - target_mz) < tolerance:
            return intensities[ixlo]
        else:
            return 0
    else:
        return intensities[(ixlo + ixhigh) // 2]


def generate_graph(mzarray, intensities, precursor_mass, max_series_charge=1,
                   fragment_tol_mass=fragment_tol_mass, verbose=False):
    virtual_mz =  mzarray.copy()   
    virtual_intensities =  intensities.copy()

    # Add virtual fragments at start and end of series coresponding to the mass 
    # additions to the b/y ion series
    for charge in range(1,max_series_charge+1):
        c_term_mz = mass.Composition({'O': 1, 'H': 2, 'H+': charge}).mass()/charge
        virtual_mz = np.insert(virtual_mz, 0, c_term_mz)
        virtual_intensities = np.insert(virtual_intensities, 0, -1.)
        end_mz = precursor_mass/charge + c_term_mz
        ix = bisect_left(virtual_mz,end_mz)
        virtual_mz = np.insert(virtual_mz,ix,end_mz)
        virtual_intensities = np.insert(virtual_intensities,ix,-1.)
    n_term_mz = mass.Composition({'H+': 1}).mass() # charge indifferent
    virtual_mz = np.insert(virtual_mz, 0, n_term_mz)
    virtual_intensities = np.insert(virtual_intensities, 0, -1.)
    for charge in range(1,max_series_charge+1):
        end_mz = precursor_mass/charge + n_term_mz
        ix = bisect_left(virtual_mz,end_mz)
        virtual_mz = np.insert(virtual_mz,ix,end_mz)
        virtual_intensities = np.insert(virtual_intensities,ix,-1.)
    _num_peaks = virtual_mz.shape[0]
    _from, _to, _edge_aa, _node_features, _edge_features = [], [], [], [], []
    for charge in range(1,max_series_charge+1):
        for i in range(_num_peaks):
            corresponding_ion = find_corresponding_ion(virtual_mz, virtual_intensities,
                                                       (precursor_mass - virtual_mz[i]),
                                                       fragment_tol_mass*virtual_mz[i]*charge/1.E6)
            lost_h20 = find_corresponding_ion(virtual_mz, virtual_intensities,
                                                       virtual_mz[i] - mass.Composition({'H':2, 'O':1}).mass(),
                                                       fragment_tol_mass*virtual_mz[i]*charge/1.E6)
            lost_amm = find_corresponding_ion(virtual_mz, virtual_intensities,
                                                       virtual_mz[i] - mass.Composition({'N':1, 'H':3}).mass(),
                                                       fragment_tol_mass*virtual_mz[i]*charge/1.E6)
            _node_features.append((corresponding_ion, lost_h20, lost_amm))
            if i == _num_peaks - 1:
                break
            _claimed = []
            for j in range(i+1,_num_peaks):
                _diff = (virtual_mz[j] - virtual_mz[i])*charge
                _aas = find_diff(_diff, fragment_tol_mass*virtual_mz[j]*charge/1.E6)
                if _aas == None:
                    break
                for _aa in _aas:
                    _previous = [_cl for _cl in _claimed if _cl in _aa]
                    if len(_previous) == 0: # else prune
                        _from.append(i)
                        _to.append(j)
                        _edge_aa.append(_aa)
                        _edge_features.append((virtual_mz[j] / virtual_mz[i] * charge) - 1)
                        _claimed.append(_aa)  # TODO to be checked
                        if verbose:
                            print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(virtual_mz[i], virtual_mz[j], _diff, _aa, std_aa_mass[_aa] ))
    return _from, _to, _edge_aa, virtual_mz, virtual_intensities, _node_features, _edge_features


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


def right_path(peptide, f, t, c, orientation, charge, peaks):
    peptide = peptide.replace("I","L") # These amino acids have identical mass
    _ion_type = "b" if orientation == 1 else "y"
    _series_mz_offset = mass.Composition({'H+': 1}).mass() if orientation == 1 else mass.Composition({'O': 1, 'H': 2, 'H+': charge}).mass()/charge
    _ion_nr = 1
    _loc = bisect_left(peaks,_series_mz_offset)
    #_peak_max = bisect_left(peaks,_series_mz_offset+(mass.calculate_mass(sequence=peptide)/charge))
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


def processSpectra(mzFile, scan2charge, scan2peptide, verbose=True, filter_emtpy_graphs=True,
                   max_spectra=None, compute_right_path=False):
    data = [] # return list of dict
    _count = 0
    print("Processing mzfile : {mzfile} ".format(mzfile=mzFile))
    if max_spectra is not None:
        max_spectra = min(max_spectra, len(scan2charge))
        print(len(scan2charge))
    else:
        max_spectra = len(scan2charge)
    print(max_spectra)
    with mz.read(mzFile) as spectra:
        for spectrum in spectra:
            if spectrum["ms level"] == 2:
                scan = spectrum["index"] + 1
                if scan in scan2charge:
                    psmPeptide = scan2peptide[scan]
                    spectrum_data = process_single_spectrum(spectrum, psmPeptide,
                                                            verbose=verbose, plot_spectrum=True,
                                                            compute_right_path=compute_right_path)
                    if filter_emtpy_graphs and len(spectrum_data['from']) == 0:
                        continue
                    _count += 1
                    data.append(spectrum_data)
                    if _count % max(max_spectra // 100, 0.001) == 0:
                        print("% of spectra reading = {perc:2.2f}".format(perc=_count / max_spectra * 100))
                    if max_spectra is not None and _count >= max_spectra:
                        break
    return data




def read_sample_spectrum(mzScan, mzFile, plot_spectrum=True,
                            fragment_tol_mass=fragment_tol_mass,
                            fragment_tol_mode=fragment_tol_mode, psmPeptide=None):
    # Well working example spectrum
    with mz.read(mzFile) as spectra:
        for spectrum in spectra:
            # print (spectrum["scanList"]["scan"][0])
            if spectrum["ms level"] == 2:
                if spectrum["index"] == mzScan - 1:
                    print(spectrum)
                    spectrum_data = process_single_spectrum(spectrum,plot_spectrum=True,
                                                            compute_right_path=True,
                                                            psmPeptide=psmPeptide)
                    return spectrum_data

def process_single_spectrum(spectrum, psmPeptide=None,
                            compute_right_path=False, plot_spectrum=False,
                            filter_peaks=True, verbose=True):
    """
    Given a spectrum dict (from mzml file) process it generating relative
    graph, compute right path and plot the spectrum.

    Parameters
    ----------
    spectrum : dict
        Spectrum dict corresponding to a specific scan
    psmPeptide : str
        Peptide sequence match if available
    compute_right_path : bool
    plot_spectrum : bool
    fliter_peaks : bool
        Perform filtering on peaks ( to be checked / defined better )
    verbose : bool
    """

    precursor = spectrum["precursorList"]['precursor'][0]['selectedIonList']['selectedIon'][0]
    p_mz = float(precursor['selected ion m/z']) # Measured mass to charge ratio of peptide
    p_z = int(precursor['charge state']) # Precursor charge
    p_m = (p_mz)*p_z # Mass of precursor ##### TO BE CHECKED
    if verbose:
        print("Spectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}, z={z}, precursor m/z={mz:1.2f} mass={mass:1.2f}".format(
            spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"],
            z=p_z, mz=p_mz, mass=p_m ))
        if psmPeptide is not None:
            print(f"Mattched to {psmPeptide}")
    mzarray = spectrum['m/z array']
    intensities = spectrum['intensity array']
    peaks = mzarray.copy()
    annotated_spectrum = sus.MsmsSpectrum("my_spectrum", p_mz, p_z,
                                            mzarray, spectrum['intensity array'])

    # Filter peaks [TODO to be checked]
    if filter_peaks:
        annotated_spectrum = (annotated_spectrum.set_mz_range(min_mz=100, max_mz=1400)
                            .filter_intensity(min_intensity=0.05, max_num_peaks=300)
                            .scale_intensity('root'))
    #print(len(annotated_spectrum.mz))
    _from, _to, _edge_aa, peaks, intensities, node_features, edge_features = \
        generate_graph(annotated_spectrum.mz,
                       annotated_spectrum.intensity,
                       p_m, 1,
                       fragment_tol_mass=fragment_tol_mass * 100,
                       verbose=False)
    if len(_edge_aa) == 0:
        print(_from)
        print(peaks)

    if plot_spectrum:
        # Plot spectrum with annotation
        annotated_spectrum = (annotated_spectrum
                                .remove_precursor_peak(fragment_tol_mass,
                                                        fragment_tol_mode)
                                .annotate_proforma(
                                    proforma_str=psmPeptide,
                                    fragment_tol_mass=fragment_tol_mass * 100,
                                    fragment_tol_mode=fragment_tol_mode,
                                    ion_types='by'))
        (sup.spectrum(annotated_spectrum).properties(width=640, height=400)
            .save(path.join('..', 'results', 'tmp', str(spectrum['id']) + '_spectrum_iplot'+
                            str(len(psmPeptide)) + '.html')))
    if compute_right_path:
        right_path(psmPeptide, _from, _to, _edge_aa, 1, 1, peaks)
        right_path(psmPeptide, _from, _to, _edge_aa, -1, 1, peaks)
        # right_path(psmPeptide, _from, _to, _edge_aa, peaks)
        # brute_force(b_from, b_to, b_edge_aa)
        # print(p_m)

    data = {'idx': spectrum['index'],
            'mz_array': peaks,
            'intensities': intensities,
            'p_m': p_m,
            'p_z': p_z,
            'p_mz': p_mz,
            'from': _from,
            'to': _to,
            'edges_aa': _edge_aa,
            'peptide_seq': psmPeptide,
            'node_features' : node_features,
            'edge_features' : edge_features
            }
    return data


if __name__ == '__main__':
    # Process sample spectrum

    mzFile = "../data/converted/LFQ_Orbitrap_DDA_Yeast_01.mzML"
    mzScan = 32688
    psmPeptide = "IANVQSQLEK"
    precursorCharge = 2

    print("Processing sample spectrum")
    # print(f"Recreate {psmPeptide} with mass {mass.calculate_mass(psmPeptide):1.2f}")
    #read_sample_spectrum(mzScan, mzFile)
    #quit()

    # Process multiple spectra
    print("Processing multiple spectra")
    scan2charge, scan2peptide = readPSMs(percolatorFile, "0", 0.01)
    print(len(scan2charge))
    print(len(scan2peptide))
    processSpectra(mzFile, scan2charge, scan2peptide, max_spectra=144, compute_right_path=True)
    #processSpectra(mzFile, scan2charge, scan2peptide)
    quit()
