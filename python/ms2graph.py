#!/usr/bin/env python
import numpy as np
import pandas as pd
from pyteomics.auxiliary.structures import Charge
import pyteomics.mzml as mz
import pyteomics.mass as mass
import matplotlib.pyplot as plt
import spectrum_utils.iplot as sup
import spectrum_utils.spectrum as sus
from os import path, scandir, listdir
import urllib.parse
import pandas
from itertools import combinations_with_replacement, groupby
from utils import Y_OFFSET_FUNCTIONS, Y_INVERSE_OFFSET_FUNCTIONS, B_OFFSET_FUNCTIONS, B_INVERSE_OFFSET_FUNCTIONS
import spectrum as Spectrum
import graph_utils
import time
import datetime

# PSI peak interpretation specification:
# https://docs.google.com/document/d/1yEUNG4Ump6vnbMDs4iV4s3XISflmOkRAyqUuutcCG2w/edit?usp=sharing

from bisect import *
from array import *

# Global Variables
percolatorFile = "../data/crux/crux-output/percolator.target.psms.txt"
fragment_tol_mass = 10
fragment_tol_mode = 'ppm'

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

def find_diff(difference, tolerance, aa_masses, aa_index):
    if difference <= aa_masses[0] - tolerance:
        return []
    if difference >= aa_masses[-1] + tolerance:
        return []
    if difference <= aa_masses[0]:
        return [aa_index[0]]
    if difference >= aa_masses[-1]:
        return [aa_index[-1]]
    ixlo = bisect_right(aa_masses, difference-tolerance)
    ixhigh = bisect_left(aa_masses, difference+tolerance)
    #print(ix, len(aa_mass), difference, aa_mass[-1])
    #print(aa_mass[ix], aa_mass[ix+1])
    return aa_index[ixlo:ixhigh]

def find_corresponding_ion(mzarray, intensities, target_mz, tolerance):
    ixlo = bisect_right(mzarray, target_mz - tolerance)
    ixhigh = bisect_left(mzarray, target_mz + tolerance)
    if ixlo >= len(mzarray):
        return 0
    if ixlo == ixhigh:
        if abs(mzarray[ixlo] - target_mz) < tolerance:
            return intensities[ixlo]
        else:
            return 0
    else:
        return intensities[(ixlo + ixhigh) // 2]

def generate_aa_masses(max_aa_considered=2):
    single_aa_mass = {aa[0]:comp.mass() for aa, comp in mass.std_aa_comp.items() if len(aa)==1 \
                  and aa != 'I' and aa != 'O' and aa != 'U' and aa != 'J'}
    composite_masses = single_aa_mass.copy()
    all_masses = single_aa_mass.copy()
    for _ in range(1,max_aa_considered):
        composite_masses = {aa1+aa2:aam1+aam2 for aa1, aam1 in composite_masses.items() for aa2, aam2 in single_aa_mass.items()}
    all_masses = dict(sorted({ **all_masses, **composite_masses}.items(), key=lambda item: item[1]))
    aa_mass = array('d',all_masses.values())
    aa_ix = list(all_masses.keys())
    return aa_mass, aa_ix

def generate_graph(mzarray, intensities, precursor_mass, max_series_charge=1,
                   fragment_tol_mass=fragment_tol_mass, verbose=False, max_miss_aa=2):
    max_series_charge = 3
    aa_mass, aa_ix = generate_aa_masses(max_miss_aa)

    print("Max series charge " + str(max_series_charge))
    virtual_mz =  mzarray.copy()   
    virtual_intensities =  intensities.copy()

    precursor_mass = precursor_mass - mass.Composition({'O': 1, 'H': 2}).mass()

    # Given a mz value remap it to actual prefix/suffix value for a set of possible offsets
    y_offset_combinations = list(combinations_with_replacement(range(\
                                len(Y_INVERSE_OFFSET_FUNCTIONS)), 2))

    b_offset_combinations = list(combinations_with_replacement(range(\
                                len(B_INVERSE_OFFSET_FUNCTIONS)), 2))
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

    # Compute offset values


    for charge_i in range(1,max_series_charge+1):
        for charge_j in range(1,max_series_charge+1):

            for i in range(_num_peaks):
                _b_peak_i_offset_values = [f(virtual_mz[i], charge_i) for f in B_INVERSE_OFFSET_FUNCTIONS]
                _y_peak_i_offset_values = [f(virtual_mz[i], charge_i) for f in Y_INVERSE_OFFSET_FUNCTIONS]
                _peak_i_offset_values = _b_peak_i_offset_values + _y_peak_i_offset_values
                if charge_j == 1:
                    corresponding_ion = find_corresponding_ion(virtual_mz, virtual_intensities,
                                                (precursor_mass - virtual_mz[i] * charge_i),
                                                fragment_tol_mass*virtual_mz[i]*charge_i/1.E6)

                    _features = list(map(lambda x: find_corresponding_ion(virtual_mz, virtual_intensities,
                                                    x, fragment_tol_mass*virtual_mz[i]*charge_i/1.E6),
                                    _peak_i_offset_values))


                    _features.append(corresponding_ion)
                    _node_features.append(np.array(_features))

                if i == _num_peaks - 1:
                    break
                _claimed = []
                for j in range(1,_num_peaks):
                    if i == j:
                        continue

                    _b_peak_j_offset_values = [f(virtual_mz[j], charge_j) for f in B_INVERSE_OFFSET_FUNCTIONS]
                    _y_peak_j_offset_values = [f(virtual_mz[j], charge_j) for f in Y_INVERSE_OFFSET_FUNCTIONS]
                    _aas_b = [find_diff(_b_peak_j_offset_values[k] - _b_peak_i_offset_values[l],
                                        fragment_tol_mass*_b_peak_j_offset_values[k]*charge_j/1.E6,
                                        aa_masses=aa_mass, aa_index=aa_ix)\
                            for (k,l) in b_offset_combinations]
                    _aas_y = [find_diff(_y_peak_j_offset_values[k] - _y_peak_i_offset_values[l],
                                        fragment_tol_mass*_b_peak_j_offset_values[k]*charge_j/1.E6,
                                        aa_masses=aa_mass, aa_index=aa_ix)\
                            for (k,l) in y_offset_combinations]
                    _aas = _aas_b + _aas_y
                    #if (len(sum(_aas, [])) > 0):
                    #    with open("log.log", "a+") as f:
                    #        f.write("\n" + str(i) + " " + str(j) + " " + str(charge_i) + " " + str(charge_j))
                    #        f.write(str(list(zip(_aas, _offset_combinations))))
                    _aas = sum(_aas, [])

                    if _aas == None:
                        break
                    _aas = list(set(_aas))
                    for _aa in _aas:
                        _previous = [_cl for _cl in _claimed if _cl in _aa]
                        if len(_previous) == 0: # else prune
                            _from.append(i)
                            _to.append(j)
                            _edge_aa.append(_aa)
                            _edge_features.append((virtual_mz[j] * charge_j / virtual_mz[i]*charge_i) - 1)
                            if verbose:
                                print("Peaks {0}, {1} have a mass diff of {2:1.3f}, i.e. a {3} of mass {4:1.3f}".format(virtual_mz[i], virtual_mz[j], _diff, _aa, std_aa_mass[_aa] ))


    _node_features = [sum(_node_features[i : i + max_series_charge + 1]) for i in \
                          range(0, len(_node_features), max_series_charge)]

    # Filter redundancy
    _from, _to, _edge_aa = graph_utils.remove_redundancy(_from, _to, _edge_aa)
    return _from, _to, _edge_aa, virtual_mz, virtual_intensities, np.array(_node_features), _edge_features


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


def right_path(peptide, f, t, c, orientation, charge, peaks, max_miss_aa=2):
    peptide = peptide.replace("I","L") # These amino acids have identical mass
    _ion_type = "b" if orientation == 1 else "y"
    _series_mz_offset = mass.Composition({'H+': 1}).mass() if orientation == 1\
        else mass.Composition({'O': 1, 'H': 2, 'H+': charge}).mass()/charge
    _ion_nr = 1
    _loc = bisect_left(peaks,_series_mz_offset)
    #_peak_max = bisect_left(peaks,_series_mz_offset+(mass.calculate_mass(sequence=peptide)/charge))
    # matching forward
    while _ion_nr <= len(peptide):
        if orientation == 1:
            residues = peptide[_ion_nr-1:
                               min(_ion_nr+max_miss_aa - 1,
                               len(peptide))]
        else:
            residues = peptide[max(0,len(peptide) - _ion_nr - max_miss_aa - 1):
                               len(peptide) - _ion_nr + 1]
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


path_table = {}
def right_path_all_paths(peptide, f, t, c, orientation, charge, peaks, start_ion, visited=[],
                         verbose=False, first_recursion=False, max_miss_aa=2, start_time=None):
    if first_recursion:
        for key in list(path_table.keys()):
            del path_table[key]
    peptide = peptide.replace("I","L") # These amino acids have identical mass
    _ion_type = "b" if orientation == 1 else "y"
    _loc = start_ion

    if abs(time.time() - start_time) > datetime.timedelta(minutes=1).total_seconds():
        return [[]]
    if _loc not in path_table:
        path_table[_loc] = {}
    else:
        if peptide in path_table[_loc]:
            return path_table[_loc][peptide]

    #_peak_max = bisect_left(peaks,_series_mz_offset+(mass.calculate_mass(sequence=peptide)/charge))
    # matching forward
    if verbose:
        print(f"Searching for peptide {peptide} (len = {len(peptide)}) starting from {start_ion} = {peaks[start_ion]}")
    if len(peptide) == 0:
        return [[]]

    if orientation == 1:
        residues = peptide[:min(max_miss_aa,
                                len(peptide))]
    else:
        residues = peptide[max(0,len(peptide) - max_miss_aa):]
    _indeces = [i for i, x in enumerate(f) if x == _loc]

    _to = [t[_ix] for _ix in _indeces if oriented_match(residues, c[_ix], orientation)]
    _char = [c[_ix] for _ix in _indeces if oriented_match(residues, c[_ix], orientation)]
    _edges = [_ix for _ix in _indeces if oriented_match(residues, c[_ix], orientation)]



    visited = list(visited)
    visited.extend(_indeces)
    if len(_to) == 0 :
        path_table[_loc][peptide] = [[]]
        return [[]]
    all_paths = []
    for i, _node in enumerate(_to):
        if _node in visited:
            continue
        _remainder = peptide[min(len(peptide), len(_char[i])):] if orientation == 1 else\
            peptide[:max(0, len(peptide) - len(_char[i]))]
        _remainder_length = len(_remainder) # save in variable for resource usage purposes
        _node_paths = right_path_all_paths(_remainder, f, t, c,
                                           orientation, charge, peaks, _node, visited,
                                           verbose=verbose, start_time=start_time)
        # print(_remainder + ' | ' + str(_node_paths))
        _node_paths = list(filter(lambda x: len("".join([c[k] for k in x])) == _remainder_length,
                                  _node_paths))
        # matches = ["".join([c[k] for k in path]) for path in _node_paths]
        # print(_remainder + ' | ' + str(matches))
        # print(_remainder + ' | ' + str(_node_paths))
        if orientation == 1:
            for path in _node_paths:
                path.insert(0, _edges[i])
        else:
            for path in _node_paths:
                path.append(_edges[i])
        all_paths.extend(_node_paths)
    path_table[start_ion][peptide]  = all_paths
    return all_paths


def processSpectra(mzFile, scan2charge, scan2peptide, verbose=True, filter_emtpy_graphs=True,
                   max_spectra=None, compute_right_path=False, filtering_max_miss=None,
                   pickle_dir=None, y_edges=False, filter_max_len=None, log=False, overwrite=True,
                   only_stats=False, compute_completness=True, compute_n_paths=False, max_miss_aa=2):
    data = [] # return list of dict
    _count = 0
    print("\n---------\nProcessing mzfile : {mzfile} ".format(mzfile=mzFile))
    if max_spectra is not None:
        max_spectra = min(max_spectra, len(scan2charge))
        print(len(scan2charge))
    else:
        max_spectra = len(scan2charge)
    print(f"Mass Spectra = {max_spectra}")
    with mz.read(mzFile) as spectra:
        for spectrum in spectra:
            if spectrum["ms level"] == 2:
                scan = spectrum["index"] + 1
                if scan in scan2charge:
                    psmPeptide = scan2peptide[scan]
                    if filter_max_len is not None:
                        if len(psmPeptide) > filter_max_len:
                            continue
                    if not overwrite:
                        if path.isfile(path.join(pickle_dir, str(scan)) + '.pickle') or \
                           (len(list(listdir(path.join(pickle_dir)))) > 0 and \
                            scan < max([int(x.split('.')[0]) for x in list(listdir(path.join(pickle_dir)))])):
                            print("Skip scan : {scan}".format(scan=scan))
                            continue
                    spectrum_data = process_single_spectrum(spectrum, psmPeptide,
                                                            verbose=verbose, plot_spectrum=False,
                                                            compute_right_path=compute_right_path,
                                                            compute_completness=compute_completness,
                                                            max_miss_aa=max_miss_aa)
                    if filtering_max_miss is not None and 'b_max_miss' in spectrum_data and 'y_max_miss' in spectrum_data:
                        b_ions_threshold, y_ions_threshold = filtering_max_miss
                        b_miss, y_miss = spectrum_data['b_max_miss'], spectrum_data['y_max_miss']
                        if b_ions_threshold < b_miss or y_ions_threshold < y_miss:
                            print(f"Skipping scan {spectrum_data['idx']}")
                            continue
                    if filter_emtpy_graphs and len(spectrum_data['from']) == 0:
                        continue
                    _count += 1
                    print(f"Number of elements : {_count}")
                    
                    _spectrum = Spectrum.Spectrum(spectrum_data, pickle_dir=pickle_dir,
                                                  y_edges=y_edges, only_stats=only_stats,
                                                  compute_completness=compute_completness,
                                                  compute_n_paths=compute_n_paths)
                    if pickle_dir is not None:
                        data.append(scan)
                    else:
                        data.append(_spectrum)

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
                            filter_peaks=True, verbose=True, compute_completness=True,
                            max_miss_aa=2):
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
    p_m = (p_mz - mass.Composition({'H+': 1}).mass())*p_z # Mass of precursor
    if verbose:
        print("\n------\nSpectrum {0}, MS level {ms_level} @ RT {scan_time:1.2f}, z={z}, precursor m/z={mz:1.2f} mass={mass:1.2f}".format(
            spectrum["id"], ms_level=spectrum["ms level"], scan_time=spectrum["scanList"]["scan"][0]["scan start time"],
            z=p_z, mz=p_mz, mass=p_m ))
        if psmPeptide is not None:
            print(f"Mattched to {psmPeptide}")
    mzarray = spectrum['m/z array']
    intensities = spectrum['intensity array']
    peaks = mzarray.copy()
    annotated_spectrum = sus.MsmsSpectrum("my_spectrum", p_mz, p_z,
                                            mzarray, spectrum['intensity array'])

    if filter_peaks:
        annotated_spectrum = (annotated_spectrum.set_mz_range(min_mz=100, max_mz=1400)
                            .filter_intensity(min_intensity=0.02, max_num_peaks=200)
                            .scale_intensity('root'))
    #print(len(annotated_spectrum.mz))
    _from, _to, _edge_aa, peaks, intensities, node_features, edge_features = \
        generate_graph(annotated_spectrum.mz,
                       annotated_spectrum.intensity,
                       p_m, p_z,
                       fragment_tol_mass=fragment_tol_mass,
                       verbose=False, max_miss_aa=max_miss_aa)
    # compute max misses
    if compute_completness:
        completness = check_spectrum_completness(peaks, psmPeptide)
        b_max_miss, y_max_miss, all_max_miss = completness[0]
        mass_ions = completness[1]

    if verbose:
        print("Max missing prefix masses = {b_max_miss:2d} \nMax missing suffix masses = {y_max_miss:2d}".\
              format(b_max_miss=b_max_miss, y_max_miss=y_max_miss))


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
                                    fragment_tol_mass=fragment_tol_mass,
                                    fragment_tol_mode=fragment_tol_mode,
                                    ion_types='by'))
        (sup.spectrum(annotated_spectrum).properties(width=640, height=400)
            .save(path.join('..', 'results', 'tmp', str(spectrum['id']) + '_spectrum_iplot'+
                            str(len(psmPeptide)) + '.html')))
    if compute_right_path:
        orientation = 0
        _series_mz_offset = mass.Composition({'H+': 1}).mass() if orientation == 1\
            else mass.Composition({'O': 1, 'H': 2, 'H+': 1}).mass()/1
        _loc = bisect_left(peaks,_series_mz_offset)
        paths = right_path_all_paths(psmPeptide, _from, _to, _edge_aa, orientation, 1, peaks, _loc,
                                     start_time=time.time(), first_recursion=True)
        # right_path(psmPeptide, _from, _to, _edge_aa, peaks)
        # brute_force(b_from, b_to, b_edge_aa)
        # print(p_m)

    data = {'idx': spectrum['index'] + 1,
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
            'edge_features' : edge_features,
            'max_miss_aa' : max_miss_aa
            }
    if compute_completness:
            data['b_max_miss'] = b_max_miss
            data['y_max_miss'] = y_max_miss
            data['all_max_miss'] = all_max_miss
            data['mass_ions'] = mass_ions
    return data


def get_source_sink_ions(mzarray, precursor_mass, orientation=1):
    """

    """

    _source = mass.Composition({'H+': 1}).mass() if orientation == 1\
            else mass.Composition({'O': 1, 'H': 2, 'H+': 1}).mass()/1
    _source = bisect_left(mzarray, _source)


    _sink = precursor_mass + mass.Composition({'H+': 1}).mass() if orientation == 1\
            else  precursor_mass + mass.Composition({'O': 1, 'H': 2, 'H+': 1}).mass()/1
    _sink = bisect_left(mzarray, _sink)
    return _source, _sink


def check_spectrum_completness(mzarray, psm, max_charge=3, tolerance=fragment_tol_mass/1.E6):
    # Give any possible ion for a given mass charge
    prefixes = [list(psm[:i]) for i in range(1, len(psm) + 1)]
    suffixes = [list(psm[len(psm) - i:]) for i in range(1,len(psm) + 1)]
    n_fragments = len(prefixes)
    assert len(prefixes) == len(suffixes)

    prefix_masses = list(map(lambda x : mass.calculate_mass(parsed_sequence=x), prefixes))
    suffix_masses = list(map(lambda x : mass.calculate_mass(parsed_sequence=x), suffixes))

    b_ions = [0 for i in range(n_fragments)]
    y_ions = [0 for i in range(n_fragments)]



    for i in range(n_fragments):
        for charge in range(1, max_charge + 1):
            b_putative = [f(prefix_masses[i], charge) for f in B_OFFSET_FUNCTIONS]
            y_putative = [f(suffix_masses[i], charge) for f in Y_OFFSET_FUNCTIONS]

            _res_b = list(map(lambda ion: find_corresponding_ion(mzarray,
                                                         [1 for i in range(len(mzarray))],
                                                          ion, tolerance * ion * charge),
                              b_putative))

            _res_y = list(map(lambda ion: find_corresponding_ion(mzarray,
                                                         [1 for i in range(len(mzarray))],
                                                          ion, tolerance * ion * charge),
                              y_putative))

            b_ions[i] += _res_b.count(1)
            y_ions[i] += _res_y.count(1)

    all_ions = [b + y for (b, y) in zip(b_ions, y_ions)]

    max_b = max([len(list(x[1])) for x in groupby(b_ions) if x[0] == 0] + [0])
    max_y = max([len(list(x[1])) for x in groupby(y_ions) if x[0] == 0] + [0])
    max_all = max([len(list(x[1])) for x in groupby(all_ions) if x[0] == 0] + [0])

    return (max_b, max_y, max_all), (b_ions, y_ions, all_ions)




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
    data = processSpectra(mzFile, scan2charge, scan2peptide, max_spectra=3,
                          compute_right_path=True, filtering_max_miss=(5,2))

    b_max_miss = 0
    y_max_miss = 0
    all_max_miss = 0
    for spectrum_data in data:
        b_miss, y_miss, all_miss = spectrum_data.missing_values
        b_max_miss += b_miss
        y_max_miss += y_max_miss
        all_max_miss += all_max_miss
    print("Average b maximum subsequent missing ions : {_miss}".format(_miss=b_max_miss / len(data)))
    print("Average y maximum subsequent missing ions : {_miss}".format(_miss=y_max_miss / len(data)))
    print("Average all maximum subsequent missing ions : {_miss}".format(_miss=all_max_miss / len(data)))
    #processSpectra(mzFile, scan2charge, scan2peptide)
    quit()
