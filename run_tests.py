# This script really works

# Initialize CUDA Context before creating any OpenMP threads
import ctypes
cuda = ctypes.CDLL('libcudart.so')
cuda.cudaFree.restype = ctypes.c_int
cuda.cudaFree.argtypes = [ctypes.c_void_p]
cuda.cudaFree(None)

# OpenMP Dynamic Mode causes compute-sanitizer to go haywire so turn it off
import os
os.environ['OMP_DYNAMIC'] = 'FALSE'

omp_threads = os.sysconf("SC_NPROCESSORS_ONLN")

# libgemm.os2 runs slower on multiple threads than on single one (due to cudaDeviceSynchronize than cudaStreamSynchronize) 
# also causes compute-sanitizer to panic; we will set it to 1 when LIBGEMM_OP_MODE >= 200
os.environ['OMP_NUM_THREADS'] = str(omp_threads)

# Prepares optimal OpenMP configuration
def prepare_mode(mode):
    if (mode == 0):
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
    else:
        os.environ['OMP_NUM_THREADS'] = "1"

# use LIBGEMM_LIMITED_OP to specifically target einsum and _dgemm only and leave the rest (BLAS/LAPACK internal calls)
# use LIBGEMM_OP_MODE to intercept every DGEMM call
# you have to use a modified version of pyscf/lib/numpy_helper.py for LIBGEMM_LIMITED_OP to work properly
interception_target = "LIBGEMM_LIMITED_OP"

# libgemm only cares about LIBGEMM_OP_MODE in the low level
# as long as this environment variable is set correctly
# interception will work.
os.environ["LIBGEMM_OP_MODE"] = '0' # default FORTRAN backend

# libgemm test suites
libgemm_test_modes_debug = [15, 103, 104, 105, 106, 107, 108]
libgemm_test_modes_small = [15, 103, 107, 111, 116, 202, 207, 212, 217, 220, 302, 307, 312, 317, 320]
libgemm_test_modes_med   = [15, 103, 105, 107, 109, 111, 113, 116, 
                    202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 
                    302, 304, 306, 308, 310, 312, 314, 316, 318, 320]
libgemm_test_modes_full  = [15, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                    202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
                    302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320]

selected_suite = libgemm_test_modes_debug 

# Output variables
# output CSV : {experiment_name}_{test_dir}_{method}_output.csv
experiment_name = "mp2ener_only_error" 
test_dir = "s66" 
method="dfmp2" # or mp2
basis_set="aug-cc-pvDZ"
output_file = experiment_name + "_" + test_dir + "_" + method + "_output.csv"

# PySCF defaults
import pyscf
from pyscf.mp.dfmp2_native import DFMP2

hartree2kcalmol = 627.509

# Utility Funcs
def get_structures():
    list = os.listdir(test_dir)
    list.sort()
    return list

import time
import datetime
def get_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S %Y-%m-%d')

def write_to_output(str):
    with open(output_file, 'a') as file:
        file.write(get_timestamp()+","+str+"\n")

# Since all of the molecules are dimers
# get all structures ready to perform CounterPoise BSSE
def process_structure (mol_file_xyz):
    mol_list = []
    mol_list.append([]) # parent mol ; full system
    mol_list.append([]) # first part ; 1st subsystem

    # category indexes
    parent_system_index = 0
    subsystem_index = 1
    
    line_index = 0
    with open(mol_file_xyz) as file:
        for line in file:
            line = line.strip()

            if line_index != 0 and len(line) != 0:
                parts = line.split(' ')
                processed_parts = []
                for p in parts:
                    if len(p) != 0:
                        processed_parts.append(p)
                
                # categorize
                if len(processed_parts) == 4:
                    mol_list[parent_system_index].append(processed_parts)

                if len(processed_parts) == 1:
                    if processed_parts[0] == "---":
                        subsystem_index += 1
                        mol_list.append([])
                else:
                    mol_list[subsystem_index].append(processed_parts)
                    
            line_index += 1

    # process all the subsystems
    processed_xyz = []
    processed_xyz.append([]) # supersystem is the first

    for i in range(1, len(mol_list)):
        processed_xyz .append([]) # add all subsystem placeholders

    # process supersystem
    for atom_xyz in mol_list[parent_system_index]:
        processed_xyz[parent_system_index].append((atom_xyz[0], (float(atom_xyz[1]), float(atom_xyz[2]), float(atom_xyz[3]))))

    # process subsystems
    for i in range(1, len(mol_list)):
        for j in range(1, len(mol_list)):
            prefix = ""
            if i != j:
                prefix = "X-"
            for atom_xyz in mol_list[j]:
                processed_xyz[i].append((prefix + atom_xyz[0], (float(atom_xyz[1]), float(atom_xyz[2]), float(atom_xyz[3]))))
    
    return processed_xyz

# --------------------------------------------------------------------
# - modify below code to control which parts are run with which mode -
# --------------------------------------------------------------------
def get_test_mode(iter_mode):
    hf_mode = 0
    int3c_mode = 0
    ener_calc_mode = iter_mode
    return [hf_mode, int3c_mode, ener_calc_mode]

def get_reference_mode():
    return [0, 0, 0]

def fmt_mode(mode):
    return str(mode[0]) + "," + str(mode[1]) + "," + str(mode[2])

# Run RI-MP2/DF-MP2 calculation
def run_dfmp2(molecule, hf_mode, int3c_mode, ener_calc_mode):
    # Hartree Fock
    os.environ[interception_target] = str(hf_mode)
    mf = molecule.RHF().run()

    # Create DFMP2 object
    dfmp2 = DFMP2(mf)

    # Calculate the Integrals and perform Cholesky decomposition
    os.environ[interception_target] = str(int3c_mode)
    dfmp2.calculate_integrals_()

    # Calculate the energies by matrix contractions
    os.environ[interception_target] = str(ener_calc_mode)
    dfmp2.calculate_energy()

    os.environ[interception_target] = "0"

    # Return total energy of system
    return dfmp2.e_tot

# Calculate Interaction Energy for a given system
def calculate_interaction_energy(system, mode):

    sep_str_long  = "=" * (6 + len(system) + 6 + 2)
    sep_str_short = "=" * 6

    print(sep_str_long)
    print(sep_str_short, system, sep_str_short)
    print(sep_str_long)
    print()

    # Get full dimer structure and monomer structures
    structures = process_structure(test_dir+"/"+system)

    # Construct PySCF Molecules
    mols = []
    for s in structures:
        mol = pyscf.M(atom=s, basis=basis_set).build()
        mols.append(mol)

    # Calculate energies for all structures present
    ie = 0
    ener = []
    for i in range(0, len(mols)):
        mol = mols[i]
        if i == 0:
            print ("0) supersystem calculation")
            ener.append(run_dfmp2(mol, mode[0], mode[1], mode[2]))
        else:
            print (str(i) + ") subsystem", i, "calculation")
            ener.append(run_dfmp2(mol, mode[0], mode[1], mode[2]))
        print()
    print()
    
    # Calculate IE with CP BSSE
    ie = ener[0]
    for i in range(1, len(mols)):
        ie -= ener[i]

    iekcal = ie * hartree2kcalmol

    print("Interaction Energy (Hartree):", ie)
    print("Interaction Energy (kcal/mol):", iekcal)

    return iekcal

# begin...

print("Experiment:", experiment_name)
print("Dataset:", test_dir)

dataset = get_structures()

for i in dataset:
    # Reference Calculation
    ref_mode = get_reference_mode()
    ref_ener = calculate_interaction_energy(i, ref_mode)

    write_to_output(fmt_mode(ref_mode)+","+i+","+str(ref_ener))

    for smode in selected_suite:
        # Test Calculations
        test_mode = get_test_mode(smode)
        test_ener = calculate_interaction_energy(i, test_mode)
        test_err = abs(test_ener-ref_ener)

        print("Testmode",fmt_mode(test_mode)," Error:", test_err)

        write_to_output(fmt_mode(test_mode)+","+i+","+str(test_err)) 

# end.
