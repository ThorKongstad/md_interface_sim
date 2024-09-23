import argparse
import sys
import pathlib
from copy import deepcopy


#from operator import itemgetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import build_pd, folder_exist

import numpy as np
from ase import Atoms, Atom
from ase.io import read, write



def main(atoms_obj, layer_heights, layer_atom_numbers, vacuum, out):
    if type(atoms_obj) == str:
        if atoms_obj.split('.')[-1] == 'db': work_atoms = build_pd(atoms_obj).iloc[-1]['atoms']
        elif atoms_obj.split('.')[-1] == 'traj': work_atoms = read(atoms_obj, -1)
        else: raise 'could not find atoms_obj type'
    elif type(atoms_obj) == Atoms: work_atoms = atoms_obj
    else: raise 'could not find atoms_obj type'

    for at_nr in layer_atom_numbers:
        new_atom: Atom = deepcopy(work_atoms[at_nr])
        new_atom.position = (a-b for a, b in zip(new_atom.position, (0, 0, layer_heights)))
        work_atoms.append(new_atom)

    work_atoms.set_cell(work_atoms.get_cell() + np.array(((0, 0, 0), (0, 0, 0), (0, 0, vacuum))))
    work_atoms.set_positions((pos + np.array((0, 0, vacuum)) for pos in work_atoms.get_positions()))

    write(out, work_atoms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('atoms_obj', type=str,  help='an atom object in either a db or traj')
    parser.add_argument('layer_heights', type=float)
    parser.add_argument('layer_atom_numbers', nargs='+', type=int)
    parser.add_argument('out')
    parser.add_argument('-vac', '--vacuum', type=float)
    args = parser.parse_args()

    main(**args.__dict__)
