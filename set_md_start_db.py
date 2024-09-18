import argparse
import os
import pickle

from ase.io import read
import ase.db as db

from gpaw import FermiDirac, PoissonSolver, Mixer
from gpaw.utilities import h2gpts


def main(structure: str, mode: str, xc: str, temperature: float, brendsen_tau: float, time_step: float, kpts: tuple[int, int, int] = (1, 1, 1), static_pwd: bool = False):
    atoms = read(structure)

    sti = os.path.dirname(os.path.realpath(structure)) if not static_pwd else './'
    name = os.path.basename(structure).split('.')[0]

    kpts_dict = dict(kpts=kpts) if mode == 'pw' else dict()

    calc_par_dict = dict(
        mode=mode,
        basis='dzp',
        #setups={'Pt': '10'},
        xc=xc,
        occupations=FermiDirac(0.1),
        poissonsolver={'dipolelayer': 'xy'},
        mixer=Mixer(beta=0.025, nmaxold=5, weight=50.0),
        gpts=h2gpts(0.18, atoms.cell, idiv=8),
        #    convergence={'energy': 2.0e-7, 'density': 1e-5}, # HIGH
        convergence={'energy': 5.0e-6, 'density': 8e-5},  # LOW
        #    parallel = dict(sl_auto = True), #Using Scalapack = 4x speedup!
        txt=f'{sti}/{name}_{xc}_{mode}'+(f'_k{"-".join(map(str, kpts))}' if mode == 'pw' else '')+'.txt',
        **kpts_dict
    )

    #atoms.set_calculator(calc_par_dict)

    calc_pickle = str(pickle.dumps(calc_par_dict))

    with db.connect(f'{sti}/{name}_{xc}_{mode}'+(f'_k{"-".join(map(str, kpts))}' if mode == 'pw' else '') + '.db') as db_obj:
        db_obj.write(atoms=atoms, kinitic_E=0, temperature=temperature, brendsen_tau=brendsen_tau, time=0, time_step_size=time_step, data=dict(calc_pickle=calc_pickle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_struc', type=str)
    parser.add_argument('temperature', type=float)
    parser.add_argument('brendsen_tau', type=float)
    parser.add_argument('time_step_size', type=float)
    parser.add_argument('XC', type=str)
    parser.add_argument('-k', '--kpts', nargs=3, default=(1, 1, 1), type=int)
    parser.add_argument('-m', '--mode', choices=('fd', 'lcao', 'pw'), default='lcao', type=str)
    parser.add_argument('--static', action='store_true', help='a bool for choosing a static workdir, files will be made wherever pwd is')
    args = parser.parse_args()

    main(
        structure=args.traj_struc,
        mode=args.mode,
        xc=args.XC,
        temperature=args.temperature,
        brendsen_tau=args.brendsen_tau,
        time_step=args.time_step_size,
        kpts=args.kpts,
        static_pwd=args.static
    )
