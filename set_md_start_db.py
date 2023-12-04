import argparse
import pickle

from ase.io import read
import ase.db as db

from gpaw import FermiDirac, PoissonSolver, Mixer
from gpaw.utilities import h2gpts


def main(structure: str, mode: str, xc: str, temperature: float, brendsen_tau: float, time_step: float, kpts: tuple[int, int, int] = (1, 1, 1),):
    atoms = read(structure)

    name = structure.split('/')[-1].split('.')[0]

    calc_par_dict = dict(
        mode=mode,
        basis='dzp',
        #setups={'Pt': '10'},
        xc=xc,
        kpts=kpts,
        occupations=FermiDirac(0.1),
        poissonsolver={'dipolelayer': 'xy'},
        mixer=Mixer(beta=0.025, nmaxold=5, weight=50.0),
        gpts=h2gpts(0.18, atoms.cell, idiv=8),
        #    convergence={'energy': 2.0e-7, 'density': 1e-5}, # HIGH
        convergence={'energy': 5.0e-6, 'density': 8e-5},  # LOW
        #    parallel = dict(sl_auto = True), #Using Scalapack = 4x speedup!
        txt=f'{name}_{xc}_{mode}_k{"-".join(map(str, kpts))}.txt'
    )

    atoms.set_calculator(calc_par_dict)

    calc_pickle = pickle.dumps(calc_par_dict)

    with db.connect(name + '.db') as db_obj:
        db_obj.write(atoms=atoms, kinitic_E=0, temperature=temperature, brendsen_tau=brendsen_tau, time=0, time_step_size=time_step, data=dict(calc_pickle=calc_pickle))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_struc', type=str)
    parser.add_argument('temperature', type=float)
    parser.add_argument('brendsen_tau', type=float)
    parser.add_argument('time_step_size', type=float)
    parser.add_argument('XC', type=str)
    parser.add_argument('-k', '--kpts', nargs=3, default=(1, 1, 1), type=int)
    parser.add_argument('-m', '--mode', choices=('FD', 'LCAO', 'PW'), default='LCAO', type=str)
    args = parser.parse_args()

    main(
        structure=args.traj_struc,
        mode=args.mode,
        xc=args.XC,
        temperature=args.temperature,
        brendsen_tau=args.brendsen_tau,
        time_step=args.time_step_size,
        kpts=args.kpts
    )
