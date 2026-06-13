import os
import argparse
import pandas as pd
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from functools import partial
from pyxtal import pyxtal
from lego.builder import builder
from pyxtal.db import database_topology
from ase.db import connect
#from juliacall import Main as jl

# Sanity check: Julia & PythonCall are live
#print(jl.seval('VERSION'))
#print(jl.seval('using PythonCall; "PythonCall loaded"'))
#jl.seval('import Pkg; Pkg.add("CrystalNets"); using CrystalNets')
#print("Success")
def process_rep(rep, discrete, discrete_cell, discrete_res):
    xtal = pyxtal()
    # Remove the extra energy and labels
    mod = (len(rep) - 7) % 4
    if mod > 0: rep = rep[:-mod]
    try:
        xtal.from_tabular_representation(rep,
                                         normalize=False,
                                         discrete=discrete,
                                         discrete_cell=discrete_cell,
                                         N_grids=discrete_res)
        if xtal.valid and len(xtal.atom_sites) > 0:
           return rep, sum(xtal.numIons)
    except:
        print(f"Failed to process: {rep}")
    return None

def rep_to_xtal(rep, discrete, discrete_cell, discrete_res):
    xtal = pyxtal()

    mod = (len(rep) - 7) % 4
    if mod > 0:
        rep = rep[:-mod]

    xtal.from_tabular_representation(
        rep,
        normalize=False,
        discrete=discrete,
        discrete_cell=discrete_cell,
        N_grids=discrete_res
    )

    if xtal.valid and len(xtal.atom_sites) > 0:
        return xtal

    return None

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Relaxation code.")
    parser.add_argument('--ncpu', type=int, default=1,
                        help='N_cpu for parallel computation')
    parser.add_argument('--csv', help='csv file path')
    parser.add_argument('--begin', type=int, default=0, help='end count')
    parser.add_argument('--end', type=int, default=-1, help='end count')
    parser.add_argument('--source', default='data/source/sp2_sacada.db', 
                        help='path to source database')
    parser.add_argument('--prototype', default='graphite',
                        help='prototype for reference environment')
    parser.add_argument('--CN', type=int, default=3,
                        help='coordination number for reference environment')
    parser.add_argument('--skip-topology', action='store_true',
                        help='Skip topology update/cleaning.')
    parser.add_argument('--skip-energy', action='store_true',
                        help='Skip GULP/ReaxFF energy calculation.')
    parser.add_argument('--save-pre-so3-db', action='store_true',
                        help='Save decoded structures before SO3')

    # Check if use_mpi is invoked
    rank, size = 0, 1
    print(f"Current rank-{rank}, size-{size}")

    # Parse arguments
    t0 = time()
    args = parser.parse_args()
    f = args.csv
    ncpu = args.ncpu
    begin, end = args.begin, args.end
    
    # Debug CPU allocation
    print(f"Command line --ncpu: {args.ncpu}")
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}")
    print(f"Using ncpu = {ncpu}")

    xtal = pyxtal()
    xtal.from_prototype(args.prototype)
    cif_file = xtal.to_pymatgen()

    name = f.split('/')[-1].split('.')[0]
    os.makedirs(name, exist_ok=True)

    # 1. Load and get valid structures sorted by number of atoms
    df = pd.read_csv(f)
    print(f"Total rows in CSV: {len(df)}")
    print(f"Begin: {begin}, End: {end}")
    
    val = df['x0'].max()
    if val < 5 + 1e-3:
        discrete, discrete_res = False, None
    elif val < 50 + 1e-3:
        discrete, discrete_res = True, 50
    else:
        discrete, discrete_res = True, 100

    if abs(df['a'][0] - round(df['a'][0])) < 1e-6 and abs(df['c'][0] - round(df['c'][0])) < 1e-6 :
        discrete_cell = True
    else:
        discrete_cell = False

    full_data = df.to_numpy()
    if end == -1:
        full_data = full_data[begin:]
    else:
        full_data = full_data[begin:end]
    
    print(f"Data chunk shape after slicing: {full_data.shape}")

    # Split the data equally among the ranks
    chunk_size = len(full_data) // size
    chunks = [full_data[i*chunk_size:(i+1)*chunk_size]
            for i in range(size)]
    data = chunks[0]
    N0 = len(data)
    print(f"Rank-{rank} receives {N0} structures from {f}")
    print(f"Processing with discrete={discrete}, discrete_res={discrete_res}, discrete_cell={discrete_cell}")

    # Use multiprocessing to speed up the processing of each rep
    partial_process_rep = partial(process_rep,
                                  discrete=discrete,
                                  discrete_cell=discrete_cell,
                                  discrete_res=discrete_res)
    lists = []
    with Pool(ncpu) as pool:
        for result in tqdm(pool.imap(partial_process_rep, data),
                           total=len(data),
                           desc=f"Processing Rank-{rank}"):
            if result is not None:
                lists.append(result)

    # list of sorted (xtal, numIons)
    sorted_lists = sorted(lists, key=lambda x: x[-1])
    reps = [l[0] for l in sorted_lists]
    N1 = len(reps)
    print(f"Rank-{rank} receives {N1} structures for optimization ")

    if args.save_pre_so3_db:
        pre_db_path = f"{name}/pre_so3.db"

    if os.path.exists(pre_db_path):
        os.remove(pre_db_path)

    pre_db = connect(pre_db_path)

    n_pre_saved = 0
    for i, rep in enumerate(tqdm(reps, desc="Saving pre-SO3 db")):
        try:
            xtal_pre = rep_to_xtal(
                rep,
                discrete=discrete,
                discrete_cell=discrete_cell,
                discrete_res=discrete_res
            )

            if xtal_pre is None:
                continue

            atoms_pre = xtal_pre.to_ase(resort=False)

            pre_db.write(
                atoms_pre,
                stage="pre_so3",
                source_csv=args.csv,
                row_index=i,
                num_atoms=len(atoms_pre),
                data={
                    "rep": rep.tolist() if hasattr(rep, "tolist") else list(rep)
                }
            )

            n_pre_saved += 1

        except Exception as e:
            print(f"Failed to save pre-SO3 structure {i}: {e}")

    print(f"Saved {n_pre_saved} pre-SO3 structures to {pre_db_path}")

    # 2. Setup builder and run optimization
    bu = builder(['C'], [1], rank=rank, prefix=f'{name}/mof')
    bu.set_descriptor_calculator(mykwargs={'rcut': 2.1})
    bu.set_reference_enviroments(cif_file)
    bu.set_criteria(CN={'C': [args.CN]})
    
    print(f"About to optimize {len(reps)} structures with ncpu={ncpu}")
    time_init = time()
    xtals = bu.optimize_reps(reps, ncpu=ncpu,
                             minimizers=[('Nelder-Mead', 100),
                                         ('L-BFGS-B', 400),
                                         ('L-BFGS-B', 200)],
                             N_grids=discrete_res)
    t_opt = round((time()-time_init), 2)
    print(f"Rank-{rank} optimization time: {t_opt} seconds")
    N2 = len(xtals)
    print(f"Rank-{rank} gets {N2} valid optimized structures")

    # 3. Optional topology analysis / cleaning
    top_time = 0.0
    N3 = N2

    if not args.skip_topology:
        t_top_start = time()
        bu.db.update_row_topology(
            overwrite=False,
            prefix=f'{name}/mof-0',
            timeout=600
        )
        top_time = round((time() - t_top_start), 2)
        print(f"Rank-{rank} topology time: {top_time} seconds")

        bu.db.clean_structures_spg_topology(dim=3)
    else:
        print("Skipping topology update and topology cleaning.")

    # 4. Optional energy calculation
    t_energy = 0.0

    if not args.skip_energy:
        t_energy_start = time()
        bu.db.update_row_energy(
            'GULP',
            ncpu=ncpu,
            calc_folder=f"{name}/gulp_{rank}"
        )
        t_energy = round((time() - t_energy_start), 2)
        print(f"Rank-{rank} energy calculation time: {t_energy} seconds")
    else:
        print("Skipping GULP/ReaxFF energy calculation.")

    # 5. Produce final.db
    #
    # If topology is enabled, preserve the original behavior:
    # unique topology extraction ranked by ff_energy if available.
    #
    # If topology is skipped, just use the raw optimized database.
    final_db = f"{name}/final.db"

    if not args.skip_topology:
        if args.skip_energy:
            # No ff_energy key available, so avoid sorting by ff_energy.
            N3 = bu.db.get_db_unique_topology(
                f'{name}/unique_{rank}.db',
                update_topology=False
            )
        else:
            N3 = bu.db.get_db_unique_topology(
                f'{name}/unique_{rank}.db',
                update_topology=False,
                key='ff_energy'
            )

        os.system(f'mv {name}/unique_{rank}.db {final_db}')

    else:
        # Raw optimized database is usually {prefix}.db, where prefix was f'{name}/mof'
        raw_db = f'{name}/mof.db'

        if os.path.exists(raw_db):
            os.system(f'cp {raw_db} {final_db}')
            print(f"Copied raw optimized database to {final_db}")
        else:
            print(f"WARNING: Could not find expected raw database: {raw_db}")
            print("Trying to continue, but final.db may be missing.")

    # Total time after optional topology/energy/final-db construction
    t = round((time() - t0), 2)

    print(f'R-{rank} N0/N1/N2/N3: {N0}/{N1}/{N2}/{N3} in {t} sec/{ncpu} cores')
    local_data = (N0, N1, N2, N3)

    # 6. Optional overlap check against source database
    #
    # This may still be useful, but if final.db was not produced, avoid crashing.
    N4 = -1
    if os.path.exists(final_db):
        try:
            db = database_topology(args.source, log_file=f'{name}/sp2.log')
            overlaps = db.check_overlap(final_db)
            N4 = len(overlaps)
        except Exception as e:
            print(f"WARNING: overlap check failed: {e}")
            N4 = -1
    else:
        print(f"WARNING: {final_db} does not exist; skipping overlap check.")

    # 7. Write metrics
    t_opt_min = round(t_opt / 60, 2)
    top_time_min = round(top_time / 60, 2)
    t_min = round(t / 60, 2)
    t_energy_min = round(t_energy / 60, 2)

    with open(f'{name}/metric.txt', 'w') as f:
        f.write(f'Source data:     {args.csv}\n')
        f.write(f'Skip topology:   {args.skip_topology}\n')
        f.write(f'Skip energy:     {args.skip_energy}\n')
        f.write(f'Optimization time minutes:       {t_opt_min:12.2f}\n')
        f.write(f'Topology time minutes:           {top_time_min:12.2f}\n')
        f.write(f'Energy calculation time minutes: {t_energy_min:12.2f}\n')
        f.write(f'Total time minutes:              {t_min:12.2f}\n')
        f.write(f'N_parallel_cpus: {ncpu:12d}\n')
        f.write(f'N_total_count:   {N0:12d}\n')
        f.write(f'N_valid_xtal:    {N1:12d}\n')
        f.write(f'N_valid_env:     {N2:12d}\n')
        f.write(f'N_unique_xtal:   {N3:12d}\n')
        f.write(f'N_train_overlap: {N4:12d}\n')
