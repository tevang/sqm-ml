from scipy import optimize
from simtk.unit import *
from parmed import unit as u
from lib.global_fun import *
from lib.utils.print_functions import ColorPrint


def optimize_openmm(simulation):
    """
    This methods implements the default OpenMM's minimizer.

    :param simulation:
    :return:
    """
    # Optimization tolerance is RMS force
    simulation.minimizeEnergy(tolerance=(0.01*kilocalories/mole/angstrom), maxIterations=10000)

def list_of_moving_atoms(simulation):
    natom = simulation.system.getNumParticles()
    simulation.system.getParticleMass(0)
    atomlist = []
    for i in range(0, natom):
        if simulation.system.getParticleMass(i).value_in_unit(dalton) != 0:
            atomlist.append(i)
    return atomlist

def select_moving_part(coords, atomlist):
    coords2 = []
    for i in atomlist:
        coords2.append(coords[i])
    return coords2

def moving_to_all(coords, atomlist, all_pos):
    coords2 = np.copy(all_pos)
    c = 0
    for i in atomlist:
        coords2[i,:] = coords[c,:]
        c += 1

    return coords2

def optimize_custom(simulation, maxGrad=0.5):
    """
    VERY SLOW!!!
    This function implements the custom optimization code from https://github.com/pandegroup/openmm/issues/1321
    Instead of using OpenMM's native optimizer that runs on GPUs and stops after a predifined number of steps of energy tolerance
    (which can vary from system to system and was unstable; led to infinite energies in production MD)

    :param simulation:  OpenMM's Simulation() object.
    :param maxGrad:     convergence limit for geometry optimization (max. grad in kcal/mol/A)
    :return:
    """

    con = simulation.context

    # Moving atoms
    moving_atoms = list_of_moving_atoms(simulation)
    all_atom = len(moving_atoms) == simulation.system.getNumParticles()

    # Initial positions
    pos = con.getState(getPositions=True).getPositions().value_in_unit(angstrom)
    if not all_atom:
        all_pos = np.array(pos)
        pos = select_moving_part(pos, moving_atoms)
    pos = np.array(pos)

    def target_function(xyz):
        #con.setPositions(xyz.reshape(-1,3) * angstrom)
        coords=xyz.reshape(-1,3)
        if not all_atom:
            coords = moving_to_all(coords, moving_atoms, all_pos)
        con.setPositions(coords * angstrom) # Set position of all atoms
        state = con.getState(getEnergy=True, getForces=True)
        frc = state.getForces().value_in_unit(kilocalories/mole/angstrom)
        if not all_atom:
            frc = select_moving_part(frc, moving_atoms)
        frc = np.array(frc)
        ene = state.getPotentialEnergy().value_in_unit(kilocalories/mole)
        return ene, -frc.flatten()

    opt_results = optimize.minimize(
            target_function,    # Function
            pos,            # x0
            method='L-BFGS-B',
            jac=True,        # target_function returns gradient
            options=dict(
                maxcor=3,    # No. of LBFGS vectors
                maxiter=5000,
                disp=0,        # Print level, 0 = no printing
                gtol=maxGrad     # Max. componet of gradient
                )
            )
    print('Optimization finished')
    if opt_results.success:
        print('   Status: converged')
    else:
        print('   Status: failed')
    print('   No. of iterations:  %d' % opt_results.nit)
    print('   No. of evaluations: %d' % opt_results.nfev)
    print('   Details: ' + opt_results.message.decode("utf-8"))

    # TODO: pass the new coordinates to the State object.
    return opt_results

    # # Save optimized state
    # state = simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)


def optimize_native_openmm(simulation, etolratio=5e-5, min_steps=5000):
    """
    This method invokes the native OpenMM Minimizer (much faster that scipy.optimize). First it runs a
    short 300 step minimization to alleviate strong steric clashes and get an accurate estimate of the
    potential energy of the system. Then via the given etolratio it estimates a
    system specific (good for comparison between different systems with varying number of atoms)
    energy tolerance for a second round of minimization (more thorough).

    :param simulation:
    :param etolratio: if 0 then only min_steps of minimization will be run
    :param min_steps: if this number of steps is exceeded given the etolratio value, then minimization stops
    :return:
    """

    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy() / u.kilocalories_per_mole
    print("Potential Energy before minimization: %f kcal/mol" % initial_energy)
    if etolratio > 0:
        # Run a short minimization to alleviate steric clashes and get a more accurate estimate of the
        # potential energy in order to calculate the energy tolerance.
        print("Running preliminary minimization of 300 steps.")
        simulation.minimizeEnergy(
            tolerance=1e-10 * u.kilocalories_per_mole,
            maxIterations=300)
        postmin_state = simulation.context.getState(getEnergy=True)
        postmin_energy = postmin_state.getPotentialEnergy() / u.kilocalories_per_mole
        print("Potential Energy after preliminary minimization: %f kcal/mol" % postmin_energy)
        etol = abs(etolratio * postmin_energy)
        min_steps -= 300    # we already did 300 steps of minimization
        ColorPrint('Minimizing energy on platform %s using energy tolerance = %f kcal/mol' %
                   (simulation.context.getPlatform().getName(), etol), "BOLDGREEN")
    else:   # do exactly min_steps, ignore the etol
        etol = 1e-10
        ColorPrint('Minimizing energy on platform %s using exactly %i steps of minimization.' %
                   (simulation.context.getPlatform().getName(), min_steps), "BOLDGREEN")
    simulation.minimizeEnergy(
        tolerance=etol * u.kilocalories_per_mole,
        maxIterations=min_steps-300)

    final_state = simulation.context.getState(getEnergy=True)
    final_energy = final_state.getPotentialEnergy() / u.kilocalories_per_mole
    print("Potential Energy after minimization: %f kcal/mol" % final_energy)