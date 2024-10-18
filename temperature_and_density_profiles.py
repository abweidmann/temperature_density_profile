import itertools

import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import click

# MATPLOTLIB CONFIGURATIONS

matplotlib.rcParams['font.family'] = "Arial"  # change the default font
matplotlib.rcParams['xtick.direction'] = 'in'  # change the ticks direction
matplotlib.rcParams['ytick.direction'] = 'in'
font = FontProperties()
font.set_family('sans-serif')  # 'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'
font.set_name('Arial')
font.set_weight('bold')
# 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
# 'demi', 'bold', 'heavy', 'extra bold', 'black'
font.set_style('normal')  # 'normal', 'italic' or 'oblique'
font.set_size('x-large')  # xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

BOLTZMANN = 1.38064852e-23    # Constante de Boltzmann
AVOGADRO = 6.02214129 * 1e23  # Avogadro's number


@click.command()
@click.option('--gro', default='in.gro', type=str, help='input trajectory file in gro format')
@click.option('--density_slabs', default=6, type=int, help='input number of slabs')
@click.option('--temperature_slabs', default=6, type=int, help='input number of slabs')
@click.option('--step', default=100_000, type=int, help='input step in ps to calculate profiles')
@click.option('--atoms', default='atoms.csv', type=str, help='inputCSV file with name \
              of atoms present in the system along with molecular weight \
              and number of constraints for unique atom for each molecule, separated by commas')
def main(gro, density_slabs, temperature_slabs, atoms, step):
    """
    This code takes a molecular dynamics simulation GROMACS .gro trajectory file and calculates the 
    temperature and density profiles for it at several frames, selected by the step parameter.

    """

    df_atoms = pd.read_csv(atoms, header=None,
                           names=['atoms', 'mol_weight', 'constraints'], delimiter=",")
    previousline = ''
    with open(gro, 'r') as f:
        gro_lines = f.readlines()

    outfile = open("grovalues.txt", "w")
    times = open("tvalues.txt", "w")
    sizes = open("sizevalues.txt", "w")

    for line in gro_lines:
        if line.find('frame t=') == -1:
            outfile.write(line)
        if line.find('frame t=') != -1:
            sizes.write(previousline)
            times.write(line)
        previousline = line

    outfile.close()
    times.close()
    sizes.close()
    sizes = open("sizevalues.txt", "a")
    with open(gro, 'r') as gro_lines:
        for line in gro_lines:
            pass
        sizes.write(line)
    sizes.close()

    # BOX SIZES
    column_names = ['xsize', 'ysize', 'zsize']
    dfsize = pd.read_csv('sizevalues.txt', delim_whitespace=True, names=column_names)

    # TIMES
    column_names = ['tname1', 'tname2', 'Time']
    dftime = pd.read_csv('tvalues.txt', delim_whitespace=True, names=column_names)
    tmax = dftime.iloc[-1, dftime.columns.get_loc('Time')]
    tstep = dftime.iloc[-2, dftime.columns.get_loc('Time')]

    original_step = int(tmax-tstep)
    step_change = int(step / original_step)

    column_names = ['Name', 'Index', 'n', 'x', 'y', 'z', 'vx', 'vy', 'vz']

    df = pd.read_csv('grovalues.txt', delim_whitespace=True, names=column_names)

    natoms = int(df.iloc[0]['Name'])
    df = df.dropna(how='any', axis=0)
    df.reset_index(drop=True, inplace=True)

    nparticles = np.arange(1, natoms + 1, 1)
    timesteps = np.arange(0, tmax + 1, original_step)

    timecolumn = []
    xsizecolumn = []
    ysizecolumn = []
    zsizecolumn = []

    # dfsizestep = pd.DataFrame()
    for t in timesteps:
        for _ in nparticles:
            timecolumn.append(t)
    df["Time"] = timecolumn

    for index, (xsize, ysize, zsize) in enumerate(zip(dfsize['xsize'],
                                                      dfsize['ysize'], dfsize['zsize']),
                                                  start=original_step):
        for n in nparticles:
            xsizecolumn.append(xsize)
            ysizecolumn.append(ysize)
            zsizecolumn.append(zsize)

    df['X Size'] = xsizecolumn
    df['Y Size'] = ysizecolumn
    df['Z Size'] = zsizecolumn

    df = df.loc[df['Time'] % step_change == 0]
    timesteps = np.arange(0, tmax + 1, step)

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['vy'] = pd.to_numeric(df['vy'], errors='coerce')
    df['n'] = pd.to_numeric(df['n'], errors='coerce')

    # Convert velocities from nm/ps to m/s
    df["vx"] = 1000 * df['vx']
    df["vy"] = 1000 * df['vy']
    df["vz"] = 1000 * df['vz']

    df["vx2"] = df['vx'] * df['vx']
    df["vy2"] = df['vy'] * df['vy']
    df["vz2"] = df['vz'] * df['vz']

    time = np.arange(0, tmax + 1, step)

    # Loop to calculate the temperature

    for t, xsize in zip(time, df['X Size']):
        temp = []
        temp_x_direction = []
        dft = df.loc[df['Time'] == t]
        temp_x_slab_size = xsize / temperature_slabs
        temp_x_limits = np.arange(0, xsize, temp_x_slab_size)

        plt.figure(figsize=(8, 4), constrained_layout=True)
        plt.axis(xmin=0, xmax=xsize)
        plt.xlabel('X Direction [nm]', fontproperties=font)
        plt.ylabel('Temperature[K]', fontproperties=font)

        for xl in temp_x_limits:
            df_temp_x = dft.loc[(dft['x'] >= xl) & (dft['x'] <= (xl + temp_x_slab_size))]

            xx = (xl + xl + temp_x_slab_size) / 2
            temp_x_direction.append(xx)

            n_atoms = [df_temp_x[df_temp_x['Index'].str.contains(atom_name)].shape[0]
                       for atom_name in df_atoms['atoms']]
            total_n_atoms = sum(n_atoms)

            sum_sqr_vel = [df_temp_x[df_temp_x['Index'].str.contains(atom_name)]['vx2'].sum() +
                           df_temp_x[df_temp_x['Index'].str.contains(atom_name)]['vy2'].sum() +
                           df_temp_x[df_temp_x['Index'].str.contains(atom_name)]['vz2'].sum()
                           for atom_name in df_atoms['atoms']]

            kin = 0.5 * 1e-3 * (sum(sum_sqr_vel * df_atoms['mol_weight']))
            ndf = 3 * total_n_atoms - sum(n_atoms * df_atoms['constraints']) - 3
            temperature = 1e3 * (2 * kin) / (ndf * BOLTZMANN * AVOGADRO)
            temp.append(temperature)

        plt.scatter(temp_x_direction, temp, edgecolor='black')
        plt.plot(temp_x_direction, temp, label=f'{t / 1000}ns')
        plt.legend()
        for xt in temp_x_limits:
            plt.axvline(x=xt, color='black', ls='--', lw=2)
        plt.savefig(f'Temperature_Profile_{t/1000}ns.png', bbox_inches='tight')

    # Loop to calculate the density

    for t, xsize, ysize, zsize in zip(time, df['X Size'], df['Y Size'], df['Z Size']):
        rosystem = []
        density_x_direction = []

        dft = df.loc[df['Time'] == t]

        density_x_slab_size = xsize / density_slabs
        density_x_limits = np.arange(0, xsize, density_x_slab_size)
        plt.figure(figsize=(8, 4), constrained_layout=True)

        plt.axis(xmin=0, xmax=xsize)
        plt.xlabel('X Direction [nm]', fontproperties=font)
        plt.ylabel('Density [natoms/nm³]', fontproperties=font)

        for xl in density_x_limits:
            df_density_x = dft.loc[(dft['x'] >= xl) & (dft['x'] <= (xl + density_x_slab_size))]

            xx = (xl + xl + density_x_slab_size) / 2
            density_x_direction.append(xx)

            n_atoms = [df_density_x[df_density_x['Index'].str.contains(atom_name)].shape[0]
                       for atom_name in df_atoms['atoms']]
            total_n_atoms = sum(n_atoms)
            volume = xsize*ysize*zsize
            ro = total_n_atoms/volume
            rosystem.append(ro)

        plt.scatter(density_x_direction, rosystem, edgecolor='black')
        plt.plot(density_x_direction, rosystem, label=f'{t / 1000}ns')
        plt.legend()
        for xt in density_x_limits:
            plt.axvline(x=xt, color='black', ls='--', lw=2)
        plt.savefig(f'Density_Profile_{t/1000}ns.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
