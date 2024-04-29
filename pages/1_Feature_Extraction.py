import os
import sys
import time
import tempfile
import streamlit as st
import numpy as np  
import pandas as pd
import mdtraj as md
from bokeh.plotting import figure
from itertools import combinations
from sklearn.preprocessing import StandardScaler

def compute_rmsd(trajectory):
    rmsds = md.rmsd(trajectory, trajectory, 1)
    st.success("Successfully computed rmsd")
    frames = np.arange(len(rmsds))  # Frame indices
    p = figure(
    title='Frame vs RMSD',
    x_axis_label='Frames',
    y_axis_label='RMSD')
    p.line(frames,rmsds, legend_label='RMSD ', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rmsds

def compute_rmsf(trajectory,n_frames):
    rmsf = []
    for i in range(n_frames):
        rms = md.rmsf(trajectory, trajectory, i)
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        fin_rms = mean_rms + std_rms
        rmsf.append(fin_rms)
    frame_numbers = np.arange(1, len(rmsf) + 1)
    p = figure(
    title='Frame vs RMSF',
    x_axis_label='Frames',
    y_axis_label='RMSF')
    p.line(frame_numbers,rmsf, legend_label='RMSF', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rmsf

def compute_sasa(trajectory):
    sasa = md.shrake_rupley(trajectory)
    st.success("Successfully computed SaSa")
    total_sasa = sasa.sum(axis=1)
    p = figure(
    title='Time vs SaSa',
    x_axis_label='Time [ps]',
    y_axis_label='Total SASA (nm)^2')
    p.line(trajectory.time,total_sasa, legend_label='SaSa', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return sasa

def compute_rog(trajectory):
    rog = md.compute_rg(trajectory, masses=None)
    st.success("Successfully computed Radius of Gyration")
    frames = np.arange(len(rog))
    p = figure(
    title='Frame vs RoG',
    x_axis_label='Frames',
    y_axis_label='RoG')
    p.line(frames,rog, legend_label='RoG', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rog

def compute_h_bonds(trajectory):
    h_bonds = md.baker_hubbard(trajectory)
    st.success("Successfully computed Hydrogen Bonds")
    st.write("Total number of hydrogen bonds formed over the simulation : %d" % len(h_bonds))
    st.text("Here below you can find the residues which contain hydrogen bonds: ")
    label = lambda hbond : '%s -- %s' % (trajectory.topology.atom(hbond[0]), trajectory.topology.atom(hbond[2]))
    for hbond in h_bonds:
        st.write(label(hbond))

def best_hummer_q(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 3])
    
    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    # print("Number of native contacts", len(native_contacts))
    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q ,native_contacts

def compute_native_contacts(trajectory):
    q, native = best_hummer_q(trajectory, trajectory[0])
    st.success("Successfully computed fraction of native contacts that determine protein folding")
    st.write("Fraction of native contacts determing native folding mechanism in the trajectory are : %d" % len(native))

def get_file_type(filename):
    _,file_extension = os.path.splitext(filename)
    return file_extension.lower()

def main():
    st.set_page_config(page_title="Tool for Analysis of MD Data",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Welcome To Our Feature Extraction web page🎈")
    st.divider()
    xtc_file = st.file_uploader("Upload your Trajectory file", type=["xtc","trr","dcd"])
    pdb_file = st.file_uploader("Upload your PDB file", type=["pdb","psf"])
    rmsd_box = st.checkbox("Compute Root Mean Square Deviation")
    rmsf_box = st.checkbox("Compute Root Mean Square Fluctuation")
    sasa_box = st.checkbox("Compute Solvent Accessible Surface Area")
    rog_box = st.checkbox("Compute Radius of Gyration")
    h_bonds = st.checkbox("Compute Hydrogen Bonds")
    native_contacts = st.checkbox("Compute Native Contacts")
    button = st.button("Process")
    
    if rmsf_box:
        st.warning("Computing RMSF takes longer time than expected", icon="⚠️")

    if sasa_box:
        st.warning("Computing SASA takes longer time than expected", icon="⚠️")

    if button:
        if xtc_file and pdb_file:
            st.success("Successfully uploaded the trajectory and the PDB file")
            xtc_ext = get_file_type(xtc_file.name)
            pdb_ext = get_file_type(pdb_file.name)
            if xtc_ext == ".xtc" and pdb_ext == ".pdb":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xtc") as tmp_xtc:
                    tmp_xtc.write(xtc_file.getvalue())
                    xtc_path = tmp_xtc.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb:
                    tmp_pdb.write(pdb_file.getvalue())
                    pdb_path = tmp_pdb.name
            if xtc_ext == ".trr" and pdb_ext == ".pdb":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".trr") as tmp_xtc:
                    tmp_xtc.write(xtc_file.getvalue())
                    xtc_path = tmp_xtc.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb:
                    tmp_pdb.write(pdb_file.getvalue())
                    pdb_path = tmp_pdb.name
            if xtc_ext == ".dcd" and pdb_ext == ".pdb":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dcd") as tmp_xtc:
                    tmp_xtc.write(xtc_file.getvalue())
                    xtc_path = tmp_xtc.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb:
                    tmp_pdb.write(pdb_file.getvalue())
                    pdb_path = tmp_pdb.name
            with st.spinner('Wait a while...'):
                trajectory = md.load(xtc_path,top=pdb_path)
                st.success("Successfully loaded the trajectory and we are ready to start analysis")
                st.write(f"Number of frames: {trajectory.n_frames}")
                st.write(f"Number of atoms: {trajectory.n_atoms}")
                st.write(f"Number of residues: {trajectory.n_residues}")
            df = pd.DataFrame({'Frame': range(1, trajectory.n_frames+1)})
            if rmsd_box:
                rmsd = compute_rmsd(trajectory)
                df['RMSD'] = rmsd
            if rmsf_box:
                rmsf = compute_rmsf(trajectory,trajectory.n_frames)
                df['RMSF'] = rmsf
            if sasa_box:
                sasa = compute_sasa(trajectory) 
                df['SASA'] = sasa
            if rog_box:
                rog = compute_rog(trajectory)
                df['ROG'] = rog  
            if h_bonds:
                compute_h_bonds(trajectory)
            if native_contacts:
                compute_native_contacts(trajectory)
        df 
        st.write(df.iloc[:,1:].describe())
        st.toast('Hooray! We have computed all your selected features', icon='🎉')

if __name__ == '__main__':
    main()