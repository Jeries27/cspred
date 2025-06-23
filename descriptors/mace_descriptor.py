
from descriptors.base_descriptor import BaseDescriptor
from mace.calculators import mace_off
import matplotlib.pyplot as plt 
import umap.umap_ as umap
import os
import numpy as np

class MaceDescriptor(BaseDescriptor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(desc_name="mace", *args, **kwargs)

    def get_model(self, model:str="large"):
        return mace_off(model=model, device=self.device)
    
    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        invariants_only = kwargs["invariants_only"]
        num_layers = kwargs["num_layers"]
        return model.get_descriptors(atoms_obj_env, invariants_only=invariants_only, num_layers=num_layers)
    
    def visualize_mace_force(self, model, atoms_obj_env):
        forces = model.get_forces(atoms_obj_env)
        coordinates = atoms_obj_env.positions
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        u, v, w = forces[:, 0], forces[:, 1], forces[:, 2]
        
        forces2 = forces[:len(forces)//2]
        coordinates2 = coordinates[:len(forces)//2]

        x2, y2, z2 = coordinates2[:, 0], coordinates2[:, 1], coordinates2[:, 2]
        u2, v2, w2 = forces2[:, 0], forces2[:, 1], forces2[:, 2]

        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(x, y, z, u, v, w, length=0.3, normalize=True, color='b', alpha=0.5)
        ax.quiver(x2, y2, z2, u2, v2, w2, length=0.3, normalize=True, color='r', label="Forces 2", alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"MACE Force Vector Field {forces.shape[0]} atoms")

        plt.savefig(f"descriptors/visualizations/mace_forces_atoms_bigger.png", dpi=300)
      
   
    def visualize_umap(self, dataset, dim = "2D", suffix = ""):
        data = np.array([data_i['descriptor'] for data_i in dataset])
        y = np.array([data_i['cs'] - data_i['rc_cs'] for data_i in dataset])

        mask = (y < 10) & (y > -10)

        y_masked = y[mask]

        if dim == "2D":
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif dim == "3D":
            reducer = umap.UMAP(n_components=3, random_state=42)

        print("fitting umap")
        embedding = reducer.fit_transform(data)
        embedding_masked = embedding[mask]
        np.save(f"/nfs/scistore20/bronsgrp/jsaleh/force_fields/descriptors/visualizations/mace_umap{dim}_{suffix}.npy", embedding)

        print("plotting umap maksed")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding_masked[:, 0],
            embedding_masked[:, 1],
            c=y_masked,
            cmap="viridis",  # choose any perceptual colormap (e.g. plasma, inferno)
            s=5,  # marker size
            alpha=0.9,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Chemical shift", rotation=270, labelpad=15)
        plt.title("UMAP of descriptors coloured by chemical shift")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.show()

        plt.savefig(f"/nfs/scistore20/bronsgrp/jsaleh/force_fields/descriptors/visualizations/mace_umap{dim}_{suffix}_masked.png", dpi=300)

        print("plotting umap unmasked")
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y,
            cmap="viridis",  # choose any perceptual colormap (e.g. plasma, inferno)
            s=5,  # marker size
            alpha=0.9,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Chemical shift", rotation=270, labelpad=15)
        plt.title("UMAP of descriptors coloured by chemical shift")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.show()

        plt.savefig(
            f"/nfs/scistore20/bronsgrp/jsaleh/force_fields/descriptors/visualizations/mace_umap{dim}_{suffix}.png",
            dpi=300)
