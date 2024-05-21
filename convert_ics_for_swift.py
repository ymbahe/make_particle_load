"""
Convert ICs produced by IC_Gen into a single file in SWIFT format.
"""

import numpy as np
import argparse
import h5py as h5
import os


def main():
    args = parse_arguments()
    swiftICs = SwiftICs(args)
    swiftICs.convert()


class SwiftICs:
    """
    Generate and store the ICs for SWIFT.
    """

    def __init__(self, args):
        self.args = args
        input_file = args.input_file_name
        if not os.path.isfile(input_file):
            raise OSError(f"Specified input file '{input_file}' not found!")
        self.header = self.load_header(input_file)
        self.meta = self.load_meta_data()
        self.num_input_files = self.header["NumFilesPerSnapshot"]
        in_file_base = ".".join(input_file.split(".")[:-2])
        self.input_files = [
            f"{in_file_base}.{ifile}.hdf5" for ifile in range(self.num_input_files)
        ]
        self.num_parts = [
            n_type + 2**32 * hw_type
            for n_type, hw_type in zip(
                self.header["NumPart_Total"],
                self.header["NumPart_Total_HighWord"],
            )
        ]
        print(f"Number of particles from IC-Gen: {self.num_parts}")
        self.particle_buffer = {0: None, 1: None, 2: None}
        self.current_offset = {0: 0, 1: 0, 2: 0}
        out_file = self.args.output_file_name
        out_dir = self.args.repo_dir
        if out_file is None:
            out_file = self.args.input_file_name
            out_file_parts = out_file.split(".")
            out_file = ".".join(out_file_parts[:-2]) + ".hdf5"
        if out_dir is not None:
            out_file_name = out_file.split("/")[-1]
            out_file = os.path.join(out_dir, out_file_name)
        self.out_file = out_file

    def convert(self):
        self.init_outfile()
        for input_file in self.input_files:
            self.load_parts(input_file, 1)
            self.load_parts(input_file, 2)
            # self.shift_and_truncate()  # not fully implemented?
            if self.args.isolate_gas:
                self.isolate_gas()
            if self.args.remap_ids:
                self.remap_ids()
            self.save_particles(input_file)
        # finished, change filename to final output filename:
        assert all(
            [self.current_offset[k] == self.num_parts[k] for k in self.current_offset]
        )
        os.rename(f"{self.out_file}.incomplete", f"{self.out_file}")

    def load_header(self, in_file):
        """Parse the header from the specified input file into a dict."""
        header = {}
        with h5.File(in_file, "r") as f:
            g = f["Header"]
            for key in g.attrs.keys():
                if key in [
                    "BoxSize",
                    "Flag_Entropy_ICs",
                    "MassTable",
                    "NumFilesPerSnapshot",
                    "NumPart_Total",
                    "Time",
                    "HubbleParam",
                ]:
                    header[key] = g.attrs[key]

        return header

    def load_parts(self, input_file, ptype):
        """Load all data for a specific particle type from one input file."""
        props = ["Coordinates", "Masses", "ParticleIDs", "Velocities"]

        with h5.File(input_file, "r") as f:
            num_this_file = f["Header"].attrs["NumPart_ThisFile"][ptype]
            if num_this_file == 0:
                print(f"No particles of type {ptype} on file {input_file}...")
                self.particle_buffer[ptype] = None
                return
            else:
                self.particle_buffer[ptype] = dict()

            g = f[f"PartType{ptype}"]
            for prop in props:
                if prop == "Masses" and self.header["MassTable"][ptype] < 0:
                    self.particle_buffer[ptype]["Masses"] = (
                        np.zeros(self.num_this_file, dtype="float32")
                        + self.header["MassTable"][ptype]
                    )
                else:
                    self.particle_buffer[ptype][prop] = g[prop][...]  # explicit read
        return

    def load_meta_data(self):
        meta = {}
        with h5.File(self.args.pl_meta_file, "r") as f:
            h = f["Header"]
            meta["Omega0"] = h.attrs["Omega0"]
            meta["OmegaDM"] = h.attrs["OmegaDM"]
            meta["OmegaLambda"] = h.attrs["OmegaLambda"]
            meta["OmegaBaryon"] = h.attrs["OmegaBaryon"]
            meta["N_Part_Equiv"] = h.attrs["N_Part_Equiv"]

            try:
                meta["Centre"] = h.attrs["CentreInParent"]
            except KeyError:
                meta["Centre"] = None
                print("No CentreInParent...")

        return meta

    def shift_and_truncate(self):
        """Shift zoom region to box centre and truncate box size."""

        for ptype in [0, 1]:
            self.particle_buffer[ptype]["Coordinates"] -= self.meta["Centre"]

        if not self.args.truncate_to_size:
            return
        else:
            raise NotImplementedError

    def isolate_gas(self):
        """Isolate gas particles, if desired."""

        if self.particle_buffer[1] is None:
            self.particle_buffer[0] = None
            return

        print("Separating DM and gas particles...")
        h = self.header["HubbleParam"]

        with h5.File(self.args.pl_meta_file, "r") as f:
            mgas = f["ZoneI"].attrs["m_gas_msun"] / 1e10 * h
            mdm = f["ZoneI"].attrs["m_dm_msun"] / 1e10 * h
            mips = f["ZoneI"].attrs["MeanInterParticleSeparation_gas_Mpc"] * h

        gas_mask = np.isclose(self.particle_buffer[1]["Masses"], mgas, rtol=1e-4)
        dm1_mask = np.isclose(self.particle_buffer[1]["Masses"], mdm, rtol=1e-4)
        # all particles accounted for and none double-counted:
        assert np.logical_xor(gas_mask, dm1_mask).all()

        n_gas = gas_mask.sum()
        n_dm = dm1_mask.sum()

        print(
            f"Found {n_dm} DM and {n_gas} gas particles "
            f"(ratio: {n_dm/n_gas:.3f}).\n"
            f"   m_dm = {mdm * 1e10 / h:.2e} M_Sun, "
            f"m_gas = {mgas * 1e10 / h:.2e} M_Sun."
        )

        if n_gas == 0:
            self.particle_buffer[0] = None
            return

        self.particle_buffer[0] = {}
        for key in self.particle_buffer[1]:
            self.particle_buffer[0][key] = self.particle_buffer[1][key][gas_mask]
            self.particle_buffer[1][key] = self.particle_buffer[1][key][dm1_mask]

        # Generate smoothing lengths for gas.
        self.particle_buffer[0]["InternalEnergy"] = np.zeros(n_gas)
        self.particle_buffer[0]["SmoothingLength"] = np.zeros(n_gas) + mips

        self.meta["m_gas"] = mgas / h
        self.meta["m_dm"] = mdm / h

    def remap_ids(self):
        """Re-map particle IDs to contiguous range."""

        if self.particle_buffer[0] is not None:
            self.particle_buffer[0]["PeanoHilbertIDs"] = np.copy(
                self.particle_buffer[0]["ParticleIDs"]
            )
            n_type0 = self.particle_buffer[0]["PeanoHilbertIDs"].size
            self.particle_buffer[0]["ParticleIDs"] = np.arange(
                self.current_offset[0] * 2 + 1,
                (self.current_offset[0] * 2 + 1) + (2 * n_type0),
                2,
            )
        if self.particle_buffer[1] is not None:
            self.particle_buffer[1]["PeanoHilbertIDs"] = np.copy(
                self.particle_buffer[1]["ParticleIDs"]
            )
            n_type1 = self.particle_buffer[1]["PeanoHilbertIDs"].size
            self.particle_buffer[1]["ParticleIDs"] = np.arange(
                self.current_offset[1],
                self.current_offset[1] + n_type1,
            )
            # Only shift DM IDs to even if there are baryons
            if self.args.isolate_gas:
                self.particle_buffer[1]["ParticleIDs"] *= 2

        if self.particle_buffer[2] is not None:
            self.particle_buffer[2]["PeanoHilbertIDs"] = np.copy(
                self.particle_buffer[2]["ParticleIDs"]
            )
            n_type2 = self.particle_buffer[2]["PeanoHilbertIDs"].size
            self.particle_buffer[2]["ParticleIDs"] = int(1e15) + np.arange(
                self.current_offset[2],
                self.current_offset[2] + n_type2,
            )
        if self.num_parts[3] > 0:
            raise NotImplementedError("Support for part type 3 in ICs not implemented.")
        if self.num_parts[4] > 0:
            raise NotImplementedError("Support for part type 4 in ICs not implemented.")
        if self.num_parts[5] > 0:
            raise NotImplementedError("Support for part type 5 in ICs not implemented.")

    def init_outfile(self):
        print(f"Initializing output file: {self.out_file}")
        if self.args.isolate_gas:
            h = self.header["HubbleParam"]
            with h5.File(self.args.pl_meta_file, "r") as f:
                OmegaDM = f["Header"].attrs["OmegaDM"]
                OmegaBaryon = f["Header"].attrs["OmegaBaryon"]
                mgas = f["ZoneI"].attrs["m_gas_msun"] / 1e10 * h
                mdm = f["ZoneI"].attrs["m_dm_msun"] / 1e10 * h
            ntype1_old = self.num_parts[1]
            self.num_parts[1] = int(
                np.rint(ntype1_old / (1 + OmegaBaryon * mdm / OmegaDM / mgas))
            )
            self.num_parts[0] = ntype1_old - self.num_parts[1]
            print("Isolating gas particles:")
            print(f" Expect {self.num_parts[1]} DM type 1 particles")
            print(f" and {self.num_parts[0]} gas (type 0) particles.")

        self.header["NumFilesPerSnapshot"] = 1
        self.header["NumPart_ThisFile"] = self.num_parts
        self.header["NumPart_Total"] = [np % 2**32 for np in self.num_parts]
        self.header["NumPart_Total_HighWord"] = [np // 2**32 for np in self.num_parts]

        with h5.File(f"{self.out_file}.incomplete", "w") as f:
            h = f.create_group("Header")
            for key in self.header:
                h.attrs[key] = self.header[key]

            # Bake in metadata for simulation setup
            m = f.create_group("Metadata")
            for key in self.meta:
                if self.meta[key] is not None:
                    m.attrs[key] = self.meta[key]

            if self.args.isolate_gas:
                pt0 = f.create_group("PartType0")
                pt0.create_dataset(
                    "ParticleIDs", shape=(self.num_parts[0],), dtype=int, chunks=True
                )
                pt0.create_dataset(
                    "Coordinates",
                    shape=(self.num_parts[0], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt0.create_dataset(
                    "Velocities",
                    shape=(self.num_parts[0], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt0.create_dataset(
                    "Masses", shape=(self.num_parts[0],), dtype=np.float32, chunks=True
                )
                pt0.create_dataset(
                    "InternalEnergy",
                    shape=(self.num_parts[0],),
                    dtype=np.float64,
                    chunks=True,
                )
                pt0.create_dataset(
                    "SmoothingLength",
                    shape=(self.num_parts[0],),
                    dtype=np.float64,
                    chunks=True,
                )
                if self.args.remap_ids:
                    pt0.create_dataset(
                        "PeanoHilbertIDs",
                        shape=(self.num_parts[0],),
                        dtype=int,
                        chunks=True,
                    )

            if self.num_parts[1] > 0:
                pt1 = f.create_group("PartType1")
                pt1.create_dataset(
                    "ParticleIDs", shape=(self.num_parts[1],), dtype=int, chunks=True
                )
                pt1.create_dataset(
                    "Coordinates",
                    shape=(self.num_parts[1], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt1.create_dataset(
                    "Velocities",
                    shape=(self.num_parts[1], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt1.create_dataset(
                    "Masses", shape=(self.num_parts[1],), dtype=np.float32, chunks=True
                )
                if self.args.remap_ids:
                    pt1.create_dataset(
                        "PeanoHilbertIDs",
                        shape=(self.num_parts[1],),
                        dtype=int,
                        chunks=True,
                    )

            if self.num_parts[2] > 0:
                pt2 = f.create_group("PartType2")
                pt2.create_dataset(
                    "ParticleIDs", shape=(self.num_parts[2],), dtype=int, chunks=True
                )
                pt2.create_dataset(
                    "Coordinates",
                    shape=(self.num_parts[2], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt2.create_dataset(
                    "Velocities",
                    shape=(self.num_parts[2], 3),
                    dtype=np.float64,
                    chunks=True,
                )
                pt2.create_dataset(
                    "Masses", shape=(self.num_parts[2],), dtype=np.float32, chunks=True
                )
                if self.args.remap_ids:
                    pt2.create_dataset(
                        "PeanoHilbertIDs",
                        shape=(self.num_parts[2],),
                        dtype=int,
                        chunks=True,
                    )

    def save_particles(self, input_file):
        """Write the data to a SWIFT-compatible single file."""

        print(f"Writing particles from file {input_file} to {self.out_file}...")
        with h5.File(f"{self.out_file}.incomplete", "r+") as f:
            for ptype in (0, 1, 2):
                if self.particle_buffer[ptype] is None:
                    continue
                ntype_this_file = self.particle_buffer[ptype]["Masses"].size
                for field in self.particle_buffer[ptype]:
                    if field in ("Coordinates", "Velocities"):
                        dest_sel = np.s_[
                            self.current_offset[ptype] : self.current_offset[ptype]
                            + ntype_this_file,
                            :,
                        ]
                    else:
                        dest_sel = np.s_[
                            self.current_offset[ptype] : self.current_offset[ptype]
                            + ntype_this_file
                        ]
                    f[f"PartType{ptype}/{field}"].write_direct(
                        self.particle_buffer[ptype][field],
                        dest_sel=dest_sel,
                    )
                self.current_offset[ptype] += ntype_this_file
                self.particle_buffer[ptype] = None


def parse_arguments():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Convert partial ICs files from IC_Gen "
        "into a single SWIFT compatible file."
    )
    parser.add_argument("workdir", help="Working directory for the IC generation.")

    parser.add_argument(
        "-r", "--repo_dir", help="[Optional] Directory in which to write output."
    )
    parser.add_argument(
        "-i", "--remap_ids", action="store_true", help="Remap IDs to contiguous range?"
    )
    parser.add_argument(
        "-f",
        "--input_file_name",
        help="[Optional] The name of one of the IC files from IC_Gen."
        "By default, it is the name of the immediate working "
        "directory plus '.0.hdf5'.",
    )
    parser.add_argument(
        "-o",
        "--output_file_name",
        help="The output file to contain the SWIFT-compatible ICs.",
    )
    parser.add_argument(
        "-g", "--isolate_gas", action="store_true", help="Isolate gas particles in ICs?"
    )
    parser.add_argument(
        "-m",
        "--pl_meta_file",
        default=None,
        help="Particle load metadata file, to isolate gas particles. "
        "By default, this is `particle_load_info.hdf5' in workdir.",
    )
    parser.add_argument(
        "-t",
        "--truncate_to_size",
        help="Truncate box size to specified value, after shifting "
        "the target object to the centre.",
    )

    args = parser.parse_args()

    if args.input_file_name is None:
        sim_name = args.workdir.split("/")[-1]
        args.input_file_name = os.path.join(args.workdir, "ICs", f"{sim_name}.0.hdf5")
    if args.pl_meta_file is None:
        args.pl_meta_file = os.path.join(args.workdir, "particle_load_info.hdf5")

    # Some sanity checks
    if not os.path.isfile(args.pl_meta_file):
        raise OSError(f"Could not find PL metadata file {args.pl_meta_file}!")
    if not os.path.isfile(args.input_file_name):
        raise OSError(f"Could not find input ICs file {args.input_file_name}!")

    return args


if __name__ == "__main__":
    main()
