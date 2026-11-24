# Docked ligand strain for OXTR screening

How much higher in energy is a ligand in its docked geometry than in a relaxed conformation? This Python workflow estimates that difference for ligands from an oxytocin receptor (OXTR) docking screen. It takes ligand coordinates as input, so the same workflow can be applied to poses from other targets.

The workflow combines RDKit conformer generation, MMFF94s geometry optimization, and single-point scoring with the four members of the published AIMNet2-CPCM ensemble. The project contribution is the screening workflow, including record-based seeding, geometry checks, per-member energy reporting, and export of the structures used in each comparison.

## Run the example

Use Python 3.12. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python oxtr_ligand_strain.py examples/ethanol.sdf --out results-demo
```

A CPU run of this example is included in [examples/ethanol-output](examples/ethanol-output). Its CSVs and paired SDFs show how to trace a reported estimate back to the scored geometries.

The ethanol file is a hand-built 3D input for trying the command. It carries no receptor or experimental activity information. Use a new output directory for each run. The default device is CPU. On a machine with a CUDA-enabled PyTorch installation, add `--device cuda`.

On first use the script downloads the pinned ensemble and verifies its SHA-256 checksum. To use a local copy, pass `--model /path/to/b973c_cpcm_ens_f.jpt`. The same checksum is required.

## Screen a docking table

Supply a 3D SDF with one connected, closed-shell ligand per record and a chemically appropriate formal charge. The script adds hydrogens with coordinates. Each ligand must have supported elements and MMFF94s parameters.

```bash
# Score every ligand using the default 15 embedding attempts
python oxtr_ligand_strain.py docked_ligands.sdf --out results-all

# Inspect the free conformers for one ligand using its 1-based record number
python oxtr_ligand_strain.py docked_ligands.sdf --record 12 --save-all --out results-record12
```

## How strain is estimated

1. Relax the input geometry with MMFF94s while restraining heavy-atom positions. This produces the bound-like geometry.
2. Release those restraints and minimize again to generate one candidate free conformer. Independently request 15 ETKDGv3 embedding attempts by default, split into batches of 3 and 12. RMSD pruning, minimization, and geometry checks determine how many conformers survive. Set the attempt budget with `--conformers`.
3. Score the accepted geometries with all four AIMNet2-CPCM models and average their predicted energies per geometry. Select the free conformer with the lowest ensemble mean.

```text
strain (kcal/mol) = (mean bound-like energy − lowest sampled free mean energy) × 23.06054783
```

The model energies are in eV and exclude MMFF restraint energy. A positive value indicates an energetic cost relative to the sampled free reference. A negative estimate is retained and flagged when the bound-like geometry scores lower than every sampled free geometry. Conformer sampling and the input protonation state affect the comparison. Use this result alongside docking and interaction analysis. Binding affinity and experimental activity require independent evidence.

## Outputs

| File | Contents |
| --- | --- |
| summary.csv | One row per processed input record, including its status, conformer counts, selected reference, and strain estimate |
| energies.csv | Four model predictions, ensemble mean, and relative energy for each accepted geometry |
| bound_relaxed.sdf | Restrained, relaxed bound-like geometry for each scored ligand |
| free_lowest.sdf | Lowest-energy sampled free geometry for each scored ligand |
| free_conformers.sdf | All accepted free geometries when using `--save-all` |

Record numbers determine the random seeds. If original record 12 becomes record 1 in an extracted SDF, add `--seed-record-offset 11` to preserve its seeds. Check the status column for excluded records. The command returns exit code 1 if any requested records could not be scored.

## Model attribution

The pretrained ensemble and underlying method are from Gokcan and Isayev. The script retrieves the weights from a fixed upstream commit. [Model repository at the pinned revision](https://github.com/isayevlab/ligand-strain-aimnet2/tree/04fe5225f9e838a0cbe5b0abe9579620dd69c5da) · [Method paper](https://doi.org/10.1021/acs.jcim.6c02468)
