#!/usr/bin/env python3
"""Calculate docked-pose strain for one SDF record or a multi-record 3D ligand SDF.

Reproduce the OXTR fast screen in Python 3.12:
    python -m pip install rdkit==2026.9.2 torch==2.8.0 scipy numpy
    python oxtr_ligand_strain.py orig_table_1.sdf --record 6815 --out rank2
    python oxtr_ligand_strain.py orig_table_1.sdf --out all_results

Default: 15 ETKDGv3 attempts split into 3 + 12, plus the released docked pose.
Use original 1-based SDF record numbers for identical random seeds. For an
extracted single record, --seed-record-offset 6814 reproduces record 6815.
First use downloads the pinned AIMNet2-CPCM weights into a local cache.
Offline: pass --model /path/to/b973c_cpcm_ens_f.jpt (same SHA-256).
Outputs: summary.csv, energies.csv (including all four predictions per accepted
geometry), free_lowest.sdf, bound_relaxed.sdf. --save-all also writes all accepted
free geometries. All energies exclude restraints. One ligand per SDF record.
MMFF94s-ineligible molecules are flagged in summary.csv. The separate neural
refinement protocol is outside this fast-screen script.
Model: Gokcan & Isayev, DOI 10.1021/acs.jcim.6c02468.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from contextlib import ExitStack
from shutil import copyfileobj

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist

MODEL_URL = ('https://raw.githubusercontent.com/isayevlab/ligand-strain-aimnet2/'
             '04fe5225f9e838a0cbe5b0abe9579620dd69c5da/models/b973c_cpcm_ens_f.jpt')
SHA256 = '53e055414c90b5d08cd0a164eba139deddda3d6be56af7b5a1ef3c3cd40efc8a'
EV_TO_KCAL = 23.06054783
ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}


def verify(path):
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SHA256:
            raise ValueError(f'Unexpected model checksum: {path}')


def load_members(local_model, device):
    path = local_model or Path.home() / '.cache/oxtr-strain/b973c_cpcm_ens_f.jpt'
    if local_model is None and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            tmp = Path(temp.name)
        try:
            with urlopen(MODEL_URL, timeout=120) as response, tmp.open('wb') as out:
                copyfileobj(response, out)
            verify(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    verify(path)
    model = torch.jit.load(str(path), map_location=device).eval()
    return [getattr(model.models, str(i)).module for i in range(4)]


def identity(mol):
    copy = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(copy, replaceExistingTags=True)
    return Chem.MolToSmiles(Chem.RemoveHs(copy))


def optimize(mol, stereo, restrained=False):
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if props is None:
        raise ValueError('missing MMFF94s parameters')
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, nonBondedThresh=100)
    if restrained:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                ff.MMFFAddPositionConstraint(atom.GetIdx(), 0.0, 10.0)
    if ff.Minimize(maxIts=1200) != 0:
        return False
    x = mol.GetConformer().GetPositions()
    return np.isfinite(x).all() and pdist(x).min() >= 0.65 and identity(mol) == stereo


def geometries(mol, seed_record, attempts):
    if mol is None or not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError('missing or invalid input 3D pose')
    mol = Chem.AddHs(mol, addCoords=True)
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() not in ELEMENTS
                                              or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('requires one supported closed-shell ligand')
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    stereo = identity(mol)
    bound = Chem.Mol(mol)
    if not optimize(bound, stereo, restrained=True):
        raise ValueError('restrained pose failed geometry checks')
    poses = [('docked_relaxed', bound)]
    released = Chem.Mol(bound)
    if optimize(released, stereo):
        poses.append(('released_pose', released))
    embedded = 0
    for batch, (count, seed) in enumerate(((min(3, attempts), 29 + seed_record),
                                           (max(0, attempts - 3), 1000029 + seed_record)), 1):
        if count == 0:
            continue
        work = Chem.Mol(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed, p.numThreads, p.pruneRmsThresh = seed, 1, 0.3
        ids = AllChem.EmbedMultipleConfs(work, numConfs=count, params=p)
        embedded += len(ids)
        for cid in ids:
            trial = Chem.Mol(work, confId=cid)
            if optimize(trial, stereo):
                poses.append((f'batch{batch}_conf{cid}', trial))
    if len(poses) == 1:
        raise ValueError('no free geometry passed checks')
    return poses, embedded


def score(poses, members, device):
    mol = poses[0][1]
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], device=device)
    q = float(Chem.GetFormalCharge(mol))
    result = []
    with torch.inference_mode():
        for start in range(0, len(poses), 4):
            xyz = np.array([m.GetConformer().GetPositions() for _, m in poses[start:start + 4]])
            xyz -= xyz.mean(axis=1, keepdims=True)
            args = dict(coord=torch.tensor(xyz, device=device, dtype=torch.float32),
                        numbers=z.repeat(len(xyz), 1),
                        charge=torch.full((len(xyz),), q, device=device, dtype=torch.float32))
            predictions = torch.stack([net(dict(args))['energy'] for net in members], dim=1)
            result.extend(predictions.cpu().numpy().tolist())
    values = np.asarray(result)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite ensemble prediction')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sdf', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--record', type=int, help='Only process this 1-based SDF record')
    parser.add_argument('--seed-record-offset', type=int, default=0)
    parser.add_argument('--conformers', type=int, default=15, help='Number of new embedding attempts')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--save-all', action='store_true')
    args = parser.parse_args()
    if args.conformers < 1 or (args.record is not None and args.record < 1) or args.seed_record_offset < 0:
        parser.error('record and conformer counts must be positive; seed offset must be nonnegative')
    if not args.sdf.is_file() or args.out.exists():
        parser.error('supply an existing input file and a new output directory')
    torch.set_num_threads(1)
    members = load_members(args.model, args.device)
    args.out.mkdir(parents=True)
    with ExitStack() as stack:
        def table(name, columns):
            stream = stack.enter_context((args.out / name).open('w', newline=''))
            writer = csv.writer(stream)
            writer.writerow(columns)
            return writer
        def sdf(name):
            writer = Chem.SDWriter(str(args.out / name))
            stack.callback(writer.close)
            return writer
        summary = table('summary.csv', ['sdf_record','input_name','status','attempts','embedded',
                        'free_retained','lowest_label','bound_eV','lowest_free_eV','strain_kcal_mol'])
        energies = table('energies.csv', ['sdf_record','label','member0_eV','member1_eV','member2_eV',
                        'member3_eV','ensemble_eV','relative_to_lowest_free_kcal_mol'])
        bound_out, free_out = sdf('bound_relaxed.sdf'), sdf('free_lowest.sdf')
        all_out = sdf('free_conformers.sdf') if args.save_all else None
        records = Chem.SDMolSupplier(str(args.sdf), removeHs=False)
        if args.record and args.record > len(records):
            parser.error('requested record exceeds SDF record count')
        indices = [args.record] if args.record else range(1, len(records) + 1)
        good = 0
        for rec in indices:
            mol = records[rec - 1]
            name = mol.GetProp('_Name') if mol is not None and mol.HasProp('_Name') else ''
            try:
                poses, embedded = geometries(mol, rec + args.seed_record_offset, args.conformers)
                scores = score(poses, members, args.device)
            except (ValueError, RuntimeError) as err:
                summary.writerow([rec, name, str(err), args.conformers, '', 0, '', '', '', ''])
                continue
            means = scores.mean(axis=1)
            best = 1 + np.argmin(means[1:])
            delta = (means - means[best]) * EV_TO_KCAL
            for i, (label, geometry) in enumerate(poses):
                energies.writerow([rec, label, *scores[i], means[i], delta[i]])
                geometry.SetProp('SDF_record', str(rec))
                geometry.SetProp('conformer_label', label)
                geometry.SetDoubleProp('ensemble_eV', float(means[i]))
                if i == 0: bound_out.write(geometry)
                if i == best: free_out.write(geometry)
                if all_out is not None and i > 0: all_out.write(geometry)
            status = 'negative estimate' if delta[0] < 0 else 'ok'
            summary.writerow([rec, name, status, args.conformers, embedded, len(poses)-1,
                              poses[best][0], means[0], means[best], delta[0]])
            good += 1
    print(f'{good}/{len(indices)} records scored. Results: {args.out}')
    return 0 if good == len(indices) else 1


if __name__ == '__main__':
    raise SystemExit(main())
