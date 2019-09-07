
import os
import argparse
from PIL import Image


def list_indir(dir):
    return [ os.path.join(dir, f) for f in os.listdir(dir) ]

def main(indir, outdir, dim):
    for f in list_indir(indir):
        img = Image.open(f)
        img = img.resize((dim, dim), Image.ANTIALIAS)
        img.save(os.path.join(outdir, os.path.basename(f)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir",  type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--dim", type=int, default=50)
    args = parser.parse_args()
    indir  = args.indir
    outdir = args.outdir
    dim = args.dim
    main(indir, outdir, dim)

