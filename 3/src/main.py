import os
from typing import Any, Dict, NamedTuple, Protocol
import PIL.Image
import numpy
import PIL
import argparse

import numpy.typing


class SVDdata:
    def __init__(self, a: NamedTuple):
        self.U: numpy.matrix = a.U
        self.S: numpy._ArrayFloat_co = a.S
        self.Vh: numpy.matrix = a.Vh


class SVDer(Protocol):
    def svd(self, a: numpy.matrix) -> SVDdata: ...


class SVDStadartLibrary:
    def svd(self, a: numpy.matrix) -> SVDdata:
        return SVDdata(numpy.linalg.svd(a))


class CompressedData:
    def __init__(self, a: list[SVDdata]):
        self.lst = a
        self.n = a[0].U.shape[0]
        self.m = a[0].Vh.shape[1]


MODE_COMPRESS: str = "compress"
MODE_DECOMPRESS: str = "decompress"

METHOD_NUMPY: str = "numpy"
METHOD_SIMPLE: str = "simple"
METHOD_ADVANCED: str = "advanced"


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Videos to images")
    parser.add_argument("in-file", type=str, default="a.in", help="Path to source file")
    parser.add_argument("out-file", type=str, default="a.out", help="Path to result file")
    parser.add_argument("--mode", type=str, default=MODE_COMPRESS, help=f"{MODE_COMPRESS}/{MODE_DECOMPRESS}")
    parser.add_argument(
        "--method",
        type=str,
        default=METHOD_NUMPY,
        help=f"{METHOD_NUMPY}/{METHOD_SIMPLE}/{METHOD_ADVANCED}; Choose SVD method",
    )
    parser.add_argument("--compression", type=int, default=4, help="How many times to compress the image")
    return parser.parse_args().__dict__


def compress(
    img: PIL.Image.Image,
    dst_size: int,
    svder: SVDer,
) -> CompressedData:
    n, m = img.size
    k = int(min(n, m) * 0.01)

    if img.palette is not None:
        print("Comprasion for image with palette is not supported")
        exit(1)

    matrixs: list[numpy.matrix] = [
        numpy.matrix(numpy.zeros([n, m], dtype=int)) for _ in range(len(img.getpixel((0, 0))))
    ]
    for i in range(n):
        for j in range(m):
            clrs = img.getpixel((i, j))
            for c in range(len(clrs)):
                matrixs[c][i, j] = clrs[c]
    svddatas: list[SVDdata] = []

    for m in matrixs:
        s = svder.svd(m)  # U S Vh
        s.U = numpy.matrix(s.U.compress([True] * k, axis=1))
        s.S = s.S[:k]
        s.Vh = numpy.matrix(s.Vh.compress([True] * k, axis=0))
        svddatas.append(s)
        # print(s.U)

    return CompressedData(svddatas)


def decompress(cd: CompressedData) -> PIL.Image.Image:
    print(f"{cd.n} {cd.m}")
    img = PIL.Image.new("RGB", (cd.n, cd.m))
    
    matrixes: list[numpy.matrix] = []
    for s in cd.lst:
        S = numpy.diag(s.S)
        a = s.U * S * s.Vh
        matrixes.append(a)
    
    for i in range(cd.n):
        for j in range(cd.m):
            img.putpixel((i, j), tuple([int(m[i, j]) for m in matrixes]))
    
    return img

def main():
    args = parse_args()
    mode = args["mode"]
    if mode == MODE_COMPRESS:
        infile = args["in-file"]
        img: PIL.Image.Image = PIL.Image.open(infile)
        bytesize: int = os.path.getsize(infile)
        finalsize: int = bytesize / args["compression"]
        svder: SVDer

        method_str = args["method"]
        if method_str == METHOD_NUMPY:
            svder = SVDStadartLibrary()
        elif method_str == METHOD_SIMPLE:
            ...
        elif method_str == METHOD_ADVANCED:
            ...
        else:
            print(f"METHOD={method_str} is not supported")
            exit(1)

        cd = compress(img, finalsize, svder)
        img = decompress(cd)
        img.save(args["out-file"], "BMP")
    elif mode == MODE_DECOMPRESS:
        ...
        # decompress()
    else:
        print(f"MODE={mode} is not supported")
        exit(1)


if __name__ == "__main__":
    main()
