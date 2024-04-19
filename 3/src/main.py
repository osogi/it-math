from __future__ import annotations
import os
from typing import Any, Dict, NamedTuple, Protocol
import PIL.Image
import numpy
import PIL
import argparse

import numpy.typing


class SVDdata:
    def __init__(self) -> None:
        self.U: numpy.matrix
        self.S: numpy._ArrayFloat_co
        self.Vh: numpy.matrix

    @staticmethod
    def new(a: NamedTuple) -> SVDdata:
        self = SVDdata()
        self.U = a.U
        self.S = a.S
        self.Vh = a.Vh

        self.U = numpy.matrix(self.U.astype(numpy.float32))
        self.S = self.S.astype(numpy.float32)
        self.Vh = numpy.matrix(self.Vh.astype(numpy.float32))

        return self


class SVDer(Protocol):
    def svd(self, a: numpy.matrix) -> SVDdata: ...


class SVDStadartLibrary:
    def svd(self, a: numpy.matrix) -> SVDdata:
        return SVDdata.new(numpy.linalg.svd(a))


COMRESS_STRUCT_HEADER_SIZE = 0x10
FLOAT_SIZE = 4


class CompressedData:

    def __init__(self) -> None:
        self.lst: list[SVDdata] = []
        self.n: int = -1
        self.m: int = -1
        self.k: int = -1
        self.mode: str = "NONE"

    @staticmethod
    def new(a: list[SVDdata], mode: str) -> CompressedData:
        self: CompressedData = CompressedData()
        self.lst = a
        self.n = a[0].U.shape[0]
        self.m = a[0].Vh.shape[1]
        self.k = len(a[0].S)
        self.mode = mode

        return self

    def serialize(self):
        data: bytearray = bytearray()

        # header
        data.extend(numpy.uint32(self.n).tobytes())
        data.extend(numpy.uint32(self.m).tobytes())
        data.extend(numpy.uint32(self.k).tobytes())
        mode = self.mode[: min(len(self.mode), 4)]
        data.extend(mode.encode("ASCII"))
        while len(data) < COMRESS_STRUCT_HEADER_SIZE:
            data.append(0x0)

        # body
        for svd in self.lst:
            data.extend(svd.U.tobytes())
            data.extend(svd.S.tobytes())
            data.extend(svd.Vh.tobytes())
        return data

    def save(self, filename: str):
        with open(filename, "wb") as f:
            f.write(self.serialize())

    @staticmethod
    def deserialize(bts: bytes) -> CompressedData:
        self: CompressedData = CompressedData()

        # header
        self.n = int(numpy.frombuffer(bts, offset=0x0, dtype=numpy.uint32, count=1)[0])
        self.m = int(numpy.frombuffer(bts, offset=0x4, dtype=numpy.uint32, count=1)[0])
        self.k = int(numpy.frombuffer(bts, offset=0x8, dtype=numpy.uint32, count=1)[0])
        self.mode = bts[0xC:0x10].decode("ASCII").replace("\x00", "")

        # body
        body_size = len(bts) - COMRESS_STRUCT_HEADER_SIZE
        one_channel_size = (self.m + self.n + 1) * self.k * FLOAT_SIZE
        channel_count = body_size // one_channel_size
        if one_channel_size * channel_count != body_size:
            print("Can't deserialize this bytes to CompressedData, some problems with sizes")
            exit(1)
        else:
            self.lst = []
            usize = self.n * self.k
            vhsize = self.k * self.m
            cur_offset = COMRESS_STRUCT_HEADER_SIZE
            for i in range(channel_count):
                svd_data = SVDdata()
                svd_data.U = numpy.matrix(
                    numpy.frombuffer(bts, offset=cur_offset, dtype=numpy.float32, count=usize).reshape((self.n, self.k))
                )
                cur_offset += svd_data.U.size * FLOAT_SIZE

                svd_data.S = numpy.frombuffer(bts, offset=cur_offset, dtype=numpy.float32, count=self.k)
                cur_offset += svd_data.S.size * FLOAT_SIZE

                svd_data.Vh = numpy.matrix(
                    numpy.frombuffer(bts, offset=cur_offset, dtype=numpy.float32, count=vhsize).reshape(
                        (self.k, self.m)
                    )
                )
                cur_offset += svd_data.Vh.size * FLOAT_SIZE
                self.lst.append(svd_data)
        return self

    @staticmethod
    def load(filename: str) -> CompressedData:
        with open(filename, "rb") as f:
            return CompressedData.deserialize(f.read())


def compress(
    img: PIL.Image.Image,
    dst_size: int,
    svder: SVDer,
) -> CompressedData:
    n, m = img.size

    if img.palette is not None:
        print("Compression for image with palette is not supported")
        exit(1)

    matrixs: list[numpy.matrix] = [
        numpy.matrix(numpy.zeros([n, m], dtype=int)) for _ in range(len(img.getpixel((0, 0))))
    ]

    k = int(min(n, m, (dst_size - COMRESS_STRUCT_HEADER_SIZE) // (len(matrixs) * (m + n + 1) * FLOAT_SIZE)))
    if k <= 0:
        print(f"Can't compress image to {dst_size} bytes")
        exit(1)

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

    return CompressedData.new(svddatas, img.mode)


def decompress(cd: CompressedData) -> PIL.Image.Image:
    img = PIL.Image.new(cd.mode, (cd.n, cd.m))

    matrixes: list[numpy.matrix] = []
    for s in cd.lst:
        S = numpy.diag(s.S)
        a = s.U * S * s.Vh
        matrixes.append(a)

    for i in range(cd.n):
        for j in range(cd.m):
            img.putpixel((i, j), tuple([int(m[i, j]) for m in matrixes]))

    return img


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


def filename_and_sz(filename: str):
    return f"{filename} ({os.path.getsize(filename)} bytes)"


def main():
    args = parse_args()
    mode = args["mode"]
    infile = args["in-file"]
    outfile = args["out-file"]
    if mode == MODE_COMPRESS:
        img: PIL.Image.Image = PIL.Image.open(infile)
        bytesize: int = os.path.getsize(infile)
        finalsize: int = bytesize // args["compression"]
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
        cd.save(outfile)
        print(f"[Compressed]: {filename_and_sz(infile)} -> {filename_and_sz(outfile)}")
    elif mode == MODE_DECOMPRESS:
        cd: CompressedData = CompressedData.load(infile)
        img = decompress(cd)
        img.save(outfile, "BMP")
        print(f"[Decompressed]: {filename_and_sz(infile)} -> {filename_and_sz(outfile)}")
    else:
        print(f"MODE={mode} is not supported")
        exit(1)


if __name__ == "__main__":
    main()
