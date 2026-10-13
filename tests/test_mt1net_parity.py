"""MT1Net: the C++ twin must produce the same number as the Python model.

`load_bin` validates a weight file only by ELEMENT COUNT. So a layout drift that leaves the total
unchanged — two layers swapped, a weight/bias boundary off by one — loads silently and produces
plausible wrong numbers. That failure mode has already cost this project once: v0.6.0.0's
`stock_close_pos` needed a C++ twin kept in sync by hand, and the section offsets in `today_arr`
were bare literals until a static_assert was added.

This compiles the header's forward pass standalone, feeds it the same random weights and input as
the torch module, and compares.
"""
import struct
import subprocess

import pytest
import torch

from models import MT1NET_LAYER_DEFS, MT1NET_PARAMS, MT1Net

CPP = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mt1_pool.h"
int main(int argc, char** argv) {
    // argv[1] = weights file, argv[2] = input file, argv[3] = extra
    std::vector<float> W(MT1NET_PARAMS), in74(74);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(W.data(), sizeof(float), MT1NET_PARAMS, f) != (size_t)MT1NET_PARAMS) return 2;
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f || fread(in74.data(), sizeof(float), 74, f) != 74u) return 3;
    fclose(f);
    printf("%.7e\n", mt1net_forward(W.data(), in74.data(), (float)atof(argv[3])));
    return 0;
}
'''


def flatten(net):
    """Weights in MT1NET_LAYER_DEFS order: each layer's (out x in) weights then its bias."""
    sd = net.state_dict()
    parts = []
    for name, _, _ in MT1NET_LAYER_DEFS:
        parts.append(sd[f'{name}.weight'].flatten())
        parts.append(sd[f'{name}.bias'].flatten())
    return torch.cat(parts)


@pytest.fixture(scope='module')
def cpp_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp('mt1net')
    src = d / 'fwd.cpp'
    src.write_text(CPP)
    exe = d / 'fwd'
    r = subprocess.run(['g++', '-std=c++20', '-O2', '-I', '.', '-o', str(exe), str(src), '-lm'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f'g++ unavailable or build failed: {r.stderr[:300]}')
    return exe


def run_cpp(exe, tmp_path, w, x, extra=0.0):
    wf, xf = tmp_path / 'w.bin', tmp_path / 'x.bin'
    wf.write_bytes(struct.pack(f'{w.numel()}f', *w.tolist()))
    xf.write_bytes(struct.pack(f'{x.numel()}f', *x.tolist()))
    out = subprocess.run([str(exe), str(wf), str(xf), str(extra)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return float(out.stdout.strip())


class TestParity:
    def test_flat_vector_is_the_right_length(self):
        assert flatten(MT1Net()).numel() == MT1NET_PARAMS

    @pytest.mark.parametrize('seed', [0, 1, 7, 42, 1234])
    def test_same_output_as_torch(self, cpp_exe, tmp_path, seed):
        torch.manual_seed(seed)
        net = MT1Net().eval()
        x = torch.randn(1, 74)
        with torch.no_grad():
            want = float(net(x)[0, 0])
        got = run_cpp(cpp_exe, tmp_path, flatten(net), x[0])
        assert abs(got - want) < 1e-4 * max(1.0, abs(want)), f'C++ {got} vs torch {want}'

    def test_reserved_input_matches_too(self, cpp_exe, tmp_path):
        torch.manual_seed(3)
        net = MT1Net().eval()
        x = torch.randn(1, 74)
        for extra in (0.0, 1.0, -2.5):
            with torch.no_grad():
                want = float(net(x, extra=torch.tensor([[extra]]))[0, 0])
            got = run_cpp(cpp_exe, tmp_path, flatten(net), x[0], extra)
            assert abs(got - want) < 1e-4 * max(1.0, abs(want)), f'extra={extra}'

    def test_zero_weights_give_zero(self, cpp_exe, tmp_path):
        net = MT1Net().eval()
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
        x = torch.randn(1, 74)
        assert abs(run_cpp(cpp_exe, tmp_path, flatten(net), x[0])) < 1e-6

    def test_block_separation_holds_in_cpp_too(self, cpp_exe, tmp_path):
        """A portfolio feature must not move the market trunk — asserted on the C++ side as well,
        since the two trunks share offsets via a base and an off-by-one there would silently mix
        market and portfolio weights."""
        torch.manual_seed(11)
        net = MT1Net().eval()
        w = flatten(net)
        x = torch.zeros(74)
        x[5] = 2.0                                    # market daily feature
        base = run_cpp(cpp_exe, tmp_path, w, x)
        x2 = x.clone()
        x2[37 + 5] = 9.0                              # same index, portfolio half
        moved = run_cpp(cpp_exe, tmp_path, w, x2)
        assert base != moved, 'portfolio half is not reaching the net at all'
        with torch.no_grad():
            t_base = float(net(x.unsqueeze(0))[0, 0])
            t_moved = float(net(x2.unsqueeze(0))[0, 0])
        assert abs(base - t_base) < 1e-4 * max(1.0, abs(t_base))
        assert abs(moved - t_moved) < 1e-4 * max(1.0, abs(t_moved))
