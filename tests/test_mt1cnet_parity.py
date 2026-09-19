"""MT1CNet: the C++ twin must produce the same number as the torch module.

Same reasoning as test_mt1net_parity — `load_bin` validates a weight file by ELEMENT COUNT only,
so a layout drift leaving the total unchanged loads silently and produces plausible wrong numbers.
This competitor exists to be COMPARED against MT1Net, so a silent drift here would not look like a
bug; it would look like a result.
"""
import struct
import subprocess

import pytest
import torch

from models import MT1CNET_LAYER_DEFS, MT1CNET_PARAMS, MT1CNet

CPP = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mt1_pool.h"
int main(int argc, char** argv) {
    std::vector<float> W(MT1CNET_PARAMS), in(MT1C_IN);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(W.data(), sizeof(float), MT1CNET_PARAMS, f) != (size_t)MT1CNET_PARAMS) return 2;
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f || fread(in.data(), sizeof(float), MT1C_IN, f) != (size_t)MT1C_IN) return 3;
    fclose(f);
    printf("%.7e\n", mt1cnet_forward(W.data(), in.data(), 0.f));
    return 0;
}
'''


def flatten(net):
    sd = net.state_dict()
    parts = []
    for name, _, _ in MT1CNET_LAYER_DEFS:
        parts.append(sd[f'{name}.weight'].flatten())
        parts.append(sd[f'{name}.bias'].flatten())
    return torch.cat(parts)


@pytest.fixture(scope='module')
def cpp_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp('mt1cnet')
    src = d / 'fwd.cpp'
    src.write_text(CPP)
    exe = d / 'fwd'
    r = subprocess.run(['g++', '-std=c++20', '-O2', '-I', '.', '-o', str(exe), str(src), '-lm'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f'g++ unavailable: {r.stderr[:300]}')
    return exe


def run_cpp(exe, tmp_path, w, x):
    wf, xf = tmp_path / 'w.bin', tmp_path / 'x.bin'
    wf.write_bytes(struct.pack(f'{w.numel()}f', *w.tolist()))
    xf.write_bytes(struct.pack(f'{x.numel()}f', *x.tolist()))
    out = subprocess.run([str(exe), str(wf), str(xf)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return float(out.stdout.strip())


class TestParity:
    def test_flat_vector_length(self):
        assert flatten(MT1CNet()).numel() == MT1CNET_PARAMS == 11177

    def test_input_width(self):
        assert MT1CNet.N_IN == 146 == MT1CNet.N_SYMS * MT1CNet.PER_SYM + 2

    @pytest.mark.parametrize('seed', [0, 1, 7, 42, 1234])
    def test_same_output_as_torch(self, cpp_exe, tmp_path, seed):
        torch.manual_seed(seed)
        net = MT1CNet().eval()
        x = torch.randn(1, MT1CNet.N_IN)
        with torch.no_grad():
            want = float(net(x)[0, 0])
        got = run_cpp(cpp_exe, tmp_path, flatten(net), x[0])
        assert abs(got - want) < 1e-4 * max(1.0, abs(want)), f'C++ {got} vs torch {want}'

    def test_zero_weights_give_zero(self, cpp_exe, tmp_path):
        net = MT1CNet().eval()
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
        x = torch.randn(1, MT1CNet.N_IN)
        assert abs(run_cpp(cpp_exe, tmp_path, flatten(net), x[0])) < 1e-6

    def test_every_input_slot_reaches_the_output(self, cpp_exe, tmp_path):
        """146 inputs laid out as 12 symbols x 12 fields + 2. An off-by-one in the builder would
        shift every symbol's block and still produce a number."""
        torch.manual_seed(5)
        net = MT1CNet().eval()
        w = flatten(net)
        base_x = torch.zeros(MT1CNet.N_IN)
        base = run_cpp(cpp_exe, tmp_path, w, base_x)
        moved = 0
        for k in range(MT1CNet.N_IN):
            x = base_x.clone()
            x[k] = 5.0
            if abs(run_cpp(cpp_exe, tmp_path, w, x) - base) > 1e-7:
                moved += 1
        # ReLU can gate some paths shut at a given weight draw; most must still get through
        assert moved > MT1CNet.N_IN * 0.5, f'only {moved}/{MT1CNet.N_IN} inputs reach the output'

    def test_taper_is_four_layers_and_decreasing(self):
        widths = [MT1CNet.N_IN] + [o for _, o, _ in MT1CNET_LAYER_DEFS]
        assert len(MT1CNET_LAYER_DEFS) == 4
        assert widths == sorted(widths, reverse=True), f'not tapering: {widths}'
        assert widths[-1] == 1

    def test_volume_is_not_in_the_layout(self):
        """Excluded deliberately: with no history there is no norm to read a raw volume against."""
        from pathlib import Path
        hdr = (Path(__file__).resolve().parent.parent / 'mt1_pool.h').read_text()
        blk = hdr[hdr.index('static constexpr int CN_OPEN'):hdr.index('static inline float mt1cnet_forward')]
        assert 'VOL' not in blk.upper()
        assert MT1CNet.PER_SYM == 12, 'per-symbol width implies volume crept back in'
