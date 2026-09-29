#!/usr/bin/env python3
"""image/fabric_ring.py: the NCCL graph a ring of boxes boots with.

Fakes /sys/class/infiniband and /sys/class/net under a temp tree (the
module's SYS_ROOT, and every path it reads, moves to the fake) and checks:
  - with both PCIe roots of each port carrying an IPv4, two RDMA devices per
    port are kept and every env var and graph channel matches what the old
    two-device code produced
  - with the second root's netdevs unaddressed (no RoCE v2 IPv4 GID), one
    device per port is kept and the graph XML is well formed
  - a port with zero usable devices raises, naming the interface
  - VLLM_ARX_RING and ARX_RING_*_HCAS are set by default and dropped when
    FABRIC_RING_ARX=0
  - NCCL_IB_GID_INDEX is dropped: ring mode picks each device's GID by its
    own address, and the cables' indexes differ
CPU only, reads nothing from the real host:

    python3 dev/patch-tests/_fabric_ring_test.py
"""
import importlib.util
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fabric_ring_under_test",
    os.path.join(HERE, "..", "..", "image", "fabric_ring.py"),
)
fabric_ring = importlib.util.module_from_spec(_spec)
os.environ.pop("MENTAT_FABRIC_LAYOUT", None)  # keep the import a no-op
_spec.loader.exec_module(fabric_ring)

# Device names follow the real boxes: first root roce<lowercase>*, second
# root roceP2*, both roots' functions sharing one BDF tail.
A_IFACE = "enp1s0f0np0"  # netdev of the port toward prev, first root
B_IFACE = "enp1s0f1np1"  # netdev of the port toward next, first root
A_DEVS = ("rocep1s0f0", "roceP2p1s0f0")  # pci 0000:01:00.1, 0002:01:00.1
B_DEVS = ("rocep1s0f1", "roceP2p1s0f1")  # pci 0000:01:00.2, 0002:01:00.2
LINK_LOCAL = "fe80:0000:0000:0000:0000:0000:0000:0002"


def v4_gid(a: int, b: int, c: int, d: int) -> str:
    return "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" % (a, b, c, d)


# bdf per device, first root then second
# (pci bus:device.function, RDMA device, the netdev of the device itself).
# The second root is domain 0002, as on the real boxes.
PORTS = [
    ("0000:01:00.1", A_DEVS[0], "enp1s0f0np0"),
    ("0000:01:00.2", B_DEVS[0], "enp1s0f1np1"),
    ("0002:01:00.1", A_DEVS[1], "enP2p1s0f0np0"),
    ("0002:01:00.2", B_DEVS[1], "enP2p1s0f1np1"),
]
# Sorting key the module uses for the union: the real path of each device's
# PCI function, so first-root devices sort before second-root ones here.
UNION_ORDER = (
    A_DEVS[0], B_DEVS[0], A_DEVS[1], B_DEVS[1]
)


class FakeSys:
    """A /sys tree with two ConnectX ports (prev, next) and one device per
    root per port. A device that is not in `addressed` has only link-local
    GIDs, like the real second root before an address is put on its netdev."""

    def __init__(self, addressed: dict[str, tuple[int, int, int, int]]):
        self.root = tempfile.mkdtemp()
        self.sys = os.path.join(self.root, "sys")
        for bdf, dev, netdev in PORTS:
            base = os.path.join(self.sys, "class", "infiniband", dev)
            self._gid(dev, 0, LINK_LOCAL, "RoCE v1", netdev)
            self._gid(dev, 1, LINK_LOCAL, "RoCE v2", netdev)
            if dev in addressed:
                self._gid(dev, 2, v4_gid(*addressed[dev]), "RoCE v2", netdev)
            pci = os.path.join(self.sys, "bus", "pci", "devices", bdf)
            # The PCI function owns its netdevs; both are reached through
            # /sys/class/infiniband/<dev>/device.
            os.makedirs(os.path.join(pci, "net"), exist_ok=True)
            open(os.path.join(pci, "net", netdev), "w").close()
            pci_ib = os.path.join(pci, "infiniband")
            os.makedirs(pci_ib)
            os.symlink(os.path.relpath(base, pci_ib), os.path.join(pci_ib, dev))
            # /sys/class/infiniband/<dev>/device -> its PCI function
            os.symlink(os.path.relpath(pci, base), os.path.join(base, "device"))
            # The addressed (first-root) netdevs are the ones mentat names;
            # each points at its own PCI function.
            if bdf.startswith("0000:"):
                net = os.path.join(self.sys, "class", "net", netdev)
                os.makedirs(net, exist_ok=True)
                os.symlink(os.path.relpath(pci, net), os.path.join(net, "device"))

    def _gid(self, dev: str, index: int, gid: str, kind: str, ndev: str) -> None:
        port = os.path.join(self.sys, "class", "infiniband", dev, "ports", "1")
        for sub, value in (("gids", gid), (os.path.join("gid_attrs", "types"), kind),
                           (os.path.join("gid_attrs", "ndevs"), ndev)):
            path = os.path.join(port, sub)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, str(index)), "w") as f:
                f.write(value + "\n")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.root)


ENV_KEYS = ("MENTAT_FABRIC_LAYOUT", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
            "NCCL_ALGO", "NCCL_GRAPH_FILE", "NCCL_MAX_NCHANNELS",
            "VLLM_ARX_RING", "ARX_RING_PREV_HCAS", "ARX_RING_NEXT_HCAS")


class RingTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        self.fake = FakeSys({})
        os.environ["MENTAT_FABRIC_LAYOUT"] = "ring"
        os.environ["MENTAT_FABRIC_PREV_IFACE"] = A_IFACE
        os.environ["MENTAT_FABRIC_NEXT_IFACE"] = B_IFACE
        os.environ.pop("NCCL_MAX_NCHANNELS", None)
        os.environ.pop("NCCL_IB_GID_INDEX", None)
        os.environ.pop("FABRIC_RING_ARX", None)
        fabric_ring.SYS_ROOT = self.fake.sys

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.fake.cleanup()

    # -- helpers ----------------------------------------------------------

    def addr(self, second_root: bool = True):
        """Addressed dict: first-root devices 10.99.x.1, second 10.99.x.2."""
        out = {A_DEVS[0]: (10, 99, 3, 1), B_DEVS[0]: (10, 99, 2, 2)}
        if second_root:
            out[A_DEVS[1]] = (10, 99, 13, 1)
            out[B_DEVS[1]] = (10, 99, 22, 1)
        return out

    # -- two devices per port: the old RING4 behaviour, unchanged ---------

    def test_two_devices_per_port_unchanged(self):
        fake = FakeSys(self.addr())
        try:
            fabric_ring.SYS_ROOT = fake.sys
            os.environ["NCCL_IB_GID_INDEX"] = "5"
            fabric_ring._setup()
            self.assertEqual(
                os.environ["NCCL_IB_HCA"],
                "=" + ",".join(UNION_ORDER),
                "first-root devices then second-root, sorted by PCI address",
            )
            self.assertNotIn("NCCL_IB_GID_INDEX", os.environ)
            self.assertEqual(os.environ["NCCL_ALGO"], "Ring")
            graph = ET.parse(os.environ["NCCL_GRAPH_FILE"]).getroot()
            self.assertEqual(graph.tag, "graphs")
            g = graph.find("graph")
            self.assertEqual(g.get("nchannels"), "8")
            chans = g.findall("channel")
            self.assertEqual(len(chans), 8)
            prev = [UNION_ORDER.index(d) for d in A_DEVS]
            nxt = [UNION_ORDER.index(d) for d in B_DEVS]
            for c, ch in enumerate(chans):
                net = [int(d.get("dev")) for d in ch.findall("net")]
                self.assertEqual(net, [prev[c % 2], nxt[c % 2]])
                self.assertEqual([int(x.get("dev")) for x in ch.findall("gpu")], [0])
            # arx on by default, both devices per side, roots in order
            self.assertEqual(os.environ["VLLM_ARX_RING"], "1")
            self.assertEqual(os.environ["ARX_RING_PREV_HCAS"], ",".join(A_DEVS))
            self.assertEqual(os.environ["ARX_RING_NEXT_HCAS"], ",".join(B_DEVS))
        finally:
            fake.cleanup()

    def test_one_device_per_port_second_root_unaddressed(self):
        fake = FakeSys(self.addr(second_root=False))
        try:
            fabric_ring.SYS_ROOT = fake.sys
            fabric_ring._setup()
            self.assertEqual(fabric_ring._port_devices(A_IFACE), [A_DEVS[0]])
            self.assertEqual(
                os.environ["NCCL_IB_HCA"],
                "=" + A_DEVS[0] + "," + B_DEVS[0],
            )
            graph = ET.parse(os.environ["NCCL_GRAPH_FILE"]).getroot()
            chans = graph.find("graph").findall("channel")
            self.assertEqual(len(chans), 8)
            for ch in chans:
                net = [int(d.get("dev")) for d in ch.findall("net")]
                self.assertEqual(net, [0, 1])
        finally:
            fake.cleanup()

    def test_zero_devices_raises(self):
        for iface in (A_IFACE, B_IFACE):
            with self.subTest(iface=iface):
                self.assertEqual(fabric_ring._port_devices(iface), [])

    def test_arx_gated_off(self):
        fake = FakeSys(self.addr())
        try:
            fabric_ring.SYS_ROOT = fake.sys
            os.environ["FABRIC_RING_ARX"] = "0"
            fabric_ring._setup()
            for k in ("VLLM_ARX_RING", "ARX_RING_PREV_HCAS", "ARX_RING_NEXT_HCAS"):
                self.assertNotIn(k, os.environ)
            self.assertEqual(os.environ["NCCL_ALGO"], "Ring")
        finally:
            fake.cleanup()

    def test_zero_devices_setup_error_names_iface(self):
        try:
            fabric_ring._setup()
        except RuntimeError as e:
            self.assertIn(A_IFACE, str(e))
        else:
            self.fail("expected RuntimeError for a port with no usable device")


if __name__ == "__main__":
    unittest.main()