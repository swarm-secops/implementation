"""
hcas_crypto.py  —  HCAS-CT shared cryptographic primitives
===========================================================

Implements every primitive referenced in the paper exactly as specified.

Hash primitive  H(·)
    SHA-256, output 32 bytes.  Used for all Merkle nodes, FORS leaves,
    digest computation, and key derivation.  Inputs are concatenated
    before hashing (single SHA-256 call per invocation).

AEAD
    AES-256-GCM.  Key K_E is 32 bytes (256 bits).  Nonce is 12 bytes
    (96 bits) as required by GCM and consistent with Trunc_96 in Eq. 10.
    AES-256 is chosen because SHA-256 provides 128-bit post-quantum
    collision resistance; AES-128 would reduce the symmetric security
    level to 64 bits post-quantum, which is inconsistent.

FORS (Forest of Random Subsets)
    k trees, each of height a, each with 2^a leaves.
    Digest-to-index mapping: consecutive a-bit windows extracted from the
    digest bytes (big-endian bit order), identical to the SPHINCS+/FIPS-205
    base-w encoding.  This is the only unambiguous way to extract k
    independent a-bit indices from a hash output; ad-hoc bit-shifting
    would be non-standard and hard to audit.

    The one-time opening sigma_q consists ONLY of:
      - k secret leaf preimages  x_{q,j,v_j}
      - k Merkle authentication paths  (each of length a)
    The verifier independently:
      (a) recomputes indices from d_q (never trusts signer's indices)
      (b) recomputes pk_{q,j,v_j} = H(x_{q,j,v_j})
      (c) walks the auth path to reconstruct r_{q,j}
      (d) reconstructs p_{L,q} = H(r_{q,1} || ... || r_{q,k})
      (e) checks reconstructed p_{L,q} == claimed p_{L,q}
    Sending the roots or indices from the signer side would let a
    compromised leader forge accepted openings, so they are omitted.

Merkle tree
    Complete binary tree over 2^ceil(log2(n)) leaves.
    Padding leaves are H(b'\x00' * 32) — a fixed, domain-separated
    constant — so padding does not create collisions with real leaves.
    Interior node: H(left_child || right_child).

Key wrapping  (Eq. 6–8)
    Z_i = H(s_i || M || R || ID_i || b"wrap")  then Trunc_{|K_E|}
    W_i = K_E XOR Trunc_{|K_E|}(Z_i)
    Unwrap: K_E = W_i XOR Trunc_{|K_E|}(Z_i)
"""

import hashlib
import hmac
import math
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASH_LEN  = 32        # SHA-256 output bytes
KE_LEN    = 32        # AES-256 key bytes  (post-quantum consistent with SHA-256)
NONCE_LEN = 12        # AES-GCM nonce bytes  (Trunc_96 per Eq. 10)

# Fixed padding leaf used when the Merkle tree size is rounded up to a power
# of two.  Using H(0^32) ensures this value cannot collide with any real
# leaf derived from H(secret_preimage), since the preimage domain is disjoint.
_PADDING_LEAF = hashlib.sha256(b'\x00' * HASH_LEN).digest()


# ---------------------------------------------------------------------------
# Hash primitive
# ---------------------------------------------------------------------------

def H(*parts: bytes) -> bytes:
    """
    SHA-256 over the concatenation of all byte arguments.
    Single call — no incremental state is exposed.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def trunc(data: bytes, n: int) -> bytes:
    """Return the first n bytes of data."""
    assert len(data) >= n, f"trunc: need {n} bytes, got {len(data)}"
    return data[:n]


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------

def _ceil_log2(n: int) -> int:
    """Ceiling of log2(n) for n >= 1."""
    return (n - 1).bit_length()


def merkle_build(leaves: list[bytes]) -> tuple[list[list[bytes]], bytes]:
    """
    Build a complete binary Merkle tree.

    The leaf list is padded to the next power of two using _PADDING_LEAF.
    Each interior node is H(left || right).

    Parameters
    ----------
    leaves : list of 32-byte leaf hashes.

    Returns
    -------
    levels : levels[0] = padded leaf level,  levels[-1] = [root]
    root   : 32-byte root hash
    """
    assert leaves, "merkle_build: leaf list must not be empty"

    # Pad to power of two
    depth = _ceil_log2(max(len(leaves), 1))
    size  = 1 << depth
    padded = list(leaves) + [_PADDING_LEAF] * (size - len(leaves))

    levels = [padded]
    current = padded
    while len(current) > 1:
        next_level = [H(current[i], current[i + 1])
                      for i in range(0, len(current), 2)]
        levels.append(next_level)
        current = next_level

    return levels, current[0]


def merkle_proof(levels: list[list[bytes]], idx: int) -> list[bytes]:
    """
    Return the authentication path (list of sibling hashes, bottom to top)
    for the leaf at position idx.
    """
    path = []
    for level in levels[:-1]:          # every level except the root
        sibling_idx = idx ^ 1
        path.append(level[sibling_idx])
        idx >>= 1
    return path


def merkle_verify(leaf: bytes, path: list[bytes], root: bytes, idx: int) -> bool:
    """
    Verify that `leaf` at position `idx` is committed under `root`
    via the authentication path `path`.

    Uses hmac.compare_digest for constant-time final comparison.
    """
    current = leaf
    for sibling in path:
        if idx & 1:
            current = H(sibling, current)
        else:
            current = H(current, sibling)
        idx >>= 1
    return hmac.compare_digest(current, root)


# ---------------------------------------------------------------------------
# FORS — Forest of Random Subsets
# ---------------------------------------------------------------------------

def _encode_u32(v: int) -> bytes:
    return v.to_bytes(4, 'big')


def _fors_digest_indices(digest: bytes, k: int, a: int) -> list[int]:
    """
    Extract k leaf indices, each of a bits, from digest.

    Method: treat the digest as a big-endian bit-string and extract
    consecutive a-bit windows starting from the most-significant end.
    This is identical to FIPS 205 base_2^a encoding.

    Requires  k * a <= len(digest) * 8  bits.
    For k=6, a=8: 48 bits needed, 256 available — fine.
    """
    assert k * a <= len(digest) * 8, (
        f"FORS parameters k={k}, a={a} require {k*a} bits "
        f"but digest is only {len(digest)*8} bits"
    )
    bits = int.from_bytes(digest, 'big')
    total_bits = len(digest) * 8
    mask = (1 << a) - 1
    indices = []
    for j in range(k):
        # Extract window starting at bit position j*a from the MSB end
        shift = total_bits - (j + 1) * a
        indices.append((bits >> shift) & mask)
    return indices


class FORS:
    """
    FORS (Forest of Random Subsets) as used in HCAS-CT.

    Parameters
    ----------
    k : number of trees
    a : height of each tree  (each tree has 2^a leaves)
    """

    def __init__(self, k: int, a: int):
        self.k = k
        self.a = a
        self.leaves_per_tree = 1 << a

    # ------------------------------------------------------------------
    # Internal: secret leaf derivation  (Eq. 1)
    # x_{q,j,v} = H(s_L^M || M || ID_L || q || j || v || "ctok")
    # ------------------------------------------------------------------

    def _secret_leaf(
        self,
        seed_l: bytes,
        mission: bytes,
        id_l: bytes,
        q: int,
        j: int,
        v: int,
    ) -> bytes:
        return H(seed_l, mission, id_l,
                 _encode_u32(q), _encode_u32(j), _encode_u32(v),
                 b"ctok")

    # ------------------------------------------------------------------
    # Internal: build public key leaves for tree (q, j)
    # pk_{q,j,v} = H(x_{q,j,v})
    # ------------------------------------------------------------------

    def _pk_leaves(
        self,
        seed_l: bytes,
        mission: bytes,
        id_l: bytes,
        q: int,
        j: int,
    ) -> list[bytes]:
        return [
            H(self._secret_leaf(seed_l, mission, id_l, q, j, v))
            for v in range(self.leaves_per_tree)
        ]

    # ------------------------------------------------------------------
    # Compute public command token p_{L,q}  (Eq. 2)
    # p_{L,q} = H(r_{q,1} || ... || r_{q,k})
    # ------------------------------------------------------------------

    def compute_token(
        self,
        seed_l: bytes,
        mission: bytes,
        id_l: bytes,
        q: int,
    ) -> bytes:
        roots = []
        for j in range(self.k):
            pk = self._pk_leaves(seed_l, mission, id_l, q, j)
            _, root = merkle_build(pk)
            roots.append(root)
        return H(*roots)

    # ------------------------------------------------------------------
    # Sign: produce one-time opening sigma_q
    #
    # sigma_q contains ONLY:
    #   leaves[j]  = x_{q,j,v_j}   (secret preimage, 32 bytes each)
    #   paths[j]   = Merkle auth-path for leaf v_j in tree j
    #
    # Roots and indices are NOT included — the verifier recomputes them
    # independently from d_q and the leaves/paths.
    # ------------------------------------------------------------------

    def sign(
        self,
        seed_l: bytes,
        mission: bytes,
        id_l: bytes,
        q: int,
        digest: bytes,
    ) -> dict:
        """
        Produce the FORS one-time opening sigma_q.

        Returns
        -------
        dict with keys:
          'leaves' : list[bytes]        k secret leaf preimages
          'paths'  : list[list[bytes]]  k Merkle auth-paths (len a each)
        """
        indices = _fors_digest_indices(digest, self.k, self.a)
        leaves_out = []
        paths_out  = []

        for j in range(self.k):
            v  = indices[j]
            pk = self._pk_leaves(seed_l, mission, id_l, q, j)
            levels, _ = merkle_build(pk)
            leaves_out.append(self._secret_leaf(seed_l, mission, id_l, q, j, v))
            paths_out.append(merkle_proof(levels, v))

        return {"leaves": leaves_out, "paths": paths_out}

    # ------------------------------------------------------------------
    # Verify: FORSVerify(p_{L,q}, d_q, sigma_q) = 1   (Eq. 15)
    #
    # The verifier:
    #   1. recomputes indices v_j from d_q independently
    #   2. recomputes pk_{q,j,v_j} = H(x_{q,j,v_j})
    #   3. walks the auth path to reconstruct r_{q,j}
    #   4. reconstructs candidate token = H(r_{q,1}||...||r_{q,k})
    #   5. checks candidate == p_{L,q}  (constant-time)
    # ------------------------------------------------------------------

    def verify(
        self,
        token: bytes,
        digest: bytes,
        sigma: dict,
    ) -> bool:
        indices = _fors_digest_indices(digest, self.k, self.a)

        reconstructed_roots = []
        for j in range(self.k):
            v          = indices[j]
            pk_leaf    = H(sigma["leaves"][j])           # pk_{q,j,v_j}
            path       = sigma["paths"][j]

            # Walk auth path from this leaf to reconstruct tree root r_{q,j}
            current = pk_leaf
            pos     = v
            for sibling in path:
                if pos & 1:
                    current = H(sibling, current)
                else:
                    current = H(current, sibling)
                pos >>= 1
            reconstructed_roots.append(current)

        candidate_token = H(*reconstructed_roots)
        return hmac.compare_digest(candidate_token, token)


# ---------------------------------------------------------------------------
# AES-256-GCM AEAD
# ---------------------------------------------------------------------------

def aead_enc(ke: bytes, nonce: bytes, plaintext: bytes, ad: bytes) -> bytes:
    """
    AES-256-GCM encryption.
    Returns ciphertext concatenated with 16-byte authentication tag.
    """
    assert len(ke) == KE_LEN,    f"K_E must be {KE_LEN} bytes, got {len(ke)}"
    assert len(nonce) == NONCE_LEN, f"nonce must be {NONCE_LEN} bytes"
    return AESGCM(ke).encrypt(nonce, plaintext, ad)


def aead_dec(ke: bytes, nonce: bytes, ciphertext: bytes, ad: bytes) -> bytes:
    """
    AES-256-GCM decryption and authentication.
    Raises cryptography.exceptions.InvalidTag on any authentication failure.
    """
    assert len(ke) == KE_LEN,    f"K_E must be {KE_LEN} bytes, got {len(ke)}"
    assert len(nonce) == NONCE_LEN, f"nonce must be {NONCE_LEN} bytes"
    return AESGCM(ke).decrypt(nonce, ciphertext, ad)


# ---------------------------------------------------------------------------
# Key wrapping  (Eq. 6–8)
# ---------------------------------------------------------------------------

def derive_wrap_mask(s_i: bytes, mission: bytes, R: bytes, id_i: bytes) -> bytes:
    """
    Z_i = H(s_i || M || R || ID_i || "wrap"),  then Trunc_{|K_E|}
    (Eq. 6)
    """
    return trunc(H(s_i, mission, R, id_i, b"wrap"), KE_LEN)


def wrap_key(ke: bytes, mask: bytes) -> bytes:
    """W_i = K_E XOR Trunc_{|K_E|}(Z_i)  (Eq. 7)"""
    assert len(ke) == KE_LEN and len(mask) == KE_LEN
    return bytes(a ^ b for a, b in zip(ke, mask))


def unwrap_key(wi: bytes, mask: bytes) -> bytes:
    """K_E = W_i XOR Trunc_{|K_E|}(Z_i)  (Eq. 8)"""
    assert len(wi) == KE_LEN and len(mask) == KE_LEN
    return bytes(a ^ b for a, b in zip(wi, mask))


# ---------------------------------------------------------------------------
# Wire serialisation
# ---------------------------------------------------------------------------
#
# All fields are fixed-size where possible (hash outputs, nonce, indices).
# Variable-size fields use a 2-byte big-endian length prefix.
# The format is self-describing so neither side needs out-of-band parameter
# agreement beyond the FORS parameters (k, a) which are fixed per mission.

def _pack_bytes(data: bytes) -> bytes:
    return struct.pack(">H", len(data)) + data


def _unpack_bytes(buf: bytes, offset: int) -> tuple[bytes, int]:
    ln = struct.unpack_from(">H", buf, offset)[0]
    offset += 2
    return buf[offset: offset + ln], offset + ln


def serialise_sigma(sigma: dict, k: int, a: int) -> bytes:
    """
    Serialise FORS opening sigma_q.

    Layout:
      For each of k trees:
        secret leaf preimage  (HASH_LEN bytes, fixed)
        auth path             (a entries * HASH_LEN bytes, fixed)
    Total: k * (1 + a) * HASH_LEN bytes.  For k=6, a=8: 54 * 32 = 1728 bytes.
    """
    out = bytearray()
    for j in range(k):
        out += sigma["leaves"][j]               # 32 bytes
        assert len(sigma["paths"][j]) == a, (
            f"FORS tree {j} path length {len(sigma['paths'][j])} != a={a}")
        for node in sigma["paths"][j]:
            out += node                         # 32 bytes each
    return bytes(out)


def deserialise_sigma(buf: bytes, offset: int, k: int, a: int) -> tuple[dict, int]:
    """
    Deserialise FORS opening.  Returns (sigma_dict, new_offset).
    """
    leaves = []
    paths  = []
    for j in range(k):
        leaf = buf[offset: offset + HASH_LEN]; offset += HASH_LEN
        path = []
        for _ in range(a):
            node = buf[offset: offset + HASH_LEN]; offset += HASH_LEN
            path.append(node)
        leaves.append(leaf)
        paths.append(path)
    return {"leaves": leaves, "paths": paths}, offset


def serialise_msg(msg: dict, k: int, a: int) -> bytes:
    """
    Serialise Msg_q = (ID_L, q, N_q, X_q, sigma_q, p_{L,q}, Omega_q).

    Fixed-size fields (written directly):
      q          : 4 bytes (uint32 big-endian)
      N_q        : NONCE_LEN bytes (12)
      p_{L,q}    : HASH_LEN bytes (32)
      sigma_q    : k*(1+a)*HASH_LEN bytes (fixed for given k,a)
    Variable-size fields (2-byte length prefix):
      ID_L       : variable
      X_q        : variable (ciphertext + 16-byte GCM tag)
      Omega_q    : depth * HASH_LEN  (depth = log2(N), fixed for given N)
    """
    out = bytearray()

    # ID_L  (variable)
    out += _pack_bytes(msg["id_l"])

    # q  (4 bytes)
    out += struct.pack(">I", msg["q"])

    # N_q  (12 bytes, fixed)
    assert len(msg["nonce"]) == NONCE_LEN
    out += msg["nonce"]

    # X_q  (variable: len(cmd) + 16 tag bytes)
    out += _pack_bytes(msg["ciphertext"])

    # sigma_q  (fixed size for given k, a)
    out += serialise_sigma(msg["sigma"], k, a)

    # p_{L,q}  (32 bytes, fixed)
    assert len(msg["token"]) == HASH_LEN
    out += msg["token"]

    # Omega_q  (variable: each node is 32 bytes, depth nodes)
    depth = len(msg["merkle_proof"])
    out += struct.pack(">H", depth)
    for node in msg["merkle_proof"]:
        assert len(node) == HASH_LEN
        out += node

    return bytes(out)


def deserialise_msg(buf: bytes, k: int, a: int) -> dict:
    offset = 0

    # ID_L
    id_l, offset    = _unpack_bytes(buf, offset)

    # q
    q = struct.unpack_from(">I", buf, offset)[0]; offset += 4

    # N_q
    nonce = buf[offset: offset + NONCE_LEN]; offset += NONCE_LEN

    # X_q
    ciphertext, offset = _unpack_bytes(buf, offset)

    # sigma_q
    sigma, offset = deserialise_sigma(buf, offset, k, a)

    # p_{L,q}
    token = buf[offset: offset + HASH_LEN]; offset += HASH_LEN

    # Omega_q
    depth = struct.unpack_from(">H", buf, offset)[0]; offset += 2
    merkle_proof_nodes = []
    for _ in range(depth):
        merkle_proof_nodes.append(buf[offset: offset + HASH_LEN])
        offset += HASH_LEN

    return {
        "id_l":         id_l,
        "q":            q,
        "nonce":        nonce,
        "ciphertext":   ciphertext,
        "sigma":        sigma,
        "token":        token,
        "merkle_proof": merkle_proof_nodes,
    }
