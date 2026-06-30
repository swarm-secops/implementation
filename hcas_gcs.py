#!/usr/bin/env python3
"""
hcas_gcs.py — GCS enrollment server   (run ONCE, on the LAPTOP, before mission start)
=======================================================================================

This script is the standalone GCS process. It is started, enrolls exactly
one leader and N_FOLLOWERS followers over TCP, then exits permanently --
matching the workflow:

    1. Start hcas_gcs.py on the laptop.            <- this script
    2. Run `hcas_leader.py enroll` on the laptop.
    3. Run `hcas_follower.py enroll` on the Raspberry Pi.
    4. Stop hcas_gcs.py. The GCS is never needed again.
    5. Run `hcas_leader.py run` on the laptop.
    6. Run `hcas_follower.py run` on the Raspberry Pi.
       Leader and follower now talk to each other directly.

Implements the GCS side of enrollment (paper Section 2, "Enrollment").
The GCS is the sole trusted party during this phase and the sole party
that ever computes W_F (Eq. 6-7). Key material flows asymmetrically,
matching the paper exactly:
  - The LEADER receives K_E directly from the GCS, in the clear, because
    Eq. 11 requires the leader to run AEAD.Enc_{K_E} at runtime and the
    leader has no other way to obtain that key.
  - Each FOLLOWER receives only W_F (the wrapped key) plus its own
    secret_f, and derives K_E locally via Eq. 8. The follower never
    receives K_E directly, and the leader never sees secret_f or W_F.

Wire protocol (enrollment phase only — never reused at runtime)
-----------------------------------------------------------------
  Leader connection:
    Leader -> GCS : ROLE byte 0x01, then ID_L  (length-prefixed)
    GCS    -> Leader : leader_provision blob (pickled dict, length-prefixed)
                       Contains: seed_l, K_E, M, ID_L, R, C_L, cl_levels,
                       tokens, FORS_K, FORS_A, N.

  Follower connection:
    Follower -> GCS : ROLE byte 0x02, then ID_F (length-prefixed)
    GCS      -> Follower : follower_provision blob (pickled dict, length-prefixed)
                       Contains: secret_f, M, R, ID_L, C_L, N, W_F,
                       FORS_K, FORS_A.

Enrollment computation (Eq. 1-8)
----------------------------------
  1. GCS fixes mission parameters: M, N, k, a.
  2. GCS generates the leader's mission seed s_L^M.
  3. For each command slot q in 0..N-1:
       p_{L,q} = H(r_{q,1} || ... || r_{q,k})                       (Eq. 2)
       where r_{q,j} = MRoot of FORS tree j for slot q, built from
       x_{q,j,v} = H(s_L^M || M || ID_L || q || j || v || "ctok")    (Eq. 1)
  4. C_L = MRoot(p_{L,0}, ..., p_{L,N-1})                            (Eq. 3)
  5. l_L = H(ID_L || "leader" || C_L || M)                          (Eq. 4)
  6. For each follower F_i, GCS generates a fresh per-UAV secret s_i and
     computes l_{F_i} (see NOTE below).
  7. R = MRoot(l_L, l_{F_1}, ..., l_{F_n})                          (Eq. 5)
  8. For each follower F_i:
       Z_i = H(s_i || M || R || ID_i || "wrap") -> Trunc_{|K_E|}    (Eq. 6)
       W_i = K_E XOR Z_i                                             (Eq. 7)
     Follower mission record: (M, R, ID_L, C_L, N, W_i)             (Eq. 8 target)

NOTE on l_F: the paper gives an explicit formula for l_L (Eq. 4) but not
for a follower's leaf l_F. This implementation uses
    l_F = H(ID_F || "follower" || M)
the natural counterpart of Eq. 4 with the command-root term omitted
(followers issue no commands, so they have no C_L of their own). Adjust
this function if your deployment defines l_F differently.

Usage
-----
    python hcas_gcs.py [--host 0.0.0.0] [--port 9000] [--n-followers 1]

Dependencies
------------
    pip install cryptography
"""

import argparse
import os
import pickle
import socket
import struct
import time

from hcas_crypto import (
    H, KE_LEN,
    FORS, merkle_build,
    derive_wrap_mask, wrap_key,
)

# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------

MISSION_ID = b"MISSION_ALPHA_001"   # M

FORS_K     = 6      # k: number of FORS trees
FORS_A     = 8      # a: tree height  ->  2^8 = 256 leaves per tree
N_COMMANDS = 1024   # N: command budget  (must be a power of 2 for MRoot)

ROLE_LEADER   = 0x01
ROLE_FOLLOWER = 0x02


# ---------------------------------------------------------------------------
# Wire framing  (length-prefixed TCP)
# ---------------------------------------------------------------------------

def _send_framed(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        buf += chunk
    return buf


def _recv_framed(sock: socket.socket) -> bytes:
    length = struct.unpack(">I", _recv_exact(sock, 4))[0]
    return _recv_exact(sock, length)


# ---------------------------------------------------------------------------
# Enrollment computation
# ---------------------------------------------------------------------------

def compute_leader_tokens(fors: FORS, seed_l: bytes, M: bytes, id_l: bytes):
    """Step 3-4: compute all N command tokens and the resulting command root."""
    print(f"[GCS] Computing {N_COMMANDS} command tokens for leader "
          f"'{id_l.decode()}' (k={FORS_K}, a={FORS_A}) ...")
    t0 = time.perf_counter()
    tokens = []
    for q in range(N_COMMANDS):
        tokens.append(fors.compute_token(seed_l, M, id_l, q))     # Eq. 1-2
        if (q + 1) % 128 == 0:
            print(f"[GCS]   {q + 1}/{N_COMMANDS} tokens ...", end="\r")
    print()
    cl_levels, C_L = merkle_build(tokens)                          # Eq. 3
    print(f"[GCS] C_L computed in {(time.perf_counter() - t0) * 1000:.1f} ms")
    return tokens, cl_levels, C_L


def leader_leaf(id_l: bytes, C_L: bytes, M: bytes) -> bytes:
    """Eq. 4."""
    return H(id_l, b"leader", C_L, M)


def follower_leaf(id_f: bytes, M: bytes) -> bytes:
    """Counterpart of Eq. 4 for a follower -- see module-level NOTE."""
    return H(id_f, b"follower", M)


def wrap_for_follower(secret_f: bytes, M: bytes, R: bytes, id_f: bytes, KE: bytes) -> bytes:
    """Eq. 6-7. GCS-only operation: only the GCS holds K_E in the clear
    alongside secret_f, so only the GCS can ever compute W_F."""
    Z_i = derive_wrap_mask(secret_f, M, R, id_f)
    return wrap_key(KE, Z_i)


# ---------------------------------------------------------------------------
# Main enrollment server
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HCAS-CT GCS enrollment server")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind address")
    parser.add_argument("--port", type=int, default=9000, help="TCP port")
    parser.add_argument("--n-followers", type=int, default=1,
                        help="number of follower UAVs to enroll before computing R")
    args = parser.parse_args()

    fors = FORS(FORS_K, FORS_A)
    M    = MISSION_ID

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(args.n_followers + 1)
    print(f"[GCS] Enrollment server listening on {args.host}:{args.port}")
    print(f"[GCS] Waiting for 1 leader and {args.n_followers} follower(s) to connect ...")
    print(f"[GCS]   Run 'hcas_leader.py enroll' on the leader machine, and")
    print(f"[GCS]   'hcas_follower.py enroll' on each follower machine, now.\n")

    leader_conn = None
    leader_id   = None
    follower_conns: list[tuple[socket.socket, bytes]] = []

    while leader_conn is None or len(follower_conns) < args.n_followers:
        conn, addr = srv.accept()
        role = _recv_exact(conn, 1)[0]
        id_bytes = _recv_framed(conn)

        if role == ROLE_LEADER:
            if leader_conn is not None:
                print(f"[GCS] Rejecting extra leader connection from {addr}")
                conn.close()
                continue
            leader_conn, leader_id = conn, id_bytes
            print(f"[GCS] Leader '{id_bytes.decode()}' connected from {addr}")
        elif role == ROLE_FOLLOWER:
            if len(follower_conns) >= args.n_followers:
                print(f"[GCS] Rejecting extra follower connection from {addr}")
                conn.close()
                continue
            follower_conns.append((conn, id_bytes))
            print(f"[GCS] Follower '{id_bytes.decode()}' connected from {addr}")
        else:
            print(f"[GCS] Unknown role byte {role:#x} from {addr}, dropping")
            conn.close()

    print()

    # ------------------------------------------------------------------
    # Enrollment computation  (Eq. 1-8) — GCS-only, never delegated
    # ------------------------------------------------------------------
    seed_l = os.urandom(32)                                        # s_L^M
    tokens, cl_levels, C_L = compute_leader_tokens(fors, seed_l, M, leader_id)

    leaf_L = leader_leaf(leader_id, C_L, M)                         # Eq. 4

    follower_secrets: dict[bytes, bytes] = {}
    follower_leaves  = [leaf_L]
    for _conn, fid in follower_conns:
        s_f = os.urandom(32)
        follower_secrets[fid] = s_f
        follower_leaves.append(follower_leaf(fid, M))

    _root_levels, R = merkle_build(follower_leaves)                 # Eq. 5

    KE = os.urandom(KE_LEN)                                         # mission key
                                                                      # Held only by GCS at
                                                                      # this point.

    print(f"[GCS] R   = {R.hex()}")
    print(f"[GCS] C_L = {C_L.hex()}\n")

    # ------------------------------------------------------------------
    # Distribute provisioning to leader -- includes K_E in the clear
    # (needed for Eq. 11 encryption), but NOT secret_f or W_F.
    # ------------------------------------------------------------------
    leader_provision = {
        "seed_l":    seed_l,
        "KE":        KE,
        "M":         M,
        "ID_L":      leader_id,
        "R":         R,
        "C_L":       C_L,
        "cl_levels": cl_levels,
        "tokens":    tokens,
        "FORS_K":    FORS_K,
        "FORS_A":    FORS_A,
        "N":         N_COMMANDS,
    }
    _send_framed(leader_conn, pickle.dumps(leader_provision))
    print(f"[GCS] Leader provisioning sent to '{leader_id.decode()}' (includes K_E, no follower secrets)")
    leader_conn.close()

    # ------------------------------------------------------------------
    # Distribute provisioning to each follower  (Eq. 6-8) -- GCS sends
    # W_F directly. The leader is never involved in this exchange.
    # ------------------------------------------------------------------
    for conn, fid in follower_conns:
        s_f = follower_secrets[fid]
        W_f = wrap_for_follower(s_f, M, R, fid, KE)                  # Eq. 6-7

        follower_provision = {
            "secret_f": s_f,
            "M":        M,
            "R":        R,
            "ID_L":     leader_id,
            "C_L":      C_L,
            "N":        N_COMMANDS,
            "W_F":      W_f,
            "ID_F":     fid,
            "FORS_K":   FORS_K,
            "FORS_A":   FORS_A,
        }
        _send_framed(conn, pickle.dumps(follower_provision))
        print(f"[GCS] Follower provisioning sent to '{fid.decode()}' (includes W_F, sent by GCS)")
        conn.close()

    srv.close()
    print("\n[GCS] Enrollment complete for all parties. GCS is now offline.")
    print("[GCS] You may now stop this process and start leader/follower in 'run' mode.")


if __name__ == "__main__":
    main()
