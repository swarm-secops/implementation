#!/usr/bin/env python3
"""
hcas_leader.py — Leader UAV   (run on the laptop / leader Raspberry Pi)
==========================================================================

Two independent subcommands, each a separate process invocation, matching
the workflow:

    python hcas_leader.py enroll --gcs-host <gcs-ip> --gcs-port 9000
        Connects to the GCS while it is online, receives leader_provision,
        and saves it to disk (default: leader_provision.pkl). Then exits.
        The GCS may now be stopped.

    python hcas_leader.py run --listen-host 0.0.0.0 --listen-port 9100
        Loads leader_provision.pkl from disk (no GCS connection made or
        needed). Opens a TCP listener for the follower and runs the
        interactive command loop, generating Msg_q per Eq. 9-13 for every
        command the operator types.

Enrollment (subcommand: enroll)
----------------------------------
  Receives from the GCS: seed_l (s_L^M), K_E, M, ID_L, R, C_L, cl_levels,
  tokens, FORS_K, FORS_A, N. K_E is given directly because the leader,
  not the GCS, performs AEAD.Enc_{K_E} at runtime (Eq. 11). The leader
  never receives secret_f or W_F -- follower key distribution is a
  GCS-only responsibility (see hcas_gcs.py).

Command generation (subcommand: run; Eq. 9-13)
--------------------------------------------------
  AD_q  = ID_L || q || M || R                                       (Eq. 9)
  N_q   = Trunc_96(H(AD_q || "nonce"))                              (Eq. 10)
  X_q   = AEAD.Enc_{K_E}(N_q, cmd_q, AD_q)                          (Eq. 11)
  d_q   = H(AD_q || N_q || X_q)                                     (Eq. 12)
  sigma_q = FORSSign(s_L^M, q, d_q)
  Omega_q = MerkleProof(p_{L,q}, C_L)
  Msg_q   = (ID_L, q, N_q, X_q, sigma_q, p_{L,q}, Omega_q)          (Eq. 13)

Dependencies
------------
    pip install cryptography
"""

import argparse
import pickle
import socket
import struct
import time

from hcas_crypto import (
    H, trunc, NONCE_LEN,
    FORS, merkle_proof,
    aead_enc,
    serialise_msg,
)

ID_L = b"UAV_LEADER_001"

ROLE_LEADER = 0x01

DEFAULT_PROVISION_FILE = "leader_provision.pkl"


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
# Subcommand: enroll
# ---------------------------------------------------------------------------

def cmd_enroll(args: argparse.Namespace) -> None:
    """
    Connect to the GCS enrollment server, announce role = leader, send
    ID_L, receive the leader_provision blob (Eq. 1-5), and persist it to
    disk. The process then exits; no socket to the GCS remains open.
    """
    print(f"[Leader] Connecting to GCS at {args.gcs_host}:{args.gcs_port} for enrollment ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.gcs_host, args.gcs_port))

    sock.sendall(bytes([ROLE_LEADER]))
    _send_framed(sock, ID_L)

    provision = pickle.loads(_recv_framed(sock))
    sock.close()

    print(f"[Leader] Enrollment complete. R = {provision['R'].hex()}")
    print(f"[Leader] C_L = {provision['C_L'].hex()}")
    print(f"[Leader] Command budget N = {provision['N']}")

    provision["q_next"] = 0     # next command index to issue at runtime

    with open(args.out, "wb") as f:
        pickle.dump(provision, f)
    print(f"\n[Leader] Provisioning saved to '{args.out}'.")
    print("[Leader] You may now stop the GCS. Run 'hcas_leader.py run' when ready.")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def generate_command(state: dict, cmd_str: str) -> bytes:
    """
    Generate and serialise Msg_q for command string cmd_str (Eq. 9-13).
    """
    q = state["q_next"]
    N = state["N"]
    if q >= N:
        raise RuntimeError(f"Command budget exhausted: all {N} tokens have been used.")

    cmd_bytes = cmd_str.encode("utf-8")
    M    = state["M"]
    R    = state["R"]
    KE   = state["KE"]
    fors = state["fors"]

    # Eq. 9
    AD_q = ID_L + q.to_bytes(4, "big") + M + R

    # Eq. 10
    N_q = trunc(H(AD_q, b"nonce"), NONCE_LEN)

    # Eq. 11 — leader performs encryption; this is exactly why the
    # leader, not the follower, needed K_E directly from the GCS.
    X_q = aead_enc(KE, N_q, cmd_bytes, AD_q)

    # Eq. 12
    d_q = H(AD_q, N_q, X_q)

    # FORS one-time opening sigma_q
    t_sign = time.perf_counter()
    sigma  = fors.sign(state["seed_l"], M, ID_L, q, d_q)
    t_sign = (time.perf_counter() - t_sign) * 1e6

    # Merkle proof Omega_q : p_{L,q} in C_L
    p_Lq    = state["tokens"][q]
    omega_q = merkle_proof(state["cl_levels"], q)

    state["q_next"] += 1

    msg = {
        "id_l":         ID_L,
        "q":            q,
        "nonce":        N_q,
        "ciphertext":   X_q,
        "sigma":        sigma,
        "token":        p_Lq,
        "merkle_proof": omega_q,
    }

    payload = serialise_msg(msg, state["FORS_K"], state["FORS_A"])
    print(f"[Leader] Cmd #{q} '{cmd_str}'  "
          f"FORS-sign {t_sign:.0f} us  payload {len(payload)} B")
    return payload


def cmd_run(args: argparse.Namespace) -> None:
    """
    Load leader_provision.pkl from disk (no GCS connection made) and run
    the interactive runtime command loop against the follower.
    """
    with open(args.state, "rb") as f:
        state = pickle.load(f)

    state["fors"] = FORS(state["FORS_K"], state["FORS_A"])

    print(f"[Leader] Loaded provisioning from '{args.state}'. "
          f"q_next={state['q_next']}, N={state['N']}")
    print(f"[Leader] R   = {state['R'].hex()}")
    print(f"[Leader] C_L = {state['C_L'].hex()}\n")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.listen_host, args.listen_port))
    srv.listen(1)
    print(f"[Leader] Listening for follower on {args.listen_host}:{args.listen_port} ...")

    conn, addr = srv.accept()
    print(f"[Leader] Follower connected from {addr}\n")

    print("Enter a command string and press Enter. Type 'quit' to exit.\n")
    try:
        while True:
            cmd = input("CMD> ").strip()
            if not cmd:
                continue
            if cmd.lower() == "quit":
                break

            payload = generate_command(state, cmd)
            _send_framed(conn, payload)

            ack = _recv_framed(conn)
            print(f"[Leader] Follower ACK -> {ack.decode()}\n")

    except (KeyboardInterrupt, EOFError):
        print("\n[Leader] Interrupted.")
    except ConnectionError as e:
        print(f"\n[Leader] Connection error: {e}")
    finally:
        conn.close()
        srv.close()
        # Persist q_next so a restart doesn't reuse already-spent tokens.
        with open(args.state, "wb") as f:
            pickle.dump(state, f)
        print(f"[Leader] State saved to '{args.state}' (q_next={state['q_next']}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HCAS-CT Leader UAV")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_enroll = sub.add_parser("enroll", help="enroll with the GCS and save provisioning to disk")
    p_enroll.add_argument("--gcs-host", required=True, help="GCS enrollment server address")
    p_enroll.add_argument("--gcs-port", type=int, default=9000, help="GCS enrollment server port")
    p_enroll.add_argument("--out", default=DEFAULT_PROVISION_FILE,
                          help=f"output file for provisioning (default: {DEFAULT_PROVISION_FILE})")
    p_enroll.set_defaults(func=cmd_enroll)

    p_run = sub.add_parser("run", help="load provisioning from disk and serve the follower at runtime")
    p_run.add_argument("--state", default=DEFAULT_PROVISION_FILE,
                       help=f"provisioning file to load (default: {DEFAULT_PROVISION_FILE})")
    p_run.add_argument("--listen-host", default="0.0.0.0", help="address to serve the follower on")
    p_run.add_argument("--listen-port", type=int, default=9100, help="port to serve the follower on")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
