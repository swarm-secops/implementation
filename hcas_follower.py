#!/usr/bin/env python3
"""
hcas_follower.py — Follower UAV   (run on the follower Raspberry Pi)
========================================================================

Two independent subcommands, each a separate process invocation, matching
the workflow:

    python hcas_follower.py enroll --gcs-host <gcs-ip> --gcs-port 9000
        Connects to the GCS while it is online, receives follower
        provisioning (secret_f, M, R, ID_L, C_L, N, W_F, FORS params),
        derives K_E locally (Eq. 8), and saves everything needed for
        runtime verification to disk (default: follower_state.pkl). Then
        exits. The GCS may now be stopped.

    python hcas_follower.py run --leader-host <leader-ip> --leader-port 9100
        Loads follower_state.pkl from disk (no GCS connection made or
        needed). Connects directly to the leader and verifies every
        Msg_q received per the "Follower verification" procedure
        (Eq. 14-16, plus freshness).

Enrollment (subcommand: enroll)
----------------------------------
  Receives from the GCS: secret_f, M, R, ID_L, C_L, N, W_F, FORS_K,
  FORS_A. K_E is never transmitted to the follower directly -- only W_F
  is, and only by the GCS. The follower derives:
      K_E = W_F XOR Trunc_{|K_E|}(H(secret_f || M || R || ID_F || "wrap"))   (Eq. 8)

Follower verification (subcommand: run; Eq. 14-16, plus freshness)
------------------------------------------------------------------------
  AD_q = ID_L || q || M || R
  d_q  = H(AD_q || N_q || X_q)                                     (Eq. 12, recomputed)
  freshness : q > q_last
  MVerify(p_{L,q}, Omega_q, C_L) == 1                               (Eq. 14)
  FORSVerify(p_{L,q}, d_q, sigma_q) == 1                            (Eq. 15)
  cmd_q = AEAD.Dec_{K_E}(N_q, X_q, AD_q)                            (Eq. 16)
  accept only if all checks pass; then q_last <- q

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
    H,
    FORS, merkle_verify,
    aead_dec,
    derive_wrap_mask, unwrap_key,
    deserialise_msg,
)

ID_F = b"UAV_FOLLOWER_001"

ROLE_FOLLOWER = 0x02

DEFAULT_STATE_FILE = "follower_state.pkl"


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
    Connect to the GCS enrollment server, announce role = follower, send
    ID_F, receive the follower_provision blob, derive K_E locally
    (Eq. 8), and persist runtime state to disk. The process then exits;
    no socket to the GCS remains open.
    """
    print(f"[Follower] Connecting to GCS at {args.gcs_host}:{args.gcs_port} for enrollment ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.gcs_host, args.gcs_port))

    sock.sendall(bytes([ROLE_FOLLOWER]))
    _send_framed(sock, ID_F)

    provision = pickle.loads(_recv_framed(sock))
    sock.close()

    # Eq. 8: K_E = W_F XOR Trunc_{|K_E|}(H(s_F || M || R || ID_F || "wrap"))
    Z_F = derive_wrap_mask(provision["secret_f"], provision["M"], provision["R"], ID_F)
    KE  = unwrap_key(provision["W_F"], Z_F)

    print(f"[Follower] Enrollment complete. R = {provision['R'].hex()}")
    print(f"[Follower] C_L = {provision['C_L'].hex()}")
    print("[Follower] Mission encryption key K_E recovered locally from W_F + secret_f.")

    state = {
        "M":       provision["M"],
        "R":       provision["R"],
        "ID_L":    provision["ID_L"],
        "C_L":     provision["C_L"],
        "N":       provision["N"],
        "KE":      KE,
        "FORS_K":  provision["FORS_K"],
        "FORS_A":  provision["FORS_A"],
        "q_last":  -1,          # no command accepted yet; q must be > q_last
    }

    with open(args.out, "wb") as f:
        pickle.dump(state, f)
    print(f"\n[Follower] State saved to '{args.out}'.")
    print("[Follower] You may now stop the GCS. Run 'hcas_follower.py run' when ready.")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def verify_and_accept(state: dict, payload: bytes) -> tuple[bool, str]:
    """
    Run the full follower verification procedure on a received Msg_q
    payload. Returns (accepted, detail) where detail is either the
    decrypted command string or a rejection reason.
    """
    t0 = time.perf_counter_ns()

    msg = deserialise_msg(payload, state["FORS_K"], state["FORS_A"])

    id_l    = msg["id_l"]
    q       = msg["q"]
    N_q     = msg["nonce"]
    X_q     = msg["ciphertext"]
    sigma   = msg["sigma"]
    p_Lq    = msg["token"]
    omega_q = msg["merkle_proof"]

    if id_l != state["ID_L"]:
        return False, f"ID_L mismatch: expected {state['ID_L']!r}, got {id_l!r}"

    # Recompute AD_q and d_q  (Eq. 9, 12)
    AD_q = id_l + q.to_bytes(4, "big") + state["M"] + state["R"]
    d_q  = H(AD_q, N_q, X_q)

    # Freshness check
    if q <= state["q_last"]:
        return False, f"replayed or out-of-order command q={q} (last accepted {state['q_last']})"

    # Eq. 14: token p_{L,q} is committed under C_L
    # NOTE: merkle_verify signature is (leaf, path, root, idx)
    if not merkle_verify(p_Lq, omega_q, state["C_L"], q):
        return False, f"MVerify failed for q={q}: token not committed under C_L"

    # Eq. 15: FORS opening sigma_q correctly opens p_{L,q} for digest d_q
    if not state["fors"].verify(p_Lq, d_q, sigma):
        return False, f"FORSVerify failed for q={q}: invalid one-time opening"

    # Eq. 16: only decrypt after both authentication checks pass
    try:
        cmd_bytes = aead_dec(state["KE"], N_q, X_q, AD_q)
    except Exception as e:
        return False, f"AEAD authentication/decryption failed for q={q}: {e}"

    state["q_last"] = q

    elapsed_us = (time.perf_counter_ns() - t0) / 1e3
    cmd_str = cmd_bytes.decode("utf-8")
    print(f"[Follower] Cmd #{q} verified+decrypted in {elapsed_us:.1f} us -> '{cmd_str}'")
    return True, cmd_str


def cmd_run(args: argparse.Namespace) -> None:
    """
    Load follower_state.pkl from disk (no GCS connection made) and
    connect directly to the leader for runtime command verification.
    """
    with open(args.state, "rb") as f:
        state = pickle.load(f)

    state["fors"] = FORS(state["FORS_K"], state["FORS_A"])

    print(f"[Follower] Loaded state from '{args.state}'. q_last={state['q_last']}, N={state['N']}")
    print(f"[Follower] R   = {state['R'].hex()}")
    print(f"[Follower] C_L = {state['C_L'].hex()}\n")

    print(f"[Follower] Connecting to leader at {args.leader_host}:{args.leader_port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.leader_host, args.leader_port))
    print("[Follower] Connected. Waiting for commands ...\n")

    try:
        while True:
            payload = _recv_framed(sock)
            accepted, detail = verify_and_accept(state, payload)

            if accepted:
                _send_framed(sock, f"ACK q={state['q_last']}".encode())
            else:
                print(f"[Follower] REJECTED: {detail}")
                _send_framed(sock, f"REJECT: {detail}".encode())

    except (KeyboardInterrupt, EOFError):
        print("\n[Follower] Interrupted.")
    except ConnectionError as e:
        print(f"\n[Follower] Connection closed: {e}")
    finally:
        sock.close()
        # Persist q_last so a restart doesn't accept replays of already-seen commands.
        state.pop("fors", None)    # FORS object is not picklable-stable across runs; rebuilt on load
        with open(args.state, "wb") as f:
            pickle.dump(state, f)
        print(f"[Follower] State saved to '{args.state}' (q_last={state['q_last']}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HCAS-CT Follower UAV")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_enroll = sub.add_parser("enroll", help="enroll with the GCS and save state to disk")
    p_enroll.add_argument("--gcs-host", required=True, help="GCS enrollment server address")
    p_enroll.add_argument("--gcs-port", type=int, default=9000, help="GCS enrollment server port")
    p_enroll.add_argument("--out", default=DEFAULT_STATE_FILE,
                          help=f"output file for state (default: {DEFAULT_STATE_FILE})")
    p_enroll.set_defaults(func=cmd_enroll)

    p_run = sub.add_parser("run", help="load state from disk and verify commands from the leader")
    p_run.add_argument("--state", default=DEFAULT_STATE_FILE,
                       help=f"state file to load (default: {DEFAULT_STATE_FILE})")
    p_run.add_argument("--leader-host", required=True, help="leader UAV runtime address")
    p_run.add_argument("--leader-port", type=int, default=9100, help="leader UAV runtime port")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
