import abc
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

from netqasm.sdk.classical_communication.message import StructuredMessage

from pydynaa import EventExpression
from squidasm.run.stack.run import run
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.csocket import ClassicalSocket
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network

from hashlib import md5
from zlib import crc32

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class PairInfo:
    """Information that a node has about one generated pair.
    The information is filled progressively during the protocol."""

    # Index in list of all generated pairs.
    index: int

    # Basis this node measured in. 0 = Z, 1 = X.
    basis: int

    # Measurement outcome (0 or 1).
    outcome: int

    # Whether the other node measured his qubit in the same basis or not.
    same_basis: Optional[bool] = None

    # Whether to use this pair to estimate errors by comparing the outcomes.
    test_outcome: Optional[bool] = None

    # Whether measurement outcome is the same as the other node's.
    # (Only for pairs used for error estimation.)
    same_outcome: Optional[bool] = None

    # if reconciliation (xor over qubit blocks) is successful one qubit of the set of epr pairs is selected
    selected_outcome: bool = False


class QkdProgram(Program, abc.ABC):
    PEER: str
    ALL_MEASURED = "All qubits measured"
    IS_INIT: bool

    def __init__(self, num_epr: int, num_test_bits: int = None):
        self._num_epr = num_epr
        self._num_test_bits = num_epr // 4 if num_test_bits is None else num_test_bits
        self._buf_msgs = []
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

    def _distribute_states(
        self, context: ProgramContext, is_init: bool
    ) -> Generator[EventExpression, None, List[PairInfo]]:
        """
        Generates and measures a number of entangled qubits in a random basis.
        :param context: Local program context
        :param is_init: Flag to designate if this node started the protocol
        :return: List of PairInfo with index, outcome and basis per measurement.
        """
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]

        results = []

        for i in range(self._num_epr):
            basis = random.randint(0, 1)
            if is_init:
                q = epr_socket.create_keep(1)[0]
            else:
                q = epr_socket.recv_keep(1)[0]
            if basis == 1:
                q.H()
            m = q.measure()
            yield from conn.flush()
            results.append(PairInfo(index=i, outcome=int(m), basis=basis))

        return results

    @staticmethod
    def _filter_bases(
        socket: ClassicalSocket, pairs_info: List[PairInfo], is_init: bool
    ) -> Generator[EventExpression, None, List[PairInfo]]:
        """
        Communicates the random base choices with the peer and filters out the measured entangled qubits
         that where measured in a different basis.
        :param socket: classical socket to peer.
        :param pairs_info: List of PairInfo object containing relevant information for the measured entangled qubits
        :param is_init: Flag to designate if this node started the protocol
        :return: List of PairInfo with the same_basis field filled in.
        """
        bases = [(i, pairs_info[i].basis) for (i, pair) in enumerate(pairs_info)]

        if is_init:
            socket.send_structured(StructuredMessage("Bases", bases))
            remote_bases = (yield from socket.recv_structured()).payload
        else:
            remote_bases = (yield from socket.recv_structured()).payload
            socket.send_structured(StructuredMessage("Bases", bases))

        for (i, basis), (remote_i, remote_basis) in zip(bases, remote_bases):
            assert i == remote_i
            pairs_info[i].same_basis = basis == remote_basis

        return pairs_info

    @staticmethod
    def _apply_reconciliation(
            socket: ClassicalSocket,
            pairs_info: List[PairInfo],
            is_init: bool
    ) -> Generator[EventExpression, None, Tuple[List[PairInfo], List[int]]]:
        key_indices = [pair.index for pair in pairs_info
                       if pair.same_basis and not pair.test_outcome]
        rec_key = [pairs_info[key_indices[2*i]].outcome ^ pairs_info[key_indices[2*i+1]].outcome
                   for i in range(len(key_indices)//2)]
        if is_init:
            socket.send_structured(StructuredMessage("Rec key", rec_key))
            target_rec_key = (yield from socket.recv_structured()).payload
        else:
            target_rec_key = (yield from socket.recv_structured()).payload
            socket.send_structured(StructuredMessage("Rec key", rec_key))

        for i in range(len(key_indices)//2):
            if rec_key[i]==target_rec_key[i]:
                pairs_info[key_indices[2*i]].selected_outcome = True
        return pairs_info, rec_key

    @staticmethod
    def _estimate_error_rate(
        socket: ClassicalSocket,
        pairs_info: List[PairInfo],
        num_test_bits: int,
        is_init: bool,
    ) -> Generator[EventExpression, None, Tuple[List[PairInfo], float]]:
        """
        Estimates the error rate for the raw key, by choosing a random subset of the key to exchange with the peer.
        :param socket: classical socket to peer.
        :param pairs_info: List of PairInfo object containing relevant information for the measured entangled qubits
        :param num_test_bits: The amount bits from the raw key to use for the estimation.
        :param is_init: Flag to designate if this node started the protocol
        :return: Tuple with list of PairInfo and the estimated error rate
        """
        if is_init:
            same_basis_indices = [pair.index for pair in pairs_info if pair.same_basis]
            test_indices = random.sample(
                same_basis_indices, min(num_test_bits, len(same_basis_indices))
            )
            for pair in pairs_info:
                pair.test_outcome = pair.index in test_indices

            test_outcomes = [(i, pairs_info[i].outcome) for i in test_indices]

            socket.send_structured(StructuredMessage("Test indices", test_indices))
            target_test_outcomes = (yield from socket.recv_structured()).payload
            socket.send_structured(StructuredMessage("Test outcomes", test_outcomes))
        else:
            test_indices = (yield from socket.recv_structured()).payload
            for pair in pairs_info:
                pair.test_outcome = pair.index in test_indices

            test_outcomes = [(i, pairs_info[i].outcome) for i in test_indices]

            socket.send_structured(StructuredMessage("Test outcomes", test_outcomes))
            target_test_outcomes = (yield from socket.recv_structured()).payload

        num_error = 0
        for (i1, t1), (i2, t2) in zip(test_outcomes, target_test_outcomes):
            assert i1 == i2
            if t1 != t2:
                num_error += 1
                pairs_info[i1].same_outcome = False
            else:
                pairs_info[i1].same_outcome = True

        return pairs_info, (num_error / num_test_bits)

    @staticmethod
    def _final_error_rate(
            socket: ClassicalSocket,
            pairs_info: List[PairInfo],
            num_test_bits: int,
            is_init: bool,
    ) -> Generator[EventExpression, None, Tuple[List[PairInfo], float]]:
        """
        Calculates the error rate for the key after applying reconciliation,
        by choosing a random subset of the key to exchange with the peer.
        :param socket: classical socket to peer.
        :param pairs_info: List of PairInfo object containing relevant information for the measured entangled qubits
        :param num_test_bits: The amount bits from the raw key to use for the estimation.
        :param is_init: Flag to designate if this node started the protocol
        :return: Tuple with list of PairInfo and the estimated error rate
        """
        if is_init:
            indices_after_conciliation = [pair.index for pair in pairs_info if pair.selected_outcome]
            test_indices = random.sample(
                indices_after_conciliation, min(num_test_bits, len(indices_after_conciliation))
            )
            for pair in pairs_info:
                pair.test_outcome = pair.index in test_indices

            test_outcomes = [(i, pairs_info[i].outcome) for i in test_indices]

            socket.send_structured(StructuredMessage("Test indices", test_indices))
            target_test_outcomes = (yield from socket.recv_structured()).payload
            socket.send_structured(StructuredMessage("Test outcomes", test_outcomes))
        else:
            test_indices = (yield from socket.recv_structured()).payload
            for pair in pairs_info:
                pair.test_outcome = pair.index in test_indices

            test_outcomes = [(i, pairs_info[i].outcome) for i in test_indices]

            socket.send_structured(StructuredMessage("Test outcomes", test_outcomes))
            target_test_outcomes = (yield from socket.recv_structured()).payload

        num_error = 0
        for (i1, t1), (i2, t2) in zip(test_outcomes, target_test_outcomes):
            assert i1 == i2
            if t1 != t2:
                num_error += 1
                pairs_info[i1].same_outcome = False
            else:
                pairs_info[i1].same_outcome = True

        return pairs_info, (num_error / num_test_bits)

    def run(
            self, context: ProgramContext
    ) -> Generator[EventExpression, None, Dict[str, Any]]:
        csocket = context.csockets[self.PEER]

        pairs_info = yield from self._distribute_states(context, is_init=self.IS_INIT)
        self.logger.info("Finished distributing states")

        if self.IS_INIT:
            m = yield from csocket.recv()
            if m != self.ALL_MEASURED:
                raise RuntimeError("Failed to distribute BB84 states")
        else:
            csocket.send(self.ALL_MEASURED)

        pairs_info = yield from self._filter_bases(csocket, pairs_info, is_init=self.IS_INIT)

        pairs_info, raw_error_rate = yield from self._estimate_error_rate(
            csocket, pairs_info, self._num_test_bits, is_init=self.IS_INIT
        )
        self.logger.info(f"Estimates error rate: {raw_error_rate}")

        raw_key = [
            pair.outcome
            for pair in pairs_info
            if pair.same_basis and not pair.test_outcome
        ]

        self.logger.info(f"Prepared Raw key: {raw_key}")

        pairs_info, rec_pattern = yield from self._apply_reconciliation(csocket, pairs_info, is_init=self.IS_INIT)

        key_after_reconciliation = [
            pair.outcome
            for pair in pairs_info
            if pair.selected_outcome
        ]
        self.logger.info(f"Key after reconciliation: {key_after_reconciliation}")

        pairs_info, coded_error_rate = yield from self._final_error_rate(
            csocket, pairs_info, num_test_bits=self._num_test_bits, is_init=self.IS_INIT)

        result = {"raw_key": raw_key,
                  "rec_pattern": rec_pattern,
                  "key_after_reconciliation": key_after_reconciliation,
                  "raw_error_rate": raw_error_rate,
                  "coded_error_rate": coded_error_rate
                  }

        return result


class AliceProgram(QkdProgram):
    PEER = "Bob"
    IS_INIT = True

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=1,
        )


class BobProgram(QkdProgram):
    PEER = "Alice"
    IS_INIT = False

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=1,
        )


def eval_network(link_noise, num_epr, quiet=False, num_test_bits=None):
    cfg = create_two_node_network(node_names=["Alice", "Bob"], link_noise=link_noise)

    alice_program = AliceProgram(num_epr=num_epr, num_test_bits=num_test_bits)
    bob_program = BobProgram(num_epr=num_epr, num_test_bits=num_test_bits)

    # toggle logging. Set to logging.INFO for logging of events.
    alice_program.logger.setLevel(logging.ERROR)
    bob_program.logger.setLevel(logging.ERROR)

    alice_results, bob_results = run(
        config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1
    )
    for i, (alice_result, bob_result) in enumerate(zip(alice_results, bob_results)):
        error_rate = bob_result["raw_error_rate"]
        coded_error_rate = bob_result["coded_error_rate"]
        if not quiet:
            rk_alice = "".join(str(b) for b in alice_result["raw_key"])
            rec_alice = "".join(" " + str(b) for b in alice_result["rec_pattern"])
            kar_alice = "".join(str(b) for b in alice_result["key_after_reconciliation"])
            rk_bob = "".join(str(b) for b in bob_result["raw_key"])
            rec_bob = "".join(" " + str(b) for b in bob_result["rec_pattern"])
            kar_bob = "".join(str(b) for b in bob_result["key_after_reconciliation"])
            print(f"                                                          HASH")
            print(f"alice raw key:                  {rk_alice[:20]} ...  {md5(rk_alice.encode()).hexdigest()[:4]}")
            print(f"      reconciliation pattern:   {rec_alice[:20]} ...  {md5(rec_alice.encode()).hexdigest()[:4]}")
            print(f"      key aft. reconciliation:  {kar_alice[:20]} ...  {md5(kar_alice.encode()).hexdigest()[:4]}")
            print(f"bob   raw key:                  {rk_bob[:20]} ...  {md5(rk_bob.encode()).hexdigest()[:4]}")
            print(f"      reconciliation key:       {rec_bob[:20]} ...  {md5(rec_bob.encode()).hexdigest()[:4]}")
            print(f"      key aft. reconciliation:  {kar_bob[:20]} ...  {md5(kar_bob.encode()).hexdigest()[:4]}")

        return {"error_rate": error_rate, "coded_error_rate": coded_error_rate}


if __name__ == "__main__":
    draw_curves = True
    single_run = False
    if single_run:
        eval_network(0.1,1000, quiet=False, num_test_bits=100)

    if draw_curves:
        results = {i: eval_network(link_noise=float(i), num_epr=2000, num_test_bits=250, quiet=True)
                   for i in tqdm(np.linspace(start=0, stop=1, num=10))}
        error_rates = {k: v['error_rate'] for k, v in results.items()}
        coded_error_rates = {k: v['coded_error_rate'] for k, v in results.items()}

        plt.figure(figsize=(5, 4))
        plt.plot(list(error_rates.keys()), list(error_rates.values()), label="raw")
        plt.plot(list(coded_error_rates.keys()), list(coded_error_rates.values()), label="coded")
        plt.autoscale()
        plt.legend()
        plt.grid()
        plt.xlim([0,1])
        plt.ylim([0,0.5])
        plt.xlabel("link noise")
        plt.ylabel("QBER")
        plt.savefig("QBER_curve.png", dpi=300)
        plt.show()

