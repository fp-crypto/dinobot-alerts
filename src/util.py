from web3 import Web3, AsyncWeb3
from web3.contract.contract import Contract
from web3.contract.async_contract import AsyncContract
import json
from typing import overload
from cachetools import cached


@overload
def get_contract(w3: AsyncWeb3, abi_file: str, address: str) -> AsyncContract:
    ...


@overload
def get_contract(w3: Web3, abi_file: str, address: str) -> Contract:
    ...

@cached(cache={})
def get_contract(
    w3: Web3 | AsyncWeb3, abi_file: str, address: str
) -> Contract | AsyncContract:
    with open(f"./contracts/{abi_file}", "r", encoding="utf-8") as f:
        abi = json.load(f)
    return w3.eth.contract(
        address=Web3.to_checksum_address(address),
        abi=abi,
        decode_tuples=True,
    )
