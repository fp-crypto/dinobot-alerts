from os import environ
from typing import Tuple, TypedDict
from enum import IntEnum


class Network(IntEnum):
    Mainnet = 1
    Gnosis = 100
    Arbitrum = 42161


class ChainValues(TypedDict):
    NETWORK_NAME: str
    NETWORK_SYMBOL: str
    NETWORK_NATIVE_TOKEN: str
    WRAPPED_NATIVE_TOKEN: str
    RPC_URL: str
    EMOJI: str
    NATIVE_USD_ORACLE: str
    EXPLORER_URL: str
    EXPLORER_NAME: str
    TENDERLY_CHAIN_IDENTIFIER: str
    COWSWAP_EXPLORER_URL: str
    COWSWAP_API_URLS: Tuple[str, str]
    TRADE_HANDLER: str


CHAIN_VALUES: dict[int, ChainValues] = {
    Network.Mainnet: {
        "NETWORK_NAME": "Ethereum Mainnet",
        "NETWORK_SYMBOL": "eth",
        "NETWORK_NATIVE_TOKEN": "ETH",
        "WRAPPED_NATIVE_TOKEN": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "RPC_URL": environ["GETH_URL"],
        "EMOJI": "🇪🇹",
        "NATIVE_USD_ORACLE": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
        "EXPLORER_URL": "https://etherscan.io/",
        "EXPLORER_NAME": "Etherscan",
        "TENDERLY_CHAIN_IDENTIFIER": "mainnet",
        "COWSWAP_EXPLORER_URL": "https://explorer.cow.fi/",
        "COWSWAP_API_URLS": (
            "https://api.cow.fi/mainnet/api/v1/",
            "https://barn.api.cow.fi/mainnet/api/v1/",
        ),
        "TRADE_HANDLER": "0xb634316E06cC0B358437CbadD4dC94F1D3a92B3b",
    },
    Network.Gnosis: {
        "NETWORK_NAME": "Gnosis Chain",
        "NETWORK_SYMBOL": "gc",
        "NETWORK_NATIVE_TOKEN": "XDAI",
        "WRAPPED_NATIVE_TOKEN": "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d",
        "RPC_URL": environ["GC_RPC_URL"],
        "EMOJI": "🦉️",
        "NATIVE_USD_ORACLE": "0x0000000000000000000000000000000000000000",
        "EXPLORER_URL": "https://gnosisscan.io/",
        "EXPLORER_NAME": "Gnosisscan",
        "TENDERLY_CHAIN_IDENTIFIER": "gnosis-chain",
        "COWSWAP_EXPLORER_URL": "https://explorer.cow.fi/gc/",
        "COWSWAP_API_URLS": (
            "https://api.cow.fi/xdai/api/v1/",
            "https://barn.api.cow.fi/xdai/api/v1/",
        ),
        "TRADE_HANDLER": "0x67a5802068f9E1ee03821Be0cD7f46D04f4dF33A",
    },
    Network.Arbitrum: {
        "NETWORK_NAME": "Arbitrum",
        "NETWORK_SYMBOL": "arb",
        "NETWORK_NATIVE_TOKEN": "ETH",
        "WRAPPED_NATIVE_TOKEN": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "RPC_URL": environ["ARBITRUM_RPC_URL"],
        "EMOJI": "🤠",
        "NATIVE_USD_ORACLE": "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612",
        "EXPLORER_URL": "https://arbiscan.io/",
        "EXPLORER_NAME": "Arbiscan",
        "TENDERLY_CHAIN_IDENTIFIER": "arbitrum",
        "COWSWAP_EXPLORER_URL": "https://explorer.cow.fi/arb1/",
        "COWSWAP_API_URLS": (
            "https://api.cow.fi/arbitrum_one/api/v1/",
            "https://barn.api.cow.fi/arbitrum_one/api/v1/",
        ),
        "TRADE_HANDLER": "0x1111111111111111111111111111111111111111",  # fake address
    },
}
