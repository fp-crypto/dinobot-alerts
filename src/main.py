from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from ape import chain, project, networks
from ape.contracts import ContractInstance
from ape.api.transactions import ReceiptAPI
from telebot.async_telebot import AsyncTeleBot

import asyncio
import concurrent

import hmac
import hashlib
from datetime import datetime
from os import environ
from dataclasses import dataclass
from functools import lru_cache


geth_url = environ["GETH_URL"]
network = networks.parse_network_choice(f"ethereum:mainnet:{geth_url}")

app = FastAPI(on_startup=[network.__enter__], on_shutdown=[network.__exit__])

telegram_bot_key = environ["TELEGRAM_BOT_KEY"]
bot = AsyncTeleBot(telegram_bot_key)
alerts_enabled = (
    True if "ALERTS_ENABLED" in environ and environ["ALERTS_ENABLED"] == "1" else False
)

etherscan_base_url = "https://etherscan.io/"

trade_handler = "0xb634316E06cC0B358437CbadD4dC94F1D3a92B3b"
barn_solver = "0x8a4e90e9AFC809a69D2a3BDBE5fff17A12979609"
prod_solver = "0x398890BE7c4FAC5d766E1AEFFde44B2EE99F38EF"
solvers = [barn_solver, prod_solver]
signing_key = (
    environ["TENDERLY_SIGNING_KEY"] if "TENDERLY_SIGNING_KEY" in environ else ""
)

sync_threads = concurrent.futures.ThreadPoolExecutor()


CHAT_IDS = {
    "FP_ALERTS": "-881132649",
    "SEASOLVER": "-1001516144118",
}

ETH_ADDR = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
WETH_ADDR = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
MKR_ADDR = "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2"


class Alert(BaseModel):
    id: str
    event_type: str | None = None
    transaction: dict


_processed_hashes: set[str] = set()

notification_lock = asyncio.Lock()

@app.post("/solver/solve", status_code=200)
async def alert_solver_solve(alert: Alert, request: Request) -> dict:

    if not await isValidSignature(request):
        raise HTTPException(status_code=401, detail="Signature not valid")

    txn = alert.transaction
    hash = txn["hash"]

    if hash in _processed_hashes:
        return {"success": True, "is_redundant": True}

    msgs = await asyncio.get_event_loop().run_in_executor(
        sync_threads, generate_solver_alerts, hash
    )

    async with notification_lock:
        # Check again
        if hash in _processed_hashes:
            return {"success": True, "is_redundant": True}

        calls = []
        for msg in msgs:
            calls.append(send_message(msg))
        await asyncio.gather(*calls)

        _processed_hashes.add(hash)

    return {"success": True}


@app.post("/solver/revert", status_code=200)
async def alert_solver_revert(alert: Alert, request: Request) -> dict:

    if not await isValidSignature(request):
        raise HTTPException(status_code=401, detail="Signature not valid")

    txn = alert.transaction
    hash = txn["hash"]

    if hash in _processed_hashes:
        return {"success": True, "is_redundant": True}

    msg = await asyncio.get_event_loop().run_in_executor(
        sync_threads, process_revert, hash
    )

    async with notification_lock:
        # Check again
        if hash in _processed_hashes:
            return {"success": True, "is_redundant": True}

        await send_message(msg)

        _processed_hashes.add(hash)

    return {"success": True}


def generate_solver_alerts(txn_hash: str) -> list[str]:

    receipt = networks.provider.get_receipt(txn_hash)
    settlement = project.Settlement.at("0x9008d19f58aabd9ed0d60971565aa8510560ab41")

    target_logs = receipt.decode_logs(
        [
            settlement.Settlement,
            settlement.Trade,
            project.ERC20.contract_type.events["Transfer"],
        ]
    )

    settlement_logs = [l for l in target_logs if l.event_name == "Settlement"]
    trade_logs = [l for l in target_logs if l.event_name == "Trade"]
    transfer_logs = [l for l in target_logs if l.event_name == "Transfer"]

    solvers = [l.dict()["event_arguments"]["solver"] for l in settlement_logs]
    solver = next((solver for solver in solvers if solver in solvers), None)
    if solver == None:
        return []

    trades = enumerate_trades(trade_logs)
    slippage = calculate_slippage(trades, transfer_logs)
    alerts = [format_solver_alert(solver, txn_hash, receipt, trades, slippage)]

    return alerts


def calculate_slippage(trades: list[dict], transfer_logs):

    slippages = {}

    for trade in trades:
        buy_token_address = trade["buy_token_address"]

        # If there is a trade for eth, use weth instead since TH will never
        # get native eth
        if buy_token_address.lower() == ETH_ADDR.lower():
            buy_token_address = WETH_ADDR

        # we might have calculated the slippage previously
        if buy_token_address in slippages:
            continue

        token_transfers = [
            l.dict()["event_arguments"]
            for l in transfer_logs
            if l.contract_address == buy_token_address
        ]

        amount_in = sum(
            [l["value"] for l in token_transfers if l["to"] == trade_handler]
        )
        amount_out = sum(
            [l["value"] for l in token_transfers if l["from"] == trade_handler]
        )
        slippages[buy_token_address] = amount_in - amount_out

    return slippages


def enumerate_trades(logs) -> list[dict]:
    trades: list[dict] = []

    for l in logs:
        args = l.dict()["event_arguments"]

        sell_token = _token_info(args["sellToken"])
        buy_token = _token_info(args["buyToken"])

        trade = {
            "owner": args["owner"],
            "sell_token_address": args["sellToken"],
            "sell_token_symbol": sell_token.symbol,
            "sell_token_decimals": sell_token.decimals,
            "buy_token_address": args["buyToken"],
            "buy_token_symbol": buy_token.symbol,
            "buy_token_decimals": buy_token.decimals,
            "sell_amount": args["sellAmount"],
            "buy_amount": args["buyAmount"],
            "fee_amount": args["feeAmount"],
            "order_uid": "0x" + args["orderUid"].hex(),
        }
        trades.append(trade)
    return trades


def format_solver_alert(
    solver,
    txn_hash: str,
    txn_receipt: ReceiptAPI,
    trade_data: list[dict],
    slippages: dict,
) -> str:

    cow_explorer_url = f'https://explorer.cow.fi/orders/{trade_data[0]["order_uid"]}'
    cow_explorer_url = f"https://explorer.cow.fi/tx/{txn_hash}"
    ethtx_explorer_url = f"https://ethtx.info/mainnet/{txn_hash}"
    eigen_url = f"https://eigenphi.io/mev/eigentx/{txn_hash}"
    ts = chain.blocks[txn_receipt.block_number].timestamp
    index = get_index_in_block(txn_receipt)
    index = index if index != 1_000_000 else "???"

    dt = datetime.utcfromtimestamp(ts).strftime("%m/%d %H:%M")
    msg = f'{"🧜‍♂️" if solver == prod_solver else "🐓"} *New solve detected!*\n'
    msg += f"by [{solver[0:7]}...]({etherscan_base_url}address/{solver})  index: {index} @ {dt}\n\n"
    msg += f"📕 *Trade(s)*:\n"
    for t in trade_data:
        user = t["owner"]
        sell_amt = round(t["sell_amount"] / 10 ** t["sell_token_decimals"], 4)
        buy_amt = round(t["buy_amount"] / 10 ** t["buy_token_decimals"], 4)
        buy_token = t["buy_token_address"]
        if buy_token.lower() == "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE".lower():
            buy_token = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        msg += f'    [{t["sell_token_symbol"]}]({etherscan_base_url}token/{t["sell_token_address"]}) {sell_amt:,} -> [{t["buy_token_symbol"]}]({etherscan_base_url}token/{t["buy_token_address"]}) {buy_amt:,} | [{user[0:7]}...]({etherscan_base_url}address/{user})\n'
    msg += "\n✂️ *Slippages*"
    for key in slippages:
        token = _token_info(key)
        slippage = slippages[key]
        color = "🔴" if slippage < 0 else "🟢"
        amount = round(slippage / 10**token.decimals, 4)
        msg += f"\n   {color} {token.symbol}: {amount}"
    msg += f"\n\n{calc_gas_cost(txn_receipt)}"
    msg += f"\n\n🔗 [Etherscan]({etherscan_base_url}tx/{txn_hash}) | [Cow]({cow_explorer_url}) | [Eigen]({eigen_url}) | [EthTx]({ethtx_explorer_url})"

    return msg


def calc_gas_cost(txn_receipt):
    oracle = project.ORACLE.at("0x83d95e0D5f402511dB06817Aff3f9eA88224B030")

    eth_used = txn_receipt.gas_price * txn_receipt.gas_used
    gas_cost = (
        oracle.getNormalizedValueUsdc(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", eth_used
        )
        / 10**6
    )
    return f"💸 ${round(gas_cost,2):,} | {round(eth_used/1e18,4)} ETH"


def process_revert(txn_hash: str) -> None | str:
    txn_receipt = networks.provider.get_receipt(txn_hash)

    failed = txn_receipt.failed
    sender = txn_receipt.transaction.sender
    if not failed or sender not in solvers:
        return
    msg = f"*🤬  Failed Transaction detected!*\n\n"
    e = "🧜‍♂️" if sender == prod_solver else "🐓"
    _, _, markdown = abbreviate_address(sender)
    msg += f"Sent from {markdown} {e}\n\n"
    msg += f"{calc_gas_cost(txn_receipt)}"
    msg += f"\n\n🔗 [Etherscan]({etherscan_base_url}tx/{txn_hash}) | [Tenderly](https://dashboard.tenderly.co/tx/mainnet/{txn_hash})"
    return msg


def get_index_in_block(tx: ReceiptAPI):
    hashes = [x.txn_hash.hex() for x in chain.blocks[tx.block_number].transactions]
    try:
        return hashes.index(tx.txn_hash)
    except:
        return 1_000_000  # Not found


def abbreviate_address(address):
    link = f"https://etherscan.io/address/{address}"
    abbr = address[0:7]
    markdown = f"[{abbr}...]({link})"
    return abbr, link, markdown


async def send_message(msg):
    if alerts_enabled:
        chat_id = CHAT_IDS["SEASOLVER"]
    else:
        chat_id = CHAT_IDS["FP_ALERTS"]
    return await bot.send_message(
        chat_id, msg, parse_mode="markdown", disable_web_page_preview=True
    )


async def isValidSignature(request: Request) -> bool:
    if signing_key == "":
        return True

    if not "X-Tenderly-Signature" in request.headers:
        return False

    signature = request.headers["X-Tenderly-Signature"]
    timestamp = request.headers["Date"]

    body = await request.body()

    h = hmac.new(str.encode(signing_key), body, hashlib.sha256)
    h.update(str.encode(timestamp))
    digest = h.hexdigest()
    return hmac.compare_digest(signature, digest)


@dataclass
class TokenInfo:
    addr: str
    symbol: str
    decimals: int
    contract: ContractInstance | None


@lru_cache
def _token_info(addr: str) -> TokenInfo:
    token = project.ERC20.at(addr)
    if addr == ETH_ADDR:
        return TokenInfo(addr, "ETH", 18, None)
    elif addr == MKR_ADDR:
        return TokenInfo(addr, "WETH", 18, token)
    elif addr == MKR_ADDR:
        return TokenInfo(addr, "MKR", 18, token)

    decimals = token.decimals()

    try:
        symbol = token.symbol()
    except:
        symbol = "? Cannot Find ?"

    return TokenInfo(addr, symbol, decimals, token)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("__main__:app", host="0.0.0.0")
