from ape.api.providers import ProviderAPI
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from ape import project, networks
from ape.contracts import ContractInstance, ContractEvent
from ape.api.transactions import ReceiptAPI
from telebot.async_telebot import AsyncTeleBot
from requests import Session as ClientSession
from requests.adapters import HTTPAdapter, Retry

import asyncio
import concurrent

import hmac
import hashlib
from datetime import datetime
from os import environ
from dataclasses import dataclass
from functools import lru_cache


eth_rpc_url = environ["GETH_URL"]
gc_rpc_url = environ["GC_RPC_URL"]
network = networks.parse_network_choice(f"ethereum:mainnet:{eth_rpc_url}")

app = FastAPI(on_startup=[network.__enter__], on_shutdown=[network.__exit__])

telegram_bot_key = environ["TELEGRAM_BOT_KEY"]
bot = AsyncTeleBot(telegram_bot_key)
alerts_enabled = (
    True if "ALERTS_ENABLED" in environ and environ["ALERTS_ENABLED"] == "1" else False
)

etherscan_base_url = "https://etherscan.io/"
gnosisscan_base_url = "https://gnosisscan.io/"

cowswap_prod_api_base_url = "https://api.cow.fi/mainnet/api/v1/"
cowswap_barn_api_base_url = "https://barn.api.cow.fi/mainnet/api/v1/"

trade_handler = {
    1: "0xb634316E06cC0B358437CbadD4dC94F1D3a92B3b",
    100: "0x67a5802068f9E1ee03821Be0cD7f46D04f4dF33A",
}

barn_solvers: list[str] = [
    "0x8a4e90e9afc809a69d2a3bdbe5fff17a12979609",
    "0xD01BA5b3C4142F358EfFB4d6Cb44A11E31600330",
    "0xAc6Cc8E2f0232B5B213503257C861235F4ED42c1",
    "0xC8D2f12a9505a82C4f6994204f4BbF095183E63A",  # gnosis chain sover
]
prod_solvers: list[str] = [
    "0x398890be7c4fac5d766e1aeffde44b2ee99f38ef",
    "0x43872b55A12E087935765611851E94e3f0a79249",
    "0x0DdcB0769a3591230cAa80F85469240b71442089",
    "0xE3068acB5b5672408eADaD4417e7d3BA41D4FEBe",  # gnosis chain sover
]
solvers: list[str] = barn_solvers + prod_solvers

signing_key = (
    environ["TENDERLY_SIGNING_KEY"] if "TENDERLY_SIGNING_KEY" in environ else ""
)

sync_threads = concurrent.futures.ThreadPoolExecutor()

CHAT_IDS = {
    "FP_ALERTS": "-881132649",
    "SEASOLVER": "-1001516144118",
    "SEASOLVER_SA": "-1001829083462",
}

NATIVE_TOKEN_ADDR = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
WETH_ADDR = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WXDAI_ADDR = "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d"

MKR_ADDR = "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2"
SDAI_ADDR = "0x83F20F44975D03b1b09e64809B757c47f942BEeA"

COW_SWAP_ETH_FLOW_ADDR = "0x40A50cf069e992AA4536211B23F286eF88752187"
COW_SWAP_SETTLEMENT_ADDR = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"

CHAINS: dict[int, str] = {
    1: "eth",
    100: "gc",
}

APE_NETWORK_STRING: dict[int, str] = {
    1: f"ethereum:mainnet:{eth_rpc_url}",
    100: f"gnosis:mainnet:{gc_rpc_url}",
}

WRAPPED_NATIVE_TOKEN_ADDR: dict[int, str] = {
    1: WETH_ADDR,
    100: WXDAI_ADDR,
}


class Transaction(BaseModel):
    hash: str
    network: str


class Alert(BaseModel):
    id: str
    event_type: str | None = None
    transaction: Transaction


_processed_hashes: set[str] = set()

notification_lock = asyncio.Lock()

# monkey patch to avoid issue parsing access_list
from ape_ethereum.transactions import DynamicFeeTransaction, AccessListTransaction

DynamicFeeTransaction.__fields__["access_list"].allow_none = True
AccessListTransaction.__fields__["access_list"].allow_none = True


@app.post("/solver/solve", status_code=200)
async def alert_solver_solve(alert: Alert, request: Request) -> dict:

    if not await isValidSignature(request):
        raise HTTPException(status_code=401, detail="Signature not valid")

    txn = alert.transaction
    hash = txn.hash
    chain_id = int(txn.network)

    if hash in _processed_hashes:
        return {"success": True, "is_redundant": True}

    async with notification_lock:
        # Check again
        if hash in _processed_hashes:
            return {"success": True, "is_redundant": True}

        msgs = await asyncio.get_event_loop().run_in_executor(
            sync_threads, generate_solver_alerts, hash, chain_id
        )

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
    chain_id = int(txn["network" if "network" in txn else "network_id"])

    if hash in _processed_hashes:
        return {"success": True, "is_redundant": True}

    async with notification_lock:
        # Check again
        if hash in _processed_hashes:
            return {"success": True, "is_redundant": True}

        msg = await asyncio.get_event_loop().run_in_executor(
            sync_threads, process_revert, hash, chain_id
        )

        await send_message(msg)

        _processed_hashes.add(hash)

    return {"success": True}


def generate_solver_alerts(txn_hash: str, chain_id: int) -> list[str]:
    with networks.parse_network_choice(APE_NETWORK_STRING[chain_id]) as provider:
        receipt = provider.get_receipt(txn_hash)
        # settlement = project.Settlement.at(COW_SWAP_SETTLEMENT_ADDR)

        target_logs = receipt.decode_logs(
            [
                # settlement.Settlement,
                # settlement.Trade,
                project.Settlement.contract_type.events["Settlement"],
                project.Settlement.contract_type.events["Trade"],
                project.ERC20.contract_type.events["Transfer"],
                project.WETH.contract_type.events["Withdrawal"],
                project.SDai.contract_type.events["Withdraw"],
                project.SDai.contract_type.events["Deposit"],
            ]
        )

        settlement_logs = [l for l in target_logs if l.event_name == "Settlement"]
        trade_logs = [l for l in target_logs if l.event_name == "Trade"]
        transfer_logs = [l for l in target_logs if l.event_name == "Transfer"]

        weth_burn_logs = [
            l
            for l in target_logs
            if l.event_name == "Withdrawal"
            and l.contract_address == WRAPPED_NATIVE_TOKEN_ADDR[chain_id]
        ]
        sdai_logs = (
            [
                l
                for l in target_logs
                if l.event_name in ["Withdraw", "Deposit"]
                and l.contract_address == SDAI_ADDR
            ]
            if chain_id == 1
            else []
        )

        solvers = [l.dict()["event_arguments"]["solver"] for l in settlement_logs]
        solver = next((solver for solver in solvers if solver in solvers), None)
        if solver == None:
            return []

        trades = enumerate_trades(
            trade_logs,
            is_barn=solver.lower() in map(str.lower, barn_solvers),
            chain_id=chain_id,
        )
        slippage = calculate_slippage(
            trades, transfer_logs, weth_burn_logs, sdai_logs, chain_id
        )
        alerts = [format_solver_alert(solver, txn_hash, receipt, trades, slippage)]

        return alerts


def calculate_slippage(
    trades: list[dict],
    transfer_logs: list[ContractEvent],
    weth_burn_logs: list[ContractEvent],
    sdai_logs: list[ContractEvent],
    chain_id: int,
):

    slippages = {}
    settlement = trades[0]["settlement"]

    for trade in trades:
        buy_token_address = trade["buy_token_address"]

        # If there is a trade for eth, use weth instead since TH will never
        # get native eth
        if buy_token_address.lower() == NATIVE_TOKEN_ADDR.lower():
            buy_token_address = WRAPPED_NATIVE_TOKEN_ADDR[chain_id]

        # we might have calculated the slippage previously
        if buy_token_address in slippages:
            continue

        token_transfers = [
            l.dict()["event_arguments"]
            for l in transfer_logs
            if l.contract_address == buy_token_address
        ]

        amount_in_th = sum(
            [l["value"] for l in token_transfers if l["to"] == trade_handler[chain_id]]
        )
        amount_out_th = sum(
            [
                l["value"]
                for l in token_transfers
                if l["from"] == trade_handler[chain_id]
            ]
        )
        slippage_th = amount_in_th - amount_out_th

        amount_in_settlement = sum(
            [l["value"] for l in token_transfers if l["to"] == settlement]
        )
        amount_out_settlement = sum(
            [l["value"] for l in token_transfers if l["from"] == settlement]
        )
        slippage_settlement = amount_in_settlement - amount_out_settlement

        if slippage_th == 0 and slippage_settlement == 0:
            continue

        slippages[buy_token_address] = {
            "th": slippage_th,
            "cow": slippage_settlement,
        }

    # adjust for fees
    for trade in trades:
        sell_token_address = trade["sell_token_address"]

        if sell_token_address not in slippages:
            continue

        fee_amount = trade["fee_amount"]
        slippages[sell_token_address]["cow"] -= fee_amount

    # adjust for any weth burns
    wrapped_native_token_addr: str = WRAPPED_NATIVE_TOKEN_ADDR[chain_id]
    if wrapped_native_token_addr in slippages:

        weth_burns_settlement = sum(
            [
                l.dict()["event_arguments"]["wad"]
                for l in weth_burn_logs
                if l.contract_address == wrapped_native_token_addr
                and l.dict()["event_arguments"]["src"] == settlement
            ]
        )
        slippages[wrapped_native_token_addr]["cow"] -= weth_burns_settlement

    if (
        wrapped_native_token_addr in slippages
        and sum(slippages[wrapped_native_token_addr].values()) == 0
    ):
        del slippages[wrapped_native_token_addr]

    if len(sdai_logs) != 0:
        sdai_deposit_withdraw_th = sum(
            [
                l.dict()["event_arguments"]["shares"] * 1
                if l.event_name == "Deposit"
                else -1
                for l in sdai_logs
                if l.dict()["event_arguments"]["owner"] == trade_handler[chain_id]
            ]
        )
        sdai_deposit_withdraw_cow = sum(
            [
                l.dict()["event_arguments"]["shares"] * 1
                if l.event_name == "Deposit"
                else -1
                for l in sdai_logs
                if l.dict()["event_arguments"]["owner"] == settlement
            ]
        )
        if SDAI_ADDR not in slippages:
            slippages[SDAI_ADDR] = {"th": 0, "cow": 0}
        slippages[SDAI_ADDR]["th"] += sdai_deposit_withdraw_th
        slippages[SDAI_ADDR]["cow"] += sdai_deposit_withdraw_cow

    return slippages


def enumerate_trades(logs, is_barn=False, chain_id=1) -> list[dict]:
    trades: list[dict] = []

    for l in logs:
        args = l.dict()["event_arguments"]

        sell_token = _token_info(args["sellToken"], chain_id)
        buy_token = _token_info(args["buyToken"], chain_id)

        owner = args["owner"]
        order_uid = "0x" + args["orderUid"].hex()

        if (
            sell_token.addr.lower() == WETH_ADDR.lower()
            and owner.lower() == COW_SWAP_ETH_FLOW_ADDR.lower()
        ):
            req_url = f"{cowswap_prod_api_base_url if not is_barn else cowswap_barn_api_base_url}orders/{order_uid}"
            with get_http_session().get(req_url) as r:
                assert r.status_code == 200
                r_json = r.json()
                if "onchainUser" in r_json:
                    owner = r_json["onchainUser"]
                    sell_token.symbol = "ETH"

        trade = {
            "owner": owner,
            "sell_token_address": args["sellToken"],
            "sell_token_symbol": sell_token.symbol,
            "sell_token_decimals": sell_token.decimals,
            "buy_token_address": args["buyToken"],
            "buy_token_symbol": buy_token.symbol,
            "buy_token_decimals": buy_token.decimals,
            "sell_amount": args["sellAmount"],
            "buy_amount": args["buyAmount"],
            "fee_amount": args["feeAmount"],
            "order_uid": order_uid,
            "settlement": l.contract_address,
            "chain_id": chain_id,
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

    is_gnosis = txn_receipt.chain_id == 100

    if is_gnosis:
        cow_explorer_url = (
            f'https://explorer.cow.fi/gc/orders/{trade_data[0]["order_uid"]}'
        )
        cow_explorer_url = f"https://explorer.cow.fi/gc/tx/{txn_hash}"
    else:
        cow_explorer_url = (
            f'https://explorer.cow.fi/orders/{trade_data[0]["order_uid"]}'
        )
        cow_explorer_url = f"https://explorer.cow.fi/tx/{txn_hash}"

    ethtx_explorer_url = f"https://ethtx.info/mainnet/{txn_hash}"
    eigen_url = f"https://eigenphi.io/mev/eigentx/{txn_hash}"
    ts = txn_receipt.provider.chain_manager.blocks[txn_receipt.block_number].timestamp
    index = get_index_in_block(txn_receipt)
    index = index if index != 1_000_000 else "???"

    xyzscan_base_url = etherscan_base_url if not is_gnosis else gnosisscan_base_url

    dt = datetime.utcfromtimestamp(ts).strftime("%m/%d %H:%M")
    msg = "🇪🇹" if not is_gnosis else "🦉️"
    msg += f'{"🧜‍♂️" if solver in prod_solvers else "🐓"} *New solve detected!*\n'
    msg += f"by [{solver[0:7]}...]({xyzscan_base_url}address/{solver})  index: {index} @ {dt}\n\n"
    msg += f"📕 *Trade(s)*:\n"
    for t in trade_data:
        user = t["owner"]
        sell_amt = round(t["sell_amount"] / 10 ** t["sell_token_decimals"], 4)
        buy_amt = round(t["buy_amount"] / 10 ** t["buy_token_decimals"], 4)
        buy_token = t["buy_token_address"]
        if buy_token.lower() == NATIVE_TOKEN_ADDR.lower():
            buy_token = WRAPPED_NATIVE_TOKEN_ADDR[txn_receipt.chain_id]
        msg += f'    [{t["sell_token_symbol"]}]({xyzscan_base_url}token/{t["sell_token_address"]}) {sell_amt:,} -> [{t["buy_token_symbol"]}]({xyzscan_base_url}token/{t["buy_token_address"]}) {buy_amt:,} | [{user[0:7]}...]({xyzscan_base_url}address/{user})\n'

    if len(slippages) != 0:
        if sum([slippage_d["th"] for slippage_d in slippages.values()]) != 0:
            msg += "\n✂️ *TH Slippages*"
            for key in slippages:
                token = _token_info(key, txn_receipt.chain_id)
                slippage = slippages[key]["th"]
                if slippage == 0:
                    continue
                color = "🔴" if slippage < 0 else "🟢"
                amount = slippage / 10 ** token.decimals
                amount = ("{0:,.4f}" if amount > 1e-5 else "{0:.4}").format(amount)
                msg += f"\n   {color} {token.symbol}: {amount}"

        if sum([slippage_d["cow"] for slippage_d in slippages.values()]) != 0:
            msg += "\n✂️ *Cow Slippages*"
            for key in slippages:
                token = _token_info(key, txn_receipt.chain_id)
                slippage = slippages[key]["cow"]
                if slippage == 0:
                    continue
                color = "🔴" if slippage < 0 else "🟢"
                amount = slippage / 10 ** token.decimals
                amount = ("{0:,.4f}" if amount > 1e-5 else "{0:.4}").format(amount)
                msg += f"\n   {color} {token.symbol}: {amount}"

        msg += "\n"

    msg += f"\n{calc_gas_cost(txn_receipt)}"
    if is_gnosis:
        msg += f"\n\n🔗 [Gnosisscan]({xyzscan_base_url}tx/{txn_hash}) | [Cow]({cow_explorer_url})"
    else:
        msg += f"\n\n🔗 [Etherscan]({xyzscan_base_url}tx/{txn_hash}) | [Cow]({cow_explorer_url}) | [Eigen]({eigen_url}) | [EthTx]({ethtx_explorer_url})"

    return msg


def calc_gas_cost(txn_receipt: ReceiptAPI):
    provider = txn_receipt.provider
    eth_used = txn_receipt.gas_price * txn_receipt.gas_used

    if txn_receipt.transaction.chain_id == 100:
        return f"💸 ${eth_used/1e18:,.4f} DAI"
    oracle = provider.project_manager.ORACLE.at(
        "0x83d95e0D5f402511dB06817Aff3f9eA88224B030"
    )

    gas_cost = oracle.getNormalizedValueUsdc(WETH_ADDR, eth_used) / 10 ** 6
    return f"💸 ${gas_cost:,.2f} | {eth_used/1e18:,.4f} ETH"


def process_revert(txn_hash: str, chain_id: int) -> None | str:
    with networks.parse_network_choice(APE_NETWORK_STRING[chain_id]) as provider:
        txn_receipt = provider.get_receipt(txn_hash)

        failed = txn_receipt.failed
        sender = txn_receipt.transaction.sender
        print(sender)
        if not failed or sender not in solvers:
            return

        is_gnosis = txn_receipt.chain_id == 100
        xyzscan_base_url = etherscan_base_url if not is_gnosis else gnosisscan_base_url
        tenderly_base_url = (
            "https://dashboard.tenderly.co/tx/mainnet/"
            if not is_gnosis
            else "https://dashboard.tenderly.co/tx/gnosis-chain/"
        )

        msg = f"*🤬  Failed Transaction detected!*\n\n"
        e = "🧜‍♂️" if sender in prod_solvers else "🐓"
        _, _, markdown = abbreviate_address(sender)
        msg += f"Sent from {markdown} {e}\n\n"
        msg += f"{calc_gas_cost(txn_receipt)}"
        msg += f"\n\n🔗 [{'Ether' if not is_gnosis else 'Gnosis'}scan]({xyzscan_base_url}tx/{txn_hash}) | [Tenderly]({tenderly_base_url}{txn_hash})"
        return msg


def get_index_in_block(tx: ReceiptAPI):
    hashes = [
        x.txn_hash.hex()
        for x in tx.provider.chain_manager.blocks[tx.block_number].transactions
    ]
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
        chat_ids = [CHAT_IDS["SEASOLVER"]]  # , CHAT_IDS["SEASOLVER_SA"]]
    else:
        chat_ids = [CHAT_IDS["FP_ALERTS"]]
    return await asyncio.wait(
        [
            bot.send_message(
                chat_id, msg, parse_mode="markdown", disable_web_page_preview=True
            )
            for chat_id in chat_ids
        ]
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
def _token_info(addr: str, chain_id: int = 1) -> TokenInfo:
    with networks.parse_network_choice(APE_NETWORK_STRING[chain_id]) as provider:
        token: ContractInstance = provider.project_manager.ERC20.at(addr)
        if addr == NATIVE_TOKEN_ADDR:
            if chain_id == 100:
                return TokenInfo(addr, "XDAI", 18, None)
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


@lru_cache()
def get_http_session() -> ClientSession:
    client_session = ClientSession()
    retries = Retry(
        total=2,
        status_forcelist=[400, 403, 429, 500, 503],
    )
    client_session.mount("http://", HTTPAdapter(max_retries=retries))
    return client_session


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("__main__:app", host="0.0.0.0", loop="none")
